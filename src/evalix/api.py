"""The public surface: `run()`, `compare()` and `estimate()`.

The CLI is a thin wrapper over exactly these, so anything the terminal can do
is available from Python with the same arguments.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from evalix import pricing, report
from evalix.cases import Case, case_key, load_cases
from evalix.diff import Diff, diff
from evalix.prompt import build_request
from evalix.runners import MissingCredentials, Refusal, Runner, RunnerError, default_runner
from evalix.scorers import Context, Scorer, as_score
from evalix.scorers import get as get_scorer
from evalix.scorers import name_of
from evalix.scoring import Result, Run, mean
from evalix.store import RunStore, project_root, runs_dir


class ScorerError(RuntimeError):
    """A scorer raised. Fatal by default — see `keep_going`."""


@dataclass
class Report:
    """One run, its diff against the previous run, and where it was saved."""

    run: Run
    diff: Diff | None = None
    path: Path | None = None
    previous: Run | None = None

    @property
    def score(self) -> float | None:
        return self.run.score

    @property
    def results(self) -> list[Result]:
        return self.run.results

    @property
    def meta(self) -> dict[str, Any]:
        return self.run.meta

    @property
    def cost_usd(self) -> float | None:
        return self.run.meta.get("cost_usd")

    def render(self, show: int = 5) -> str:
        return report.render_run(self.run, self.diff, self.path, show=show)


@dataclass
class Estimate:
    """What a run would send, and roughly what it would cost."""

    calls: int
    input_tokens: int
    max_output_tokens: int
    model: str | None
    cost_low: float | None = None
    cost_high: float | None = None
    extra_judge_calls: int = 0
    previews: list[tuple[str, int, str]] = field(default_factory=list)

    def render(self, quiet: bool = False) -> str:
        lines: list[str] = []
        if not quiet:
            for cid, chars, first in self.previews:
                lines.append(f"  {cid:<10} ~{chars // 4:>5} tok in   {first[:70]}")
        lines.append("")
        lines.append(f"  {self.calls} calls · ~{self.input_tokens} input tokens")
        lines.append(
            f"  worst case {self.max_output_tokens} output tokens (every case hitting max-tokens)"
        )
        if self.cost_low is not None and self.cost_high is not None:
            lines.append(
                f"  estimated  ${self.cost_low:.4f} … ${self.cost_high:.4f}"
                "   (input only … input + max output)"
            )
        if self.extra_judge_calls:
            lines.append(f"  note       the judge scorer adds {self.extra_judge_calls} more calls")
        lines.append("")
        lines.append("  dry run — nothing was sent, no run file written")
        return "\n".join(lines)


def _prepare(
    cases: str | Path | list[Case],
    prompt: str | Path | None,
    only_tag: str | None,
    limit: int | None,
    repeat: int,
) -> tuple[list[Case], str, Path | None]:
    prompt_path = Path(prompt) if prompt else None
    prompt_text = prompt_path.read_text().strip() if prompt_path else ""
    if isinstance(cases, (str, Path)):
        loaded = load_cases(cases, only_tag=only_tag, limit=limit, repeat=repeat)
    else:
        loaded = list(cases)
    return loaded, prompt_text, prompt_path


def run_case(
    case: Case,
    prompt_text: str,
    *,
    runner: Runner,
    scorer: Scorer,
    ctx: Context,
    placement: str = "system",
    model: str | None = None,
    max_tokens: int = 2000,
    effort: str | None = None,
    keep_going: bool = False,
) -> Result:
    """One case, end to end. Never raises for a failed model call.

    A single bad response should not kill a 22-case run, so transport failures
    and refusals are recorded and scored zero. Two things do stop everything:
    missing credentials (every remaining case will fail the same way) and a
    scorer that raises — because a broken scorer reports a clean 0.000 that
    looks exactly like a failing prompt.
    """
    request = build_request(
        case,
        prompt_text,
        placement=placement,
        model=model,
        max_tokens=max_tokens,
        effort=effort,
    )
    started = time.time()
    try:
        response = runner(request)
    except MissingCredentials:
        raise
    except Refusal as exc:
        return Result(id=case.id, tag=case.tag, score=0.0, note=f"REFUSAL: {exc}", error="refusal")
    except RunnerError as exc:
        return Result(id=case.id, tag=case.tag, score=0.0, note=f"ERROR: {exc}", error="runner")
    except Exception as exc:  # noqa: BLE001 - a runner someone else wrote
        return Result(
            id=case.id, tag=case.tag, score=0.0, note=f"ERROR: {exc}", error=type(exc).__name__
        )

    try:
        score = as_score(scorer(response.text, case, ctx))
    except Exception as exc:  # noqa: BLE001
        if not keep_going:
            raise ScorerError(
                f"scorer raised on case {case.id}: {type(exc).__name__}: {exc}. "
                "Pass keep_going=True (--keep-going) to score these zero and continue."
            ) from exc
        score = as_score((0.0, f"SCORER ERROR: {type(exc).__name__}: {exc}"))

    return Result(
        id=case.id,
        tag=case.tag,
        score=score.value,
        note=score.note,
        output=response.text,
        latency=round(time.time() - started, 2),
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def run(
    cases: str | Path | list[Case],
    prompt: str | Path | None = None,
    *,
    scorer: str | Scorer = "exact",
    runner: Runner | None = None,
    model: str | None = None,
    placement: str = "system",
    max_tokens: int = 2000,
    effort: str | None = None,
    only_tag: str | None = None,
    limit: int | None = None,
    repeat: int = 1,
    workers: int = 8,
    label: str | None = None,
    scorer_config: dict[str, Any] | None = None,
    keep_going: bool = False,
    save: bool = True,
    runs: str | Path | None = None,
    on_result: Callable[[Result], None] | None = None,
) -> Report:
    """Score a prompt against a case file and diff it against the previous run."""
    loaded, prompt_text, prompt_path = _prepare(cases, prompt, only_tag, limit, repeat)
    if not loaded:
        raise ValueError("no cases to run (did --only-tag match nothing?)")

    provided_runner = runner is not None
    runner = runner or default_runner()
    if model is None and not provided_runner:
        from evalix.runners.capix import default_model

        model = default_model()

    scorer_fn = get_scorer(scorer)
    ctx = Context(runner=runner, model=model, config=dict(scorer_config or {}))
    pricing.load_overrides()

    results: list[Result | None] = [None] * len(loaded)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_case,
                case,
                prompt_text,
                runner=runner,
                scorer=scorer_fn,
                ctx=ctx,
                placement=placement,
                model=model,
                max_tokens=max_tokens,
                effort=effort,
                keep_going=keep_going,
            ): i
            for i, case in enumerate(loaded)
        }
        for done in concurrent.futures.as_completed(futures):
            index = futures[done]
            results[index] = done.result()
            if on_result:
                on_result(results[index])

    final: list[Result] = [r for r in results if r is not None]
    tok_in = sum(r.input_tokens for r in final)
    tok_out = sum(r.output_tokens for r in final)

    root = project_root()
    key = case_key(cases, root) if isinstance(cases, (str, Path)) else "cases"
    meta = {
        "case_key": key,
        "cases": str(cases) if isinstance(cases, (str, Path)) else f"<{len(loaded)} cases>",
        "prompt": str(prompt_path) if prompt_path else None,
        "label": label or (prompt_path.stem if prompt_path else "no-prompt"),
        "model": model,
        "effort": effort,
        "placement": placement,
        "scorer": name_of(scorer),
        "score": mean(final),
        "input_tokens": tok_in,
        "output_tokens": tok_out,
        "cost_usd": pricing.cost(model, tok_in, tok_out),
        "timestamp": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
    }
    current = Run(meta=meta, results=final)

    store = RunStore(runs_dir(runs, root))
    previous = store.previous(key)
    path = store.save(current) if save else None

    return Report(
        run=current,
        diff=diff(previous, current) if previous else None,
        path=path,
        previous=previous,
    )


def estimate(
    cases: str | Path | list[Case],
    prompt: str | Path | None = None,
    *,
    scorer: str | Scorer = "exact",
    model: str | None = None,
    placement: str = "system",
    max_tokens: int = 2000,
    only_tag: str | None = None,
    limit: int | None = None,
    repeat: int = 1,
) -> Estimate:
    """Resolve everything a run needs and price it, without sending anything.

    Token counts are a rough chars/4 guess — enough to catch a prompt that grew
    tenfold, not accurate enough to bill against.
    """
    loaded, prompt_text, _ = _prepare(cases, prompt, only_tag, limit, repeat)
    pricing.load_overrides()

    total_chars = 0
    previews: list[tuple[str, int, str]] = []
    for case in loaded:
        request = build_request(case, prompt_text, placement=placement, max_tokens=max_tokens)
        chars = len(request.text) + len(request.system or "")
        total_chars += chars
        previews.append((case.id, chars, (request.text.splitlines() or [""])[0]))

    est_in = total_chars // 4
    est_out = len(loaded) * max_tokens
    price = pricing.PRICES.get(model or "")
    low = est_in * price[0] / 1_000_000 if price else None
    high = (low + est_out * price[1] / 1_000_000) if price and low is not None else None

    return Estimate(
        calls=len(loaded),
        input_tokens=est_in,
        max_output_tokens=est_out,
        model=model,
        cost_low=low,
        cost_high=high,
        extra_judge_calls=len(loaded) if name_of(scorer) == "judge" else 0,
        previews=previews,
    )


def compare(old: str | Path, new: str | Path, *, runs: str | Path | None = None):
    """Diff any two saved runs. Returns (old_path, new_path, old, new, Diff)."""
    store = RunStore(runs_dir(runs))
    old_path, old_run = store.resolve(old)
    new_path, new_run = store.resolve(new)
    if old_path == new_path:
        raise ValueError(f"both arguments resolved to the same run ({old_path.name})")
    return old_path, new_path, old_run, new_run, diff(old_run, new_run)
