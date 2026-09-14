"""Rendering. Every `print` in the package comes from here.

Keeping the text separate from the computation is what lets the rest of the
package be tested without capturing stdout — and it is why the Python API can
hand you the same summary the CLI shows.
"""

from __future__ import annotations

from evalix.diff import Diff
from evalix.scoring import Run, mean, passed


def _fmt(value) -> str:
    return "—" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def _one_line(text: str, width: int) -> str:
    return " ".join(text.split())[:width]


def render_case_line(result) -> str:
    mark = "·" if result.score is None else ("✓" if result.passed else "✗")
    score = "  — " if result.score is None else f"{result.score:.2f}"
    return f"  {mark} {result.id:<10} {score}  {result.note[:88]}"


def render_run(run: Run, diff: Diff | None = None, path=None, show: int = 5) -> str:
    lines: list[str] = []
    results = run.results
    scored = [r for r in results if r.score is not None]
    avg = mean(results)

    lines.append("")
    lines.append(f"  score      {_fmt(avg)}   ({passed(results)}/{len(scored)} perfect)")

    tok_in = sum(r.input_tokens for r in results)
    tok_out = sum(r.output_tokens for r in results)
    scorer_in = sum(r.scorer_input_tokens for r in results)
    scorer_out = sum(r.scorer_output_tokens for r in results)
    cost = run.meta.get("cost_usd")
    tokens = f"  tokens     {tok_in} in / {tok_out} out"
    if scorer_in or scorer_out:
        # A judge often costs more than the run it grades, so it gets its own
        # line and the cost has to say it includes both.
        lines.append(tokens)
        lines.append(f"  scorer     {scorer_in} in / {scorer_out} out")
        if cost is not None:
            lines.append(f"  cost       ≈ ${cost:.4f}   (run + scorer)")
    else:
        if cost is not None:
            tokens += f"   ≈ ${cost:.4f}"
        lines.append(tokens)

    errors = [r for r in results if r.error]
    if errors:
        kinds = ", ".join(sorted({r.error for r in errors if r.error}))
        lines.append(f"  errors     {len(errors)} ({kinds})")

    if run.meta.get("scorer") == "json_fields" and results:
        unparseable = sum(1 for r in results if r.note.startswith("UNPARSEABLE"))
        rate = (len(results) - unparseable) / len(results)
        lines.append(f"  parse rate {rate:.3f}   (format vs. content)")

    if diff is not None:
        lines.append("")
        lines.extend(render_diff_body(diff, prefix="  "))

    losers = sorted((r for r in scored if not r.passed), key=lambda r: r.score or 0.0)
    if losers and show:
        lines.append("")
        lines.append(f"  worst {min(show, len(losers))}:")
        for r in losers[:show]:
            lines.append(f"    {r.id} ({r.score:.2f}) {r.note[:70]}")
            lines.append(f"        → {_one_line(r.output, 160)}")

    if path is not None:
        lines.append("")
        lines.append(f"  saved {path}")
    return "\n".join(lines)


def render_diff_body(diff: Diff, prefix: str = "  ") -> list[str]:
    lines: list[str] = []
    for warning in diff.warnings:
        lines.append(f"{prefix}⚠ {warning}")

    if not diff.comparable:
        lines.append(f"{prefix}no shared case ids — nothing to compare")
        return lines

    delta = diff.delta
    delta_text = "" if delta is None else f"  ({delta:+.3f})"
    lines.append(
        f"{prefix}{_fmt(diff.old_mean)} → {_fmt(diff.new_mean)}{delta_text}"
        f"   over {len(diff.changes)} shared cases"
    )
    if diff.fixed:
        lines.append(f"{prefix}  fixed  {', '.join(diff.fixed)}")
    if diff.broke:
        lines.append(f"{prefix}  BROKE  {', '.join(diff.broke)}")
    if not diff.fixed and not diff.broke:
        lines.append(f"{prefix}  no per-case changes")
    return lines


def render_comparison(
    old_path, new_path, old: Run, new: Run, diff: Diff, show: int = 5, show_all: bool = False
) -> str:
    fields = ("label", "model", "effort", "placement", "scorer", "prompt", "cases", "timestamp")
    lines = [f"  old  {old_path.name}", f"  new  {new_path.name}", ""]

    for field_name in fields:
        a, b = _fmt(old.meta.get(field_name)), _fmt(new.meta.get(field_name))
        lines.append(f"    {field_name:<10} {a}" if a == b else f"  ≠ {field_name:<10} {a}  →  {b}")

    for warning in diff.warnings:
        lines.append("")
        lines.append(f"  ⚠ {warning}")

    if not diff.comparable:
        lines.append("")
        lines.append("  the two runs share no case ids (--only-tag or --repeat mismatch?)")
        return "\n".join(lines)

    delta = diff.delta
    delta_text = "" if delta is None else f"  ({delta:+.3f})"
    lines.append("")
    lines.append(
        f"  score      {_fmt(diff.old_mean)} → {_fmt(diff.new_mean)}{delta_text}"
        f"   over {len(diff.changes)} shared cases"
    )
    if diff.only_old:
        lines.append(f"  only old   {', '.join(diff.only_old)}")
    if diff.only_new:
        lines.append(f"  only new   {', '.join(diff.only_new)}")

    interesting = [c for c in diff.changes if c.kind != "same"]
    lines.append("")
    lines.append(
        f"  {len(interesting)} of {len(diff.changes)} cases changed"
        f"   (fixed {len(diff.fixed)} · broke {len(diff.broke)})"
    )
    for change in diff.changes if show_all else interesting:
        mark = {"fixed": "✓", "BROKE": "✗"}.get(change.kind, " ")
        before = "  — " if change.before is None else f"{change.before:.2f}"
        after = "  — " if change.after is None else f"{change.after:.2f}"
        lines.append(
            f"    {mark} {change.kind:<8} {change.id:<10} {(change.tag or '—'):<6} {before} → {after}"
        )

    if diff.by_tag:
        lines.append("")
        lines.append(f"    {'tag':<8} {'old':>7} {'new':>8} {'delta':>9} {'n':>4}")
        for tag, (a, b, n) in diff.by_tag.items():
            d = "" if a is None or b is None else f"{b - a:>+9.3f}"
            lines.append(f"    {tag:<8} {_fmt(a):>7} {_fmt(b):>8} {d:>9} {n:>4}")
        a, b = diff.old_mean, diff.new_mean
        d = "" if a is None or b is None else f"{b - a:>+9.3f}"
        lines.append(f"    {'(all)':<8} {_fmt(a):>7} {_fmt(b):>8} {d:>9} {len(diff.changes):>4}")

    regressions = [c for c in diff.changes if c.kind in ("BROKE", "worse")]
    if regressions and show:
        old_by, new_by = old.by_id, new.by_id
        lines.append("")
        lines.append(f"  regressions ({min(show, len(regressions))} of {len(regressions)}):")
        for change in regressions[:show]:
            lines.append(
                f"    {change.id}  {change.before:.2f} → {change.after:.2f}"
                f"   {new_by[change.id].note[:70]}"
            )
            lines.append(f"        old → {_one_line(old_by[change.id].output, 120)}")
            lines.append(f"        new → {_one_line(new_by[change.id].output, 120)}")

    return "\n".join(lines)
