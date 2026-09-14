"""LLM-as-judge: a scorer that calls a model.

A judge is just another prompt, and it can be wrong in ways a string
comparison cannot. Audit it against your own labels before trusting a number
it produced.

It reaches the model through `ctx.runner`, the same seam the run itself uses,
so a judge on a different provider is a `judge_runner` in the config rather
than a second integration.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evalix.runners import Message, Request
from evalix.scorers.base import Context, Score

TEMPLATE = """You are grading one output against a rubric. Be strict and consistent.

<rubric>
{rubric}
</rubric>

<input>
{input}
</input>
{reference}
<output_to_grade>
{output}
</output_to_grade>

Reply with only a JSON object: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""


def judge(output: str, case, ctx: Context) -> Score:
    rubric = ctx.config.get("rubric")
    if not rubric:
        raise ValueError("the judge scorer needs a rubric (config key 'rubric')")
    rubric_text = _read_rubric(str(rubric))

    runner = ctx.config.get("judge_runner") or ctx.runner
    request = Request(
        messages=[
            Message(
                role="user",
                content=TEMPLATE.format(
                    rubric=rubric_text,
                    input=case.input,
                    reference=_reference(case.expected),
                    output=output,
                ),
            )
        ],
        model=ctx.config.get("judge_model"),
        # Generous: on a thinking-capable judge, max_tokens covers thinking too,
        # and a truncated verdict looks exactly like a judge that cannot follow
        # the format.
        max_tokens=int(ctx.config.get("judge_max_tokens", 4000)),
    )
    text = runner(request).text

    from evalix.scorers.builtin import extract_json

    try:
        parsed = extract_json(text)
    except ValueError:
        parsed = None

    if isinstance(parsed, dict) and "score" in parsed:
        raw, reason = _valid_score(parsed["score"]), parsed.get("reason", "")
        if raw is None:
            # Not clamped: a 10 from a judge that misread the scale is a broken
            # verdict, and clamping it to 5 would hide that in the mean.
            return Score(0.0, f"judge score out of range 1-5: {parsed['score']!r}")
    else:
        # Judges quote the graded output inside their own `reason`, producing an
        # unescaped quote and invalid JSON. The score is the part we need, so dig
        # it out before giving up — otherwise a cosmetic slip scores the case 0
        # and quietly poisons the eval. The lookarounds keep 10 and 4.5 out.
        match = re.search(r'"score"\s*:\s*([1-5])(?![\d.])', text)
        if not match:
            return Score(0.0, f"judge returned unparseable verdict: {' '.join(text.split())[:150]}")
        raw, reason = int(match.group(1)), "(reason unparseable)"

    return Score((raw - 1) / 4.0, f"judge {raw}/5: {reason}")


def _read_rubric(rubric: str) -> str:
    """A path to a rubric file, or the rubric itself."""
    path = Path(rubric)
    try:
        is_file = path.is_file()
    except OSError:
        # A long inline rubric is not a valid filename on every OS.
        is_file = False
    return path.read_text(encoding="utf-8") if is_file else rubric


def _valid_score(value) -> int | None:
    """An integer 1-5, or None. `4.0` and `"4"` pass; `4.5`, `10` and `true` do not."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number.is_integer() or not 1 <= number <= 5:
        return None
    return int(number)


def _reference(expected) -> str:
    """The case's `expected`, when it has one, as a reference answer to grade against."""
    if expected is None:
        return ""
    text = expected if isinstance(expected, str) else json.dumps(expected, ensure_ascii=False)
    return f"\n<reference_answer>\n{text}\n</reference_answer>\n"
