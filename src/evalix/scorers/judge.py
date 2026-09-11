"""LLM-as-judge: a scorer that calls a model.

A judge is just another prompt, and it can be wrong in ways a string
comparison cannot. Audit it against your own labels before trusting a number
it produced.

It reaches the model through `ctx.runner`, the same seam the run itself uses,
so a judge on a different provider is a `judge_runner` in the config rather
than a second integration.
"""

from __future__ import annotations

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

<output_to_grade>
{output}
</output_to_grade>

Reply with only a JSON object: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""


def judge(output: str, case, ctx: Context) -> Score:
    rubric = ctx.config.get("rubric")
    if not rubric:
        raise ValueError("the judge scorer needs a rubric (config key 'rubric')")
    rubric_text = Path(rubric).read_text() if Path(rubric).exists() else str(rubric)

    runner = ctx.config.get("judge_runner") or ctx.runner
    request = Request(
        messages=[
            Message(
                role="user",
                content=TEMPLATE.format(rubric=rubric_text, input=case.input, output=output),
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
        raw, reason = int(parsed["score"]), parsed.get("reason", "")
    except (ValueError, KeyError, TypeError):
        # Judges quote the graded output inside their own `reason`, producing an
        # unescaped quote and invalid JSON. The score is the part we need, so dig
        # it out before giving up — otherwise a cosmetic slip scores the case 0
        # and quietly poisons the eval.
        match = re.search(r'"score"\s*:\s*([1-5])', text)
        if not match:
            return Score(0.0, f"judge returned unparseable verdict: {' '.join(text.split())[:150]}")
        raw, reason = int(match.group(1)), "(reason unparseable)"

    return Score((raw - 1) / 4.0, f"judge {raw}/5: {reason}")
