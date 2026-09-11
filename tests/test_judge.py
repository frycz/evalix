"""The judge is a scorer that calls a model — through the same seam as the run.

These tests pass a fake runner, which is the whole point: a scorer that can
reach a model is testable without reaching one.
"""

from __future__ import annotations

import pytest

from evalix import Case, Context, Response
from evalix.scorers.judge import judge


def ctx_with(text, rubric="be strict"):
    return Context(
        runner=lambda request: Response(text=text),
        model="test",
        config={"rubric": rubric, "judge_model": "test-judge"},
    )


CASE = Case(id="d1", input="some document")


def test_maps_five_point_scale_onto_zero_to_one():
    assert judge("out", CASE, ctx_with('{"score": 5, "reason": "great"}')).value == 1.0
    assert judge("out", CASE, ctx_with('{"score": 1, "reason": "bad"}')).value == 0.0
    assert judge("out", CASE, ctx_with('{"score": 3, "reason": "ok"}')).value == 0.5


def test_recovers_a_score_from_invalid_json():
    """Judges quote the output in `reason`, breaking their own JSON. The score
    is the part we need, so a cosmetic slip must not silently score zero."""
    broken = '{"score": 4, "reason": "it said "hello" which is fine"}'
    score = judge("out", CASE, ctx_with(broken))
    assert score.value == 0.75
    assert "unparseable" in score.note


def test_gives_up_loudly_when_there_is_no_score():
    score = judge("out", CASE, ctx_with("I cannot grade this."))
    assert score.value == 0.0
    assert "unparseable verdict" in score.note


def test_requires_a_rubric():
    ctx = Context(runner=lambda r: Response(text=""), model="t", config={})
    with pytest.raises(ValueError, match="rubric"):
        judge("out", CASE, ctx)


def test_uses_a_separate_judge_runner_when_given():
    seen = []

    def judge_runner(request):
        seen.append(request)
        return Response(text='{"score": 5}')

    ctx = Context(
        runner=lambda r: Response(text='{"score": 1}'),
        model="t",
        config={"rubric": "r", "judge_runner": judge_runner},
    )
    assert judge("out", CASE, ctx).value == 1.0
    assert len(seen) == 1
