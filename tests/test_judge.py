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


@pytest.mark.parametrize("verdict", ['{"score": 10}', '{"score": 0}', '{"score": 4.5}',
                                     '{"score": true}', '{"score": "high"}'])
def test_an_out_of_range_score_is_not_clamped(verdict):
    """10 used to become 2.25 and 0 became -0.25, quietly skewing the mean."""
    score = judge("out", CASE, ctx_with(verdict))
    assert score.value == 0.0
    assert "out of range" in score.note


@pytest.mark.parametrize("broken", ['{"score": 10, "reason": "said "hi""}',
                                    '{"score": 4.5, "reason": "said "hi""}'])
def test_the_invalid_json_fallback_does_not_misread_multi_digit_scores(broken):
    assert "unparseable verdict" in judge("out", CASE, ctx_with(broken)).note


def test_accepts_an_integral_float_or_string():
    assert judge("out", CASE, ctx_with('{"score": 4.0}')).value == 0.75
    assert judge("out", CASE, ctx_with('{"score": "4"}')).value == 0.75


def test_sends_expected_as_a_reference_answer():
    seen = []

    def judge_runner(request):
        seen.append(request.text)
        return Response(text='{"score": 5}')

    ctx = Context(runner=judge_runner, model="t", config={"rubric": "r"})
    judge("out", Case(id="d1", input="q", expected={"total": 42}), ctx)
    judge("out", Case(id="d2", input="q"), ctx)
    assert '<reference_answer>\n{"total": 42}\n</reference_answer>' in seen[0]
    assert "reference_answer" not in seen[1]


def test_a_long_inline_rubric_is_not_mistaken_for_a_path():
    assert judge("out", CASE, ctx_with('{"score": 5}', rubric="x" * 5000)).value == 1.0
