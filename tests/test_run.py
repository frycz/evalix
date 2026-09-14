"""The run loop. `runner=` is a fake throughout, so none of this costs anything."""

from __future__ import annotations

import time

import pytest

from evalix import (
    Case,
    Context,
    Message,
    MissingCredentials,
    Refusal,
    Request,
    Response,
    RunnerError,
    ScorerError,
    run,
    run_case,
)
from evalix.scorers import builtin
from evalix.scorers.judge import judge as builtin_judge

CASE = Case(id="t1", input="x", expected="billing")


def go(runner, scorer=builtin.exact, **kwargs):
    return run_case(
        CASE, "", runner=runner, scorer=scorer, ctx=Context(runner=runner, model="m"), **kwargs
    )


class TestRunCase:
    def test_scores_a_successful_call(self, runner):
        result = go(runner("billing"))
        assert result.score == 1.0
        assert (result.input_tokens, result.output_tokens) == (11, 7)
        assert result.latency is not None

    def test_refusal_scores_zero_without_killing_the_run(self, runner):
        result = go(runner(error=Refusal("nope")))
        assert result.score == 0.0
        assert result.error == "refusal"

    def test_transport_error_is_recorded_not_raised(self, runner):
        result = go(runner(error=RunnerError("503")))
        assert result.score == 0.0
        assert result.error == "runner"

    def test_unexpected_runner_error_is_recorded(self, runner):
        """A runner someone else wrote may raise anything."""
        result = go(runner(error=ZeroDivisionError()))
        assert result.score == 0.0
        assert result.error == "ZeroDivisionError"

    def test_missing_credentials_stops_the_run(self, runner):
        """One bad case is survivable; no API key is not."""
        with pytest.raises(MissingCredentials):
            go(runner(error=MissingCredentials()))


class TestScorerFailure:
    def boom(self, output, case, ctx):
        raise ZeroDivisionError("bad scorer")

    def test_is_fatal_by_default(self, runner):
        """A broken scorer otherwise reports a clean 0.000 that looks exactly
        like a failing prompt."""
        with pytest.raises(ScorerError, match="scorer raised on case t1"):
            go(runner("billing"), scorer=self.boom)

    def test_keep_going_scores_zero_and_keeps_the_output(self, runner):
        result = go(runner("billing"), scorer=self.boom, keep_going=True)
        assert result.score == 0.0
        assert result.output == "billing"
        assert "SCORER ERROR" in result.note


class TestRun:
    def test_scores_every_case_and_keeps_order(self, cases_file, runner, tmp_path):
        p = cases_file([{"id": f"t{i}", "input": "x", "expected": "a"} for i in range(5)])
        report = run(p, runner=runner("a"), model="test", runs=tmp_path / "runs")
        assert report.score == 1.0
        assert [r.id for r in report.results] == [f"t{i}" for i in range(5)]

    def test_writes_a_run_file(self, cases_file, runner, tmp_path):
        p = cases_file([{"id": "t1", "input": "x", "expected": "a"}])
        report = run(p, runner=runner("a"), model="test", runs=tmp_path / "runs")
        assert report.path is not None and report.path.exists()
        assert report.meta["case_key"]

    def test_save_false_writes_nothing(self, cases_file, runner, tmp_path):
        p = cases_file([{"id": "t1", "input": "x", "expected": "a"}])
        report = run(p, runner=runner("a"), model="test", save=False, runs=tmp_path / "runs")
        assert report.path is None
        assert not (tmp_path / "runs").exists()

    def test_diffs_against_the_previous_run(self, cases_file, runner, tmp_path):
        p = cases_file([{"id": "t1", "input": "x", "expected": "a"}])
        runs = tmp_path / "runs"
        run(p, runner=runner("wrong"), model="test", runs=runs)
        report = run(p, runner=runner("a"), model="test", runs=runs)
        assert report.diff is not None
        assert report.diff.fixed == ["t1"]

    def test_unscored_cases_drop_out_of_the_mean(self, cases_file, runner, tmp_path):
        p = cases_file([{"id": "t1", "input": "x"}, {"id": "t2", "input": "y"}])
        report = run(p, runner=runner("a"), scorer="none", model="test", runs=tmp_path / "runs")
        assert report.score is None

    def test_empty_case_set_is_loud(self, cases_file, runner, tmp_path):
        p = cases_file([{"id": "t1", "tag": "easy"}])
        with pytest.raises(ValueError, match="no cases"):
            run(p, runner=runner(), only_tag="edge", runs=tmp_path / "runs")

    def test_render_does_not_crash(self, cases_file, runner, tmp_path):
        p = cases_file([{"id": "t1", "input": "x", "expected": "a"}])
        report = run(p, runner=runner("nope"), model="test", runs=tmp_path / "runs")
        assert "score" in report.render()


    def test_a_scorer_error_cancels_the_cases_not_yet_started(self, cases_file, tmp_path):
        """Leaving the pool used to wait for every queued case, so a scorer that
        failed on case 1 still paid for all 40 calls."""
        calls = []

        def slow(request):
            calls.append(request)
            time.sleep(0.02)
            return Response(text="a")

        def boom(output, case, ctx):
            raise ZeroDivisionError

        p = cases_file([{"id": f"t{i}", "input": "x"} for i in range(40)])
        with pytest.raises(ScorerError):
            run(p, runner=slow, scorer=boom, model="test", workers=2, runs=tmp_path / "runs")
        assert len(calls) < 10


class TestScorerCalls:
    """A judge makes model calls of its own. They cost money and they can fail."""

    def judge_ctx_runner(self, tokens=(100, 20), error=None):
        def run_(request):
            if error is not None:
                raise error
            return Response(text='{"score": 5}', input_tokens=tokens[0], output_tokens=tokens[1])

        return run_

    def test_scorer_tokens_are_counted_and_priced(self, cases_file, tmp_path):
        def judge_like(output, case, ctx):
            ctx.runner(Request(messages=[Message("user", "grade")], model="claude-opus-5"))
            return 1.0

        runner = self.judge_ctx_runner()
        p = cases_file([{"id": "t1", "input": "x"}, {"id": "t2", "input": "y"}])
        report = run(p, runner=runner, scorer=judge_like, model="claude-haiku-4-5",
                     runs=tmp_path / "runs")
        assert report.meta["input_tokens"] == 200
        assert report.meta["scorer_input_tokens"] == 200
        assert report.meta["scorer_output_tokens"] == 40
        run_cost = (200 * 1.0 + 40 * 5.0) / 1_000_000
        judge_cost = (200 * 5.0 + 40 * 25.0) / 1_000_000
        assert report.cost_usd == pytest.approx(run_cost + judge_cost)
        assert "(run + scorer)" in report.render()

    def test_a_separate_judge_runner_is_counted_too(self, cases_file, runner, tmp_path):
        p = cases_file([{"id": "t1", "input": "x"}])
        report = run(
            p, runner=runner("out"), scorer="judge", model="claude-haiku-4-5",
            scorer_config={"rubric": "r", "judge_model": "claude-opus-5",
                           "judge_runner": self.judge_ctx_runner()},
            runs=tmp_path / "runs",
        )
        assert report.results[0].scorer_input_tokens == 100

    def test_an_unpriced_scorer_call_makes_the_cost_unknown(self, cases_file, tmp_path):
        def judge_like(output, case, ctx):
            ctx.runner(Request(messages=[Message("user", "grade")], model="mystery-model"))
            return 1.0

        p = cases_file([{"id": "t1", "input": "x"}])
        report = run(p, runner=self.judge_ctx_runner(), scorer=judge_like,
                     model="claude-haiku-4-5", runs=tmp_path / "runs")
        assert report.cost_usd is None

    def test_a_failed_judge_call_is_recorded_not_fatal(self, cases_file, runner, tmp_path):
        p = cases_file([{"id": "t1", "input": "x"}, {"id": "t2", "input": "y"}])
        report = run(
            p, runner=runner("out"), scorer="judge", model="test",
            scorer_config={"rubric": "r", "judge_runner": self.judge_ctx_runner(
                error=RunnerError("529 overloaded"))},
            runs=tmp_path / "runs",
        )
        assert [r.error for r in report.results] == ["scorer_call", "scorer_call"]
        assert all(r.score is None for r in report.results)

    def test_missing_credentials_in_the_judge_still_stops_the_run(self, runner):
        judge_runner = self.judge_ctx_runner(error=MissingCredentials("no key"))
        ctx = Context(runner=runner("out"), model="m",
                      config={"rubric": "r", "judge_runner": judge_runner})
        with pytest.raises(MissingCredentials):
            run_case(CASE, "", runner=runner("out"), scorer=builtin_judge, ctx=ctx)
