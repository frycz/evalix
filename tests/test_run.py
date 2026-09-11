"""The run loop. `runner=` is a fake throughout, so none of this costs anything."""

from __future__ import annotations

import pytest

from evalix import Case, Context, MissingCredentials, Refusal, RunnerError, ScorerError, run, run_case
from evalix.scorers import builtin

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
