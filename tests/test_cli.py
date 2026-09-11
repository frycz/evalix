"""The CLI is a thin wrapper — these check the wiring, not the logic."""

from __future__ import annotations

import json

import pytest

from evalix import cli


@pytest.fixture
def cases(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"id": "t1", "input": "hello", "expected": "billing"}) + "\n")
    return path


class TestDryRun:
    def test_makes_no_api_call(self, cases, capsys, monkeypatch):
        def explode():
            raise AssertionError("--dry-run must not build a runner")

        monkeypatch.setattr("evalix.runners.default_runner", explode)
        rc = cli.main(["run", "--cases", str(cases), "--dry-run", "-m", "claude-haiku-4-5"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "nothing was sent" in out
        assert "estimated" in out

    def test_writes_no_run_file(self, cases, tmp_path):
        cli.main(["run", "--cases", str(cases), "--dry-run", "--runs-dir", str(tmp_path / "runs")])
        assert not (tmp_path / "runs").exists()

    def test_flags_the_extra_judge_calls(self, cases, tmp_path, capsys):
        rubric = tmp_path / "rubric.txt"
        rubric.write_text("be strict")
        cli.main(
            ["run", "--cases", str(cases), "--dry-run", "--scorer", "judge", "--rubric", str(rubric)]
        )
        assert "adds 1 more calls" in capsys.readouterr().out


class TestArgumentChecks:
    def test_custom_scorer_needs_a_file(self, cases, capsys):
        rc = cli.main(["run", "--cases", str(cases), "--scorer", "custom"])
        assert rc == 2
        assert "requires --scorer-file" in capsys.readouterr().err

    def test_judge_needs_a_rubric(self, cases, capsys):
        rc = cli.main(["run", "--cases", str(cases), "--scorer", "judge"])
        assert rc == 2
        assert "requires --rubric" in capsys.readouterr().err

    def test_a_bad_prompt_placeholder_exits_cleanly(self, cases, tmp_path, capsys):
        prompt = tmp_path / "p.txt"
        prompt.write_text("Classify {input}")
        rc = cli.main(["run", "--cases", str(cases), "--prompt", str(prompt), "--dry-run"])
        assert rc == 1
        assert "placement system" in capsys.readouterr().err


class TestCompare:
    def test_diffs_two_saved_runs(self, tmp_path, capsys):
        from evalix.scoring import Result, Run
        from evalix.store import RunStore

        store = RunStore(tmp_path / "runs")
        store.save(Run(meta={"case_key": "k", "label": "v1", "model": "m",
                             "timestamp": "20260101-000001"},
                       results=[Result(id="a", score=0.0)]))
        store.save(Run(meta={"case_key": "k", "label": "v2", "model": "m",
                             "timestamp": "20260101-000002"},
                       results=[Result(id="a", score=1.0)]))
        rc = cli.main(["compare", "v1", "v2", "--runs-dir", str(tmp_path / "runs")])
        out = capsys.readouterr().out
        assert rc == 0
        assert "0.000 → 1.000" in out
        assert "fixed 1" in out
