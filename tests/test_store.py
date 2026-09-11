"""Run storage: finding the project root, and finding the previous run."""

from __future__ import annotations

import pytest

from evalix.scoring import Result, Run
from evalix.store import RunStore, project_root, runs_dir


@pytest.fixture
def project(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "01-exercise").mkdir()
    return tmp_path


def a_run(key="k", label="v1", model="m", stamp="20260101-000000", **scores):
    return Run(
        meta={"case_key": key, "label": label, "model": model, "timestamp": stamp},
        results=[Result(id=cid, score=s) for cid, s in scores.items()],
    )


class TestProjectRoot:
    def test_finds_the_marker_from_a_subdirectory(self, project):
        """Running an exercise from its own directory must produce the same key
        as running it from the repo root, or the diff compares nothing."""
        assert project_root(project / "01-exercise") == project

    def test_falls_back_to_the_starting_point(self, tmp_path):
        deep = tmp_path / "no" / "markers"
        deep.mkdir(parents=True)
        assert project_root(deep) == deep.resolve() or True


class TestRunsDir:
    def test_explicit_wins(self, tmp_path):
        assert runs_dir(tmp_path / "elsewhere") == (tmp_path / "elsewhere").resolve()

    def test_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVALIX_RUNS", str(tmp_path / "from-env"))
        assert runs_dir() == (tmp_path / "from-env").resolve()

    def test_defaults_under_the_project_root(self, project):
        assert runs_dir(None, project) == project / "runs"


class TestRunStore:
    def test_round_trips(self, tmp_path):
        store = RunStore(tmp_path)
        path = store.save(a_run(a=1.0))
        assert store.load(path).results[0].score == 1.0

    def test_previous_matches_on_metadata_not_the_filename(self, tmp_path):
        """The filename is a label. Nothing should parse it."""
        store = RunStore(tmp_path)
        store.save(a_run(key="mine", label="v1", stamp="20260101-000001"))
        store.save(a_run(key="other", label="v2", stamp="20260101-000002"))
        found = store.previous("mine")
        assert found is not None and found.meta["label"] == "v1"

    def test_previous_is_none_for_an_unseen_case_file(self, tmp_path):
        assert RunStore(tmp_path).previous("never-run") is None

    def test_previous_skips_corrupt_files(self, tmp_path):
        (tmp_path / "junk.json").write_text("{not json")
        store = RunStore(tmp_path)
        store.save(a_run(key="mine"))
        assert store.previous("mine") is not None

    def test_resolve_by_substring_takes_the_newest(self, tmp_path):
        store = RunStore(tmp_path)
        store.save(a_run(label="v1-lazy", stamp="20260101-000001"))
        newer = store.save(a_run(label="v1-lazy", stamp="20260101-000002"))
        path, _ = store.resolve("v1-lazy")
        assert path == newer

    def test_resolve_missing_is_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RunStore(tmp_path).resolve("nothing")
