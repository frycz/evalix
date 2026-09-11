"""Case loading: the filters that shape a run, and the ids the diff depends on."""

from __future__ import annotations

import pytest

from evalix import Case, load_cases
from evalix.cases import case_key


class TestLoadCases:
    def test_skips_blanks_and_comments(self, tmp_path):
        p = tmp_path / "cases.jsonl"
        p.write_text('// a comment\n\n{"id": "a", "input": "x"}\n')
        assert len(load_cases(p)) == 1

    def test_only_tag_filters(self, cases_file):
        p = cases_file([{"id": "a", "tag": "edge"}, {"id": "b", "tag": "normal"}])
        assert [c.id for c in load_cases(p, only_tag="edge")] == ["a"]

    def test_limit_truncates(self, cases_file):
        p = cases_file([{"id": str(i)} for i in range(10)])
        assert len(load_cases(p, limit=3)) == 3

    def test_repeat_suffixes_ids(self, cases_file):
        p = cases_file([{"id": "a"}])
        assert [c.id for c in load_cases(p, repeat=3)] == ["a#1", "a#2", "a#3"]

    def test_limit_applies_after_repeat(self, cases_file):
        """--limit 2 --repeat 3 is two calls, not six. The limit is a spend cap."""
        p = cases_file([{"id": "a"}, {"id": "b"}])
        assert len(load_cases(p, repeat=3, limit=2)) == 2

    def test_missing_id_is_loud(self, cases_file):
        """Ids are the diff key. Defaulting them collapses every case onto one."""
        p = cases_file([{"input": "x"}])
        with pytest.raises(ValueError, match="no 'id'"):
            load_cases(p)

    def test_duplicate_ids_are_loud(self, cases_file):
        p = cases_file([{"id": "a"}, {"id": "a"}])
        with pytest.raises(ValueError, match="duplicate case ids"):
            load_cases(p)


class TestCase:
    def test_unknown_keys_survive(self, cases_file):
        p = cases_file([{"id": "a", "input": "x", "must_contain": ["billing"]}])
        loaded = load_cases(p)[0]
        assert loaded.get("must_contain") == ["billing"]
        assert loaded["must_contain"] == ["billing"]

    def test_reserved_keys_reachable_both_ways(self):
        c = Case(id="a", input="x", expected="y", tag="edge")
        assert c["expected"] == "y" and c.expected == "y"
        assert c.get("tag") == "edge"

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            Case(id="a")["nope"]

    def test_contains(self):
        c = Case(id="a", extra={"anchors": []})
        assert "anchors" in c and "nope" not in c


class TestCaseKey:
    def test_is_relative_to_the_project_root(self, tmp_path):
        (tmp_path / "01-exercise").mkdir()
        path = tmp_path / "01-exercise" / "cases.jsonl"
        path.write_text("")
        assert case_key(path, tmp_path) == "01-exercise-cases"

    def test_two_exercises_do_not_collide(self, tmp_path):
        keys = set()
        for name in ("01-a", "02-b"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "cases.jsonl").write_text("")
            keys.add(case_key(tmp_path / name / "cases.jsonl", tmp_path))
        assert len(keys) == 2

    def test_outside_the_root_falls_back_to_the_name(self, tmp_path):
        other = tmp_path / "elsewhere"
        other.mkdir()
        (other / "cases.jsonl").write_text("")
        assert case_key(other / "cases.jsonl", tmp_path / "root") == "cases"
