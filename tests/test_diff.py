"""The diff — one implementation, and the bug that came from having two.

The failure mode being pinned here: if an id missing from the previous run is
treated as a zero, every new case is reported as a win you did not earn.
"""

from __future__ import annotations

from evalix import Run, diff
from evalix.scoring import Result


def make(meta=None, **scores):
    return Run(
        meta={"case_key": "k", "scorer": "exact", "model": "m", **(meta or {})},
        results=[Result(id=cid, score=s, tag="edge") for cid, s in scores.items()],
    )


class TestClassification:
    def test_fixed_and_broke(self):
        d = diff(make(a=0.0, b=1.0), make(a=1.0, b=0.0))
        assert d.fixed == ["a"]
        assert d.broke == ["b"]

    def test_partial_movement_is_not_fixed(self):
        d = diff(make(a=0.2), make(a=0.6))
        assert d.better == ["a"]
        assert d.fixed == []

    def test_same(self):
        assert diff(make(a=1.0), make(a=1.0)).same == ["a"]

    def test_unscored_on_either_side(self):
        d = diff(make(a=None), make(a=1.0))
        assert d.changes[0].kind == "unscored"
        assert d.fixed == []


class TestMismatchedIds:
    def test_ids_only_in_the_new_run_are_not_counted_as_fixed(self):
        """The regression this package exists to not repeat."""
        d = diff(make(a=1.0), make(a=1.0, b=1.0))
        assert d.fixed == []
        assert d.only_new == ["b"]

    def test_ids_only_in_the_old_run_are_not_counted_as_broken(self):
        d = diff(make(a=1.0, b=1.0), make(a=1.0))
        assert d.broke == []
        assert d.only_old == ["b"]

    def test_means_use_shared_cases_only(self):
        """Otherwise a run that merely covered fewer cases looks like progress."""
        d = diff(make(a=0.0, b=0.0), make(a=1.0))
        assert (d.old_mean, d.new_mean) == (0.0, 1.0)
        assert d.delta == 1.0

    def test_mismatch_is_warned_about(self):
        d = diff(make(a=1.0), make(a=1.0, b=1.0))
        assert any("only in the new" in w for w in d.warnings)

    def test_no_overlap_at_all(self):
        d = diff(make(a=1.0), make(b=1.0))
        assert not d.comparable
        assert d.delta is None


class TestWarnings:
    def test_different_case_files(self):
        d = diff(make({"case_key": "one"}, a=1.0), make({"case_key": "two"}, a=1.0))
        assert any("not comparable" in w for w in d.warnings)

    def test_different_scorers(self):
        d = diff(make({"scorer": "exact"}, a=1.0), make({"scorer": "contains"}, a=1.0))
        assert any("different scales" in w for w in d.warnings)

    def test_two_axes_at_once_is_unattributable(self):
        old = make({"model": "haiku", "prompt": "v1.txt"}, a=1.0)
        new = make({"model": "opus", "prompt": "v2.txt"}, a=1.0)
        assert any("unattributable" in w for w in diff(old, new).warnings)

    def test_one_axis_is_fine(self):
        old = make({"model": "haiku", "prompt": "v1.txt"}, a=1.0)
        new = make({"model": "haiku", "prompt": "v2.txt"}, a=1.0)
        assert not any("unattributable" in w for w in diff(old, new).warnings)


class TestByTag:
    def test_groups_and_counts(self):
        old = Run(meta={"case_key": "k"}, results=[
            Result(id="a", score=0.0, tag="easy"), Result(id="b", score=0.0, tag="edge")])
        new = Run(meta={"case_key": "k"}, results=[
            Result(id="a", score=1.0, tag="easy"), Result(id="b", score=0.0, tag="edge")])
        d = diff(old, new)
        assert d.by_tag["easy"] == (0.0, 1.0, 1)
        assert d.by_tag["edge"] == (0.0, 0.0, 1)


def test_compare_returns_named_fields(tmp_path):
    from evalix import Comparison, compare
    from evalix.store import RunStore

    store = RunStore(tmp_path)
    store.save(Run(meta={"case_key": "k", "label": "old"}, results=[Result(id="a", score=0.0)]))
    store.save(Run(meta={"case_key": "k", "label": "new"}, results=[Result(id="a", score=1.0)]))
    result = compare("old", "new", runs=tmp_path)
    assert isinstance(result, Comparison)
    assert result.diff.fixed == ["a"]
    assert result.old.meta["label"] == "old"
    assert "fixed 1" in result.render()
