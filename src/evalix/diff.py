"""Comparing two runs — the part of this tool that justifies its existence.

The score tells you *whether*; the diff tells you *which*. A prompt change that
gains three cases and loses two is a `+1` you would otherwise call an
improvement, and only a per-case diff shows you the two you broke.

There is exactly one implementation, used both by `run` (against the previous
run of the same case file) and by `compare` (against any two runs). The bug it
replaces came from having two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evalix.scoring import Result, Run
from evalix.scorers import PASS

AXES = ("model", "effort", "placement", "prompt")


@dataclass(frozen=True)
class CaseChange:
    id: str
    tag: str | None
    before: float | None
    after: float | None
    kind: str  # fixed | BROKE | better | worse | same | unscored


@dataclass
class Diff:
    fixed: list[str] = field(default_factory=list)
    broke: list[str] = field(default_factory=list)
    better: list[str] = field(default_factory=list)
    worse: list[str] = field(default_factory=list)
    same: list[str] = field(default_factory=list)
    unscored: list[str] = field(default_factory=list)
    only_old: list[str] = field(default_factory=list)
    only_new: list[str] = field(default_factory=list)
    changes: list[CaseChange] = field(default_factory=list)
    old_mean: float | None = None
    new_mean: float | None = None
    by_tag: dict[str, tuple[float | None, float | None, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def delta(self) -> float | None:
        if self.old_mean is None or self.new_mean is None:
            return None
        return self.new_mean - self.old_mean

    @property
    def comparable(self) -> bool:
        return bool(self.changes)


def _mean(results: dict[str, Result], ids: list[str]) -> float | None:
    scores = [results[i].score for i in ids if results[i].score is not None]
    return sum(scores) / len(scores) if scores else None


def diff(old: Run, new: Run, pass_at: float = PASS) -> Diff:
    old_by, new_by = old.by_id, new.by_id
    common = sorted(set(old_by) & set(new_by))

    out = Diff(
        only_old=sorted(set(old_by) - set(new_by)),
        only_new=sorted(set(new_by) - set(old_by)),
    )
    out.warnings = _warn(old, new, out)
    if not common:
        return out

    buckets = {
        "fixed": out.fixed,
        "BROKE": out.broke,
        "better": out.better,
        "worse": out.worse,
        "same": out.same,
        "unscored": out.unscored,
    }
    for cid in common:
        before, after = old_by[cid].score, new_by[cid].score
        if before is None or after is None:
            kind = "unscored"
        elif before < pass_at <= after:
            kind = "fixed"
        elif after < pass_at <= before:
            kind = "BROKE"
        elif after > before:
            kind = "better"
        elif after < before:
            kind = "worse"
        else:
            kind = "same"
        buckets[kind].append(cid)
        out.changes.append(
            CaseChange(cid, new_by[cid].tag or old_by[cid].tag, before, after, kind)
        )

    # Means are over shared cases only. Averaging over a different denominator
    # than the one you are comparing against is how a run that merely covered
    # fewer cases looks like an improvement.
    out.old_mean = _mean(old_by, common)
    out.new_mean = _mean(new_by, common)

    tags = sorted({c.tag for c in out.changes if c.tag})
    for tag in tags:
        ids = [c.id for c in out.changes if c.tag == tag]
        out.by_tag[tag] = (_mean(old_by, ids), _mean(new_by, ids), len(ids))

    return out


def _warn(old: Run, new: Run, partial: Diff) -> list[str]:
    """The opinions live here, not in the renderer, so the Python API gets them too."""
    warnings: list[str] = []

    if old.meta.get("case_key") != new.meta.get("case_key"):
        warnings.append("different case files — these runs are not comparable")

    if old.meta.get("scorer") != new.meta.get("scorer"):
        warnings.append("different scorers — the scores are on different scales")

    changed = [a for a in AXES if old.meta.get(a) != new.meta.get(a)]
    if len(changed) > 1:
        warnings.append(
            f"{len(changed)} axes changed at once ({', '.join(changed)}) — "
            "the delta is unattributable"
        )

    if partial.only_old or partial.only_new:
        warnings.append(
            f"{len(partial.only_old)} case(s) only in the old run, "
            f"{len(partial.only_new)} only in the new — scored separately, not as fixed/broke"
        )

    return warnings
