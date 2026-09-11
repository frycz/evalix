"""Results, runs, and turning per-case scores into numbers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from evalix.scorers import PASS


@dataclass
class Result:
    """One case, after it has been run and scored.

    `score is None` means not scored (the `none` scorer, or a case skipped for
    manual review) and drops out of every mean. `error` is set when the model
    call failed — those score zero, because a case you could not get an answer
    for is not a case you passed.
    """

    id: str
    tag: str | None = None
    score: float | None = None
    note: str = ""
    output: str = ""
    latency: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.score is not None and self.score >= PASS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Result":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in row.items() if k in known})


@dataclass
class Run:
    """A scored pass over a case file, plus the metadata describing it."""

    meta: dict[str, Any] = field(default_factory=dict)
    results: list[Result] = field(default_factory=list)

    @property
    def by_id(self) -> dict[str, Result]:
        return {r.id: r for r in self.results}

    @property
    def score(self) -> float | None:
        return mean(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "meta": self.meta,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Run":
        return cls(
            meta=data.get("meta", {}),
            results=[Result.from_dict(r) for r in data.get("results", [])],
        )


SCHEMA = 1
"""Run-file schema version. `compare` reads these files; renaming a meta key
without bumping this is how a diffing tool starts silently lying."""


def mean(results: list[Result]) -> float | None:
    """Mean over scored cases only. None when nothing was scored."""
    scores = [r.score for r in results if r.score is not None]
    return sum(scores) / len(scores) if scores else None


def passed(results: list[Result]) -> int:
    return sum(1 for r in results if r.passed)


def by_tag(results: list[Result]) -> dict[str, list[Result]]:
    groups: dict[str, list[Result]] = {}
    for r in results:
        groups.setdefault(r.tag or "—", []).append(r)
    return groups
