"""The scorer contract.

A scorer turns one model output into a number. It is the only task-specific
part of the harness — running, aggregating, diffing and storing are identical
whether you are classifying tickets or extracting invoices — which is why it is
the pluggable one.

It is also the part most likely to be silently wrong. Every other component
fails loudly; a scorer that is too lenient just reports progress that isn't
there, and looks exactly like a real win. Test yours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from evalix.runners import Runner

PASS = 0.999
"""Score at or above which a case counts as 'perfect' in fixed/BROKE."""


@dataclass(frozen=True)
class Score:
    """`value is None` means not scored — it drops out of the mean entirely.

    Unscored is not the same as zero, and the distinction is load-bearing: a
    manual-review case that counted as zero would make every run look worse
    than it is.
    """

    value: float | None
    note: str = ""


@dataclass(frozen=True)
class Context:
    """What a scorer gets besides the output and the case.

    `runner` is the interesting one. A scorer that can call a model is a judge,
    and being able to write your own — with your own rubric and your own
    parsing — is the difference between an extension point and a fork.
    """

    runner: Runner
    model: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


ScorerResult = float | tuple[float, str] | Score


class Scorer(Protocol):
    def __call__(self, output: str, case: Any, ctx: Context) -> ScorerResult: ...


def as_score(result: ScorerResult) -> Score:
    """Accept the three shapes a scorer may return, normalise to one."""
    if isinstance(result, Score):
        return result
    if isinstance(result, tuple):
        value, note = result
        return Score(None if value is None else float(value), str(note))
    return Score(None if result is None else float(result), "")
