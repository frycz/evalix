"""Scorer registry.

`get()` takes a name or a callable, so `scorer="exact"` and
`scorer=my_function` are the same argument.
"""

from __future__ import annotations

from evalix.scorers import builtin
from evalix.scorers.base import PASS, Context, Score, Scorer, ScorerResult, as_score
from evalix.scorers.custom import ScorerFileError, load_custom
from evalix.scorers.judge import judge

BUILTIN: dict[str, Scorer] = {
    "exact": builtin.exact,
    "contains": builtin.contains,
    "not_contains": builtin.not_contains,
    "regex": builtin.regex,
    "json_parse": builtin.json_parse,
    "json_fields": builtin.json_fields,
    "judge": judge,
    "none": builtin.none,
}

NAMES = tuple(BUILTIN)


def get(scorer: str | Scorer) -> Scorer:
    if callable(scorer):
        return scorer
    if scorer in BUILTIN:
        return BUILTIN[scorer]
    raise KeyError(f"unknown scorer {scorer!r}. Known: {', '.join(NAMES)}, or pass a callable.")


def name_of(scorer: str | Scorer) -> str:
    if isinstance(scorer, str):
        return scorer
    return getattr(scorer, "__name__", "custom")


__all__ = [
    "BUILTIN",
    "NAMES",
    "PASS",
    "Context",
    "Score",
    "Scorer",
    "ScorerFileError",
    "ScorerResult",
    "as_score",
    "get",
    "load_custom",
    "name_of",
]
