"""Loading a scorer from a user's .py file.

The signature is the same one the built-ins use. A two-argument
`score(output, case)` is accepted too, because that is what the older harness
took and there is no reason to break it — the third argument is simply not
passed.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from evalix.scorers.base import Context, Scorer, ScorerResult


def load_custom(path: str | Path) -> Scorer:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("evalix_custom_scorer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, "score", None)
    if fn is None:
        raise AttributeError(f"{path} defines no score() function")

    takes_ctx = len(inspect.signature(fn).parameters) >= 3

    def scorer(output: str, case, ctx: Context) -> ScorerResult:
        return fn(output, case, ctx) if takes_ctx else fn(output, case)

    scorer.__name__ = f"custom:{path.name}"
    return scorer
