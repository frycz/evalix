"""evalix — a small eval harness that tells you which cases your prompt change broke.

    fixed case set → change one thing → measure → keep or revert

The score tells you whether; the diff tells you which. A prompt change that
gains three cases and loses two is a +1 you would otherwise call a win.

    from evalix import run

    report = run(cases="cases.jsonl", prompt="v2.txt", scorer="exact")
    print(report.render())
"""

from __future__ import annotations

from evalix.api import (
    Comparison,
    Estimate,
    Report,
    ScorerError,
    compare,
    estimate,
    run,
    run_case,
)
from evalix.cases import Case, load_cases
from evalix.diff import CaseChange, Diff, diff
from evalix.prompt import PromptError, build_request
from evalix.runners import (
    Message,
    MissingCredentials,
    Refusal,
    Request,
    Response,
    Runner,
    RunnerError,
)
from evalix.scorers import PASS, Context, Score, Scorer, ScorerFileError, load_custom
from evalix.scoring import Result, Run

__version__ = "0.1.0"

__all__ = [
    "PASS",
    "Case",
    "CaseChange",
    "Comparison",
    "Context",
    "Diff",
    "Estimate",
    "Message",
    "MissingCredentials",
    "PromptError",
    "Refusal",
    "Report",
    "Request",
    "Response",
    "Result",
    "Run",
    "Runner",
    "RunnerError",
    "Score",
    "Scorer",
    "ScorerError",
    "ScorerFileError",
    "__version__",
    "build_request",
    "compare",
    "diff",
    "estimate",
    "load_cases",
    "load_custom",
    "run",
    "run_case",
]
