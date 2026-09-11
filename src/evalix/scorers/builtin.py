"""The built-in scorers. All pure functions — no network, no state.

`expected` means something different in each one; that is the point. The
harness never interprets it, the scorer does.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evalix.scorers.base import Context, Score


def norm(text: Any) -> str:
    """Lowercase, collapse whitespace, drop surrounding punctuation."""
    return re.sub(r"\s+", " ", str(text)).strip().strip(".,;:!\"'` ").lower()


def extract_json(text: str) -> Any:
    """Best-effort JSON parse: tolerates ``` fences and surrounding prose.

    Raises ValueError when there is nothing parseable. Scorers use the raise to
    tell 'did not produce JSON' apart from 'produced JSON with wrong values' —
    different failures wanting different prompt fixes.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no parseable JSON in output")


def exact(output: str, case, ctx: Context) -> Score:
    want, got = norm(case.expected), norm(output)
    return Score(1.0 if want == got else 0.0, f"want={want!r} got={got[:60]!r}")


def contains(output: str, case, ctx: Context) -> Score:
    want = norm(case.expected)
    return Score(1.0 if want in norm(output) else 0.0, f"want substring {want!r}")


def not_contains(output: str, case, ctx: Context) -> Score:
    """`expected` is the string that must NOT appear (the injection canary)."""
    bad = norm(case.expected)
    hit = bad in norm(output)
    return Score(0.0 if hit else 1.0, ("LEAKED " + bad) if hit else "clean")


def regex(output: str, case, ctx: Context) -> Score:
    ok = re.search(str(case.expected), output, re.I | re.S) is not None
    return Score(1.0 if ok else 0.0, f"pattern {case.expected!r}")


def json_parse(output: str, case, ctx: Context) -> Score:
    """Format adherence only — did we get JSON at all, values ignored."""
    try:
        extract_json(output)
        return Score(1.0, "parsed")
    except ValueError as exc:
        return Score(0.0, str(exc))


def json_fields(output: str, case, ctx: Context) -> Score:
    """Fraction of expected fields that match. Unparseable output scores 0.

    Format failures and content failures are different bugs, so the note keeps
    them apart: UNPARSEABLE is what the parse-rate line counts.
    """
    try:
        got = extract_json(output)
    except ValueError as exc:
        return Score(0.0, f"UNPARSEABLE: {exc}")
    if not isinstance(got, dict):
        return Score(0.0, f"expected an object, got {type(got).__name__}")
    want: dict = case.expected or {}
    if not want:
        return Score(0.0, "case has no expected fields")
    wrong = [k for k, v in want.items() if norm(got.get(k, "")) != norm(v)]
    value = (len(want) - len(wrong)) / len(want)
    return Score(value, "all fields ok" if not wrong else "wrong: " + ", ".join(wrong))


def none(output: str, case, ctx: Context) -> Score:
    """No automatic score — record the output for manual reading."""
    return Score(None, "manual review")
