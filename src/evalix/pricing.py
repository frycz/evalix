"""Rough cost estimates.

Deliberately approximate: the point is to catch "this prompt is 10x larger
than I thought" before you spend, not to bill against. Prices are $ per 1M
tokens as (input, output), and can be extended at runtime — this table will
rot, and a wrong number is worse than no number.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def load_overrides() -> None:
    """Merge $EVALIX_PRICES, a JSON file of {"model": [in, out]}."""
    path = os.environ.get("EVALIX_PRICES")
    if not path or not Path(path).exists():
        return
    for model, pair in json.loads(Path(path).read_text()).items():
        PRICES[model] = (float(pair[0]), float(pair[1]))


def cost(model: str | None, input_tokens: int, output_tokens: int) -> float | None:
    """None when the model is unpriced — better than a confident wrong number."""
    price = PRICES.get(model or "")
    if not price:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
