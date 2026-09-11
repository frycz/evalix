"""Turning a prompt file and a case into one `Request`.

The placement choice is not cosmetic. `system` carries operator authority and
is the half of the payload that can cache; the user turn is where data belongs.
Concatenating both into one user turn is easier to talk out of its job — which
is the whole point of measuring it rather than assuming.
"""

from __future__ import annotations

from evalix.cases import Case
from evalix.runners import Message, Request

PLACEMENTS = ("system", "user")


class PromptError(ValueError):
    """The prompt file and the placement disagree about where data goes."""


def build_request(
    case: Case,
    prompt_text: str,
    *,
    placement: str = "system",
    model: str | None = None,
    max_tokens: int = 2000,
    effort: str | None = None,
) -> Request:
    if placement not in PLACEMENTS:
        raise PromptError(f"placement must be one of {PLACEMENTS}, got {placement!r}")

    if placement == "system":
        # A `{input}` placeholder here is always a mistake: the system prompt is
        # the stable half and the case never gets substituted into it. bench.py
        # sent the literal string "{input}" to the model and scored the result.
        if "{input}" in prompt_text:
            raise PromptError(
                "the prompt contains '{input}' but --placement system never substitutes it "
                "(the case goes in the user turn). Use --placement user, or remove it."
            )
        return Request(
            messages=[Message(role="user", content=case.input)],
            system=prompt_text or None,
            model=model,
            max_tokens=max_tokens,
            effort=effort,
        )

    if "{input}" in prompt_text:
        content = prompt_text.replace("{input}", case.input)
    else:
        content = f"{prompt_text}\n\n{case.input}" if prompt_text else case.input

    return Request(
        messages=[Message(role="user", content=content)],
        system=None,
        model=model,
        max_tokens=max_tokens,
        effort=effort,
    )
