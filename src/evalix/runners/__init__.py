"""The provider seam: everything that talks to a model goes through here.

A runner is any callable taking a `Request` and returning a `Response`. The
default one wraps capix, but nothing in the core imports capix — so running
evalix against another provider is an argument, not a fork.

The request is a structured object rather than `(prompt, system)` on purpose.
v1 is single-turn, but that is a documented scope boundary, not a design one:
extra turns and tools become a field here instead of a new major version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class RunnerError(Exception):
    """A provider call failed. One bad case should not kill a whole run."""


class Refusal(RunnerError):
    """The model declined to answer. Scored zero, recorded, run continues."""


class MissingCredentials(RunnerError):
    """No API key. Unlike the others this is fatal — every case will fail."""


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Request:
    """One model call.

    `messages` is always length 1 in v1. It is a list anyway so that adding a
    turn later is additive rather than breaking.
    """

    messages: list[Message]
    system: str | None = None
    model: str | None = None
    max_tokens: int = 2000
    effort: str | None = None

    @property
    def text(self) -> str:
        """The single user turn, for runners that only speak in strings."""
        return self.messages[-1].content


@dataclass(frozen=True)
class Response:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = field(default=None, repr=False)  # the provider object, if you need it


class Runner(Protocol):
    def __call__(self, request: Request) -> Response: ...


def default_runner() -> Runner:
    """The capix runner, imported lazily so the core stays provider-free."""
    from evalix.runners.capix import capix_runner

    return capix_runner()


__all__ = [
    "Message",
    "MissingCredentials",
    "Refusal",
    "Request",
    "Response",
    "Runner",
    "RunnerError",
    "default_runner",
]
