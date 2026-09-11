"""The default runner: Claude via capix.

This is the only module in the package that imports capix. Everything it
raises is translated into evalix's own error types — a seam that lets a
provider exception through is not a seam.
"""

from __future__ import annotations

from typing import Callable

from evalix.runners import (
    MissingCredentials,
    Refusal,
    Request,
    Response,
    RunnerError,
)


def default_model() -> str:
    from capix import default_model as _default_model

    return _default_model()


def capix_runner() -> Callable[[Request], Response]:
    import capix

    def run(request: Request) -> Response:
        try:
            message = capix.ask_message(
                request.text,
                system=request.system,
                model=request.model,
                max_tokens=request.max_tokens,
                effort=request.effort,
            )
        except capix.MissingCredentialsError as exc:
            raise MissingCredentials(str(exc)) from exc
        except capix.RefusalError as exc:
            raise Refusal(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - anything else is a transport failure
            raise RunnerError(f"{type(exc).__name__}: {exc}") from exc

        return Response(
            text="".join(b.text for b in message.content if b.type == "text"),
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            raw=message,
        )

    return run
