"""Shared fixtures. Nothing here touches the network or needs an API key.

That is not incidental: the seam that makes evalix provider-agnostic is the
same seam that makes the whole pipeline testable offline. A fake runner is
three lines.
"""

from __future__ import annotations

import json

import pytest

from evalix import Context, Response


@pytest.fixture
def runner():
    """A runner returning canned text, recording what it was asked."""

    def make(text="ok", *, calls=None, error=None):
        def run(request):
            if calls is not None:
                calls.append(request)
            if error is not None:
                raise error
            reply = text(request) if callable(text) else text
            return Response(text=reply, input_tokens=11, output_tokens=7)

        return run

    return make


@pytest.fixture
def ctx(runner):
    def make(**config):
        return Context(runner=runner(), model="test-model", config=config)

    return make


@pytest.fixture
def cases_file(tmp_path):
    def make(rows, name="cases.jsonl"):
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return path

    return make
