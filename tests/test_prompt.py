"""Prompt placement: the same words in a different slot behave differently."""

from __future__ import annotations

import pytest

from evalix import Case, PromptError, build_request

CASE = Case(id="t1", input="hello")


class TestSystemPlacement:
    def test_keeps_the_prompt_out_of_the_user_turn(self):
        request = build_request(CASE, "You are terse.")
        assert request.text == "hello"
        assert request.system == "You are terse."

    def test_empty_prompt_gives_no_system(self):
        assert build_request(CASE, "").system is None

    def test_placeholder_is_an_error_not_a_silent_literal(self):
        """bench.py sent the literal '{input}' to the model and scored the result."""
        with pytest.raises(PromptError, match="placement system"):
            build_request(CASE, "Classify <q>{input}</q>")


class TestUserPlacement:
    def test_concatenates(self):
        request = build_request(CASE, "You are terse.", placement="user")
        assert request.system is None
        assert request.text == "You are terse.\n\nhello"

    def test_placeholder_is_substituted(self):
        request = build_request(CASE, "Answer <q>{input}</q>", placement="user")
        assert request.text == "Answer <q>hello</q>"

    def test_empty_prompt_is_just_the_case(self):
        assert build_request(CASE, "", placement="user").text == "hello"


def test_unknown_placement_is_loud():
    with pytest.raises(PromptError, match="placement must be"):
        build_request(CASE, "x", placement="sidecar")


def test_request_carries_the_call_settings():
    request = build_request(CASE, "x", model="m", max_tokens=64, effort="low")
    assert (request.model, request.max_tokens, request.effort) == ("m", 64, "low")
    assert len(request.messages) == 1
