"""Scorer tests. Every one runs offline — no API key, no tokens.

The scorers are where an eval harness quietly lies to you: one that is too
lenient reports progress that isn't there. They are pure functions, so they
can be pinned down exactly.
"""

from __future__ import annotations

import pytest

from evalix import Case, Score
from evalix.scorers import as_score, builtin, get


def case(**kwargs):
    return Case(id=kwargs.pop("id", "t1"), **kwargs)


class TestNormalisation:
    def test_case_and_whitespace_are_ignored(self):
        assert builtin.norm("  Billing\n") == builtin.norm("billing")

    def test_non_strings_are_coerced(self):
        assert builtin.norm(42) == "42"


class TestExtractJson:
    def test_plain_object(self):
        assert builtin.extract_json('{"a": 1}') == {"a": 1}

    def test_object_inside_prose(self):
        assert builtin.extract_json('Sure!\n{"a": 1}\nHope that helps') == {"a": 1}

    def test_fenced_block(self):
        assert builtin.extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_brackets_inside_strings_do_not_confuse_it(self):
        assert builtin.extract_json('Sure: {"a": "}"} done') == {"a": "}"}
        assert builtin.extract_json('Sure: ["]", 1] done') == ["]", 1]

    def test_skips_a_brace_that_is_not_json(self):
        assert builtin.extract_json('Use {name} here: {"a": 1}') == {"a": 1}

    def test_raises_on_garbage(self):
        with pytest.raises(ValueError):
            builtin.extract_json("no json here")


class TestExact:
    def test_match_ignores_case_and_padding(self, ctx):
        assert builtin.exact("  Billing  ", case(expected="billing"), ctx()).value == 1.0

    def test_substring_is_not_a_match(self, ctx):
        assert builtin.exact("billing question", case(expected="billing"), ctx()).value == 0.0


class TestContains:
    def test_substring_matches(self, ctx):
        assert builtin.contains("a Billing issue", case(expected="billing"), ctx()).value == 1.0

    def test_absent_scores_zero(self, ctx):
        assert builtin.contains("about invoices", case(expected="billing"), ctx()).value == 0.0


class TestNotContains:
    """The injection canary: expected is what must NOT appear."""

    def test_clean_output_scores_one(self, ctx):
        score = builtin.not_contains("here is the summary", case(expected="PWNED"), ctx())
        assert (score.value, score.note) == (1.0, "clean")

    def test_leak_scores_zero(self, ctx):
        score = builtin.not_contains("sure: PWNED", case(expected="PWNED"), ctx())
        assert score.value == 0.0
        assert "LEAKED" in score.note


class TestJsonScorers:
    def test_json_parse_ignores_values(self, ctx):
        c = case(expected={"vendor": "ACME"})
        assert builtin.json_parse('{"vendor": "WRONG"}', c, ctx()).value == 1.0

    def test_json_parse_fails_on_prose(self, ctx):
        assert builtin.json_parse("The vendor is ACME.", case(expected={}), ctx()).value == 0.0

    def test_fields_partial_credit(self, ctx):
        c = case(expected={"vendor": "ACME", "total": "42.00"})
        score = builtin.json_fields('{"vendor": "ACME", "total": "9.99"}', c, ctx())
        assert score.value == 0.5
        assert "total" in score.note

    def test_fields_all_correct(self, ctx):
        c = case(expected={"vendor": "ACME", "total": "42.00"})
        assert builtin.json_fields('{"vendor": "acme", "total": "42.00"}', c, ctx()).value == 1.0

    def test_unparseable_scores_zero_and_is_labelled(self, ctx):
        """The UNPARSEABLE prefix is what the parse-rate line counts."""
        score = builtin.json_fields("no json", case(expected={"a": "b"}), ctx())
        assert score.value == 0.0
        assert score.note.startswith("UNPARSEABLE")

    def test_missing_field_is_not_silently_ok(self, ctx):
        assert builtin.json_fields('{"other": "x"}', case(expected={"vendor": "A"}), ctx()).value == 0.0


class TestNoneScorer:
    def test_returns_none_so_it_drops_out_of_the_mean(self, ctx):
        assert builtin.none("anything", case(), ctx()).value is None


class TestRegistry:
    def test_names_resolve(self):
        assert get("exact") is builtin.exact

    def test_callables_pass_through(self):
        def mine(output, case, ctx):
            return 1.0

        assert get(mine) is mine

    def test_unknown_name_is_loud(self):
        with pytest.raises(KeyError, match="unknown scorer"):
            get("exakt")


class TestAsScore:
    def test_bare_float(self):
        assert as_score(0.5) == Score(0.5, "")

    def test_tuple(self):
        assert as_score((1.0, "why")) == Score(1.0, "why")

    def test_score_passes_through(self):
        assert as_score(Score(None, "manual")) == Score(None, "manual")
