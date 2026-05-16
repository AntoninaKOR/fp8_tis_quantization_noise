"""
Unit tests for GSM8K data utilities.

Tests reflect QeRL's exact logic (utils/rewards.py, utils/data.py):
  - extract_hash_answer : splits on "####", takes right side
  - extract_solution    : strict (requires "#### <num>") or flexible (last number)
  - gsm8k_score         : binary exact-match on strict-extracted answer
"""

import pytest

from data.gsm8k import (
    extract_hash_answer,
    extract_solution,
    gsm8k_score,
    normalize_number,
    GSM8K_INSTRUCTION,
)


# ---------------------------------------------------------------------------
# Gold extraction  (extract_hash_answer)
# ---------------------------------------------------------------------------

class TestExtractHashAnswer:
    def test_standard_marker(self):
        assert extract_hash_answer("Natalia sold 48/2 = 24 clips. #### 72") == "72"

    def test_comma_preserved(self):
        # extract_hash_answer returns raw text — commas not stripped here
        assert extract_hash_answer("She earned #### 1,000") == "1,000"

    def test_no_marker_returns_none(self):
        assert extract_hash_answer("  42  ") is None

    def test_whitespace_stripped(self):
        assert extract_hash_answer("####   72   ") == "72"

    def test_decimal(self):
        assert extract_hash_answer("#### 3.14") == "3.14"

    def test_negative(self):
        assert extract_hash_answer("#### -5") == "-5"


# ---------------------------------------------------------------------------
# Predicted answer extraction  (extract_solution)
# ---------------------------------------------------------------------------

class TestExtractSolutionStrict:
    """Strict mode: requires '#### <number>' in the response."""

    def test_marker_present(self):
        assert extract_solution("We compute carefully. #### 72", "strict") == "72"

    def test_comma_stripped(self):
        result = extract_solution("#### 1,000", "strict")
        assert result == "1000"

    def test_decimal(self):
        assert extract_solution("#### 3.14", "strict") == "3.14"

    def test_negative(self):
        assert extract_solution("#### -5", "strict") == "-5"

    def test_no_marker_returns_none(self):
        assert extract_solution("The answer is 42", "strict") is None

    def test_no_number_returns_none(self):
        assert extract_solution("There is no number here.", "strict") is None

    def test_takes_last_of_multiple_markers(self):
        assert extract_solution("#### 10 some reasoning #### 72", "strict") == "72"


class TestExtractSolutionFlexible:
    """Flexible mode: last number in the string."""

    def test_last_number(self):
        assert extract_solution("Step 1: 5 apples. Step 2: 3 more. Total: 8", "flexible") == "8"

    def test_no_number_returns_none(self):
        assert extract_solution("no digits here", "flexible") is None

    def test_lone_dot_returns_none(self):
        assert extract_solution("There is no number here.", "flexible") is None

    def test_negative(self):
        assert extract_solution("temp is -5", "flexible") == "-5"


# ---------------------------------------------------------------------------
# Reward  (gsm8k_score)
# ---------------------------------------------------------------------------

class TestGsm8kScore:
    def test_correct(self):
        assert gsm8k_score("The answer is #### 72", "72") == 1.0

    def test_incorrect(self):
        assert gsm8k_score("The answer is #### 71", "72") == 0.0

    def test_no_marker_returns_zero(self):
        assert gsm8k_score("I don't know", "72") == 0.0

    def test_comma_normalised(self):
        assert gsm8k_score("#### 1,000", "1000") == 1.0

    def test_negative(self):
        assert gsm8k_score("#### -5", "-5") == 1.0

    def test_float_does_not_match_int(self):
        # strict extraction returns "72.0" which != "72"
        assert gsm8k_score("#### 72.0", "72") == 0.0


# ---------------------------------------------------------------------------
# Normalisation  (normalize_number)
# ---------------------------------------------------------------------------

class TestNormalizeNumber:
    def test_int_string(self):
        assert normalize_number("72") == "72"

    def test_float_to_int(self):
        assert normalize_number("72.0") == "72"

    def test_comma_stripped(self):
        assert normalize_number("1,000") == "1000"

    def test_preserves_decimal(self):
        assert normalize_number("3.14") == "3.14"


# ---------------------------------------------------------------------------
# Instruction constant  (GSM8K_INSTRUCTION)
# ---------------------------------------------------------------------------

class TestGsm8kInstruction:
    def test_contains_think_tags(self):
        assert "<think>" in GSM8K_INSTRUCTION

    def test_contains_hash_marker(self):
        assert "####" in GSM8K_INSTRUCTION

    def test_prompt_includes_question(self):
        q = "What is 2 + 2?"
        prompt = q + " " + GSM8K_INSTRUCTION
        assert q in prompt
        assert "####" in prompt
