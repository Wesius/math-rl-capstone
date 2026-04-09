"""Tests for reward functions."""

from src.reward import (
    safe_eval_expr,
    extract_expression,
    correctness_reward,
    closeness_reward,
    format_reward,
    complexity_reward,
)


def test_safe_eval_basic():
    assert safe_eval_expr("3 + 4") == 7.0
    assert safe_eval_expr("10 * 5") == 50.0
    assert safe_eval_expr("100 / 4") == 25.0
    assert safe_eval_expr("2 ** 10") == 1024.0
    assert safe_eval_expr("(3 + 4) * 10") == 70.0
    assert safe_eval_expr("-5 + 10") == 5.0


def test_safe_eval_rejects_unsafe():
    assert safe_eval_expr("__import__('os').system('ls')") is None
    assert safe_eval_expr("open('file')") is None
    assert safe_eval_expr("") is None
    assert safe_eval_expr("hello") is None


def test_safe_eval_huge_exponent():
    # Should reject huge exponents
    assert safe_eval_expr("2 ** 1000") is None


def test_extract_expression_tagged():
    assert extract_expression("<expr>3 + 4</expr>") == "3 + 4"
    assert extract_expression("The answer is <expr>70 + 3</expr> because...") == "70 + 3"


def test_extract_expression_fallback():
    assert extract_expression("70 + 3") == "70 + 3"
    assert extract_expression("Some text\n70 + 3\nMore text") == "70 + 3"


def test_correctness_reward():
    completions = ["<expr>70 + 3</expr>", "<expr>70 + 4</expr>", "garbage"]
    targets = [73, 73, 73]
    rewards = correctness_reward(completions, targets)
    assert rewards == [1.0, 0.0, 0.0]


def test_closeness_reward():
    completions = ["<expr>73</expr>", "<expr>74</expr>", "<expr>0</expr>"]
    targets = [73, 73, 73]
    rewards = closeness_reward(completions, targets)
    assert rewards[0] == 1.0  # exact
    assert 0.9 < rewards[1] < 1.0  # close
    assert rewards[2] == 0.0  # far


def test_format_reward():
    completions = ["<expr>3+4</expr>", "3+4", "blah <expr>x</expr> blah"]
    rewards = format_reward(completions)
    assert rewards == [0.5, 0.0, 0.5]


def test_complexity_reward():
    completions = ["<expr>73</expr>", "<expr>70 + 3</expr>", "<expr>7 * 10 + 3</expr>"]
    rewards = complexity_reward(completions)
    assert rewards[0] == 0.0  # no operators
    assert rewards[1] == 0.1  # 1 operator
    assert rewards[2] == 0.2  # 2 operators


if __name__ == "__main__":
    test_safe_eval_basic()
    test_safe_eval_rejects_unsafe()
    test_safe_eval_huge_exponent()
    test_extract_expression_tagged()
    test_extract_expression_fallback()
    test_correctness_reward()
    test_closeness_reward()
    test_format_reward()
    test_complexity_reward()
    print("All tests passed!")
