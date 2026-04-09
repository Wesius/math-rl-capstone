"""
Reward functions for GRPO training.

The task: given a target number, generate a math expression that evaluates to it.
Reward is 1.0 if the expression evaluates correctly, 0.0 otherwise.
We also provide partial reward for being close and a format reward.
"""

import re
import ast
import operator

# Safe operators for evaluation (no exec/eval exploits)
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Also allow true division
SAFE_OPS[ast.Div] = operator.truediv


def safe_eval_expr(expr: str) -> float | None:
    """
    Safely evaluate a math expression string using AST parsing.
    Only allows: integers, floats, and basic arithmetic (+, -, *, /, //, %, **).
    Returns None if the expression is invalid or unsafe.
    """
    try:
        # Clean the expression
        expr = expr.strip()
        if not expr:
            return None

        tree = ast.parse(expr, mode="eval")
        return _eval_node(tree.body)
    except Exception:
        return None


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        # Prevent huge exponents
        if op_type == ast.Pow and right > 100:
            raise ValueError("Exponent too large")
        return SAFE_OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        return SAFE_OPS[op_type](_eval_node(node.operand))
    elif isinstance(node, ast.Expression):
        return _eval_node(node.body)
    else:
        raise ValueError(f"Unsupported node type: {type(node)}")


def extract_expression(text: str) -> str | None:
    """
    Extract a math expression from model output.
    Looks for content inside <expr>...</expr> tags first,
    then falls back to finding the last line that looks like math.
    """
    # Try tagged format first
    match = re.search(r"<expr>(.*?)</expr>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: find lines that look like math expressions
    # Match lines containing digits and operators
    math_pattern = re.compile(r"^[\d\s\+\-\*\/\%\(\)\.\^]+$")
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if math_pattern.match(line) and any(c.isdigit() for c in line):
            # Replace ^ with ** for Python
            line = line.replace("^", "**")
            return line

    return None


def correctness_reward(completions: list[str], target: list[int], **kwargs) -> list[float]:
    """
    Binary reward: 1.0 if expression evaluates to target, 0.0 otherwise.
    """
    rewards = []
    for completion, tgt in zip(completions, target):
        expr = extract_expression(completion)
        if expr is None:
            rewards.append(0.0)
            continue

        result = safe_eval_expr(expr)
        if result is not None and abs(result - tgt) < 1e-6:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def closeness_reward(completions: list[str], target: list[int], **kwargs) -> list[float]:
    """
    Partial reward based on how close the expression result is to the target.
    Uses: max(0, 1 - |result - target| / |target|)
    Capped at [0, 1]. Gives 0 for unparseable expressions.
    """
    rewards = []
    for completion, tgt in zip(completions, target):
        expr = extract_expression(completion)
        if expr is None:
            rewards.append(0.0)
            continue

        result = safe_eval_expr(expr)
        if result is None:
            rewards.append(0.0)
            continue

        if tgt == 0:
            # Special case: target is 0
            rewards.append(1.0 if abs(result) < 1e-6 else max(0.0, 1.0 - abs(result) / 100.0))
        else:
            relative_error = abs(result - tgt) / abs(tgt)
            rewards.append(max(0.0, 1.0 - relative_error))

    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Reward for using the correct output format: <expr>...</expr> tags.
    0.5 for having the tags, 0.0 otherwise.
    """
    rewards = []
    for completion in completions:
        if re.search(r"<expr>.*?</expr>", completion, re.DOTALL):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def complexity_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Small bonus for using multiple operations (more interesting expressions).
    Rewards expressions with 2+ operators, up to 0.3 max.
    """
    rewards = []
    for completion in completions:
        expr = extract_expression(completion)
        if expr is None:
            rewards.append(0.0)
            continue

        # Count operators
        num_ops = sum(expr.count(op) for op in ["+", "-", "*", "/", "**", "%"])
        # Subtract back double-counted ** from * count
        num_ops -= expr.count("**")
        reward = min(0.3, num_ops * 0.1)
        rewards.append(reward)

    return rewards
