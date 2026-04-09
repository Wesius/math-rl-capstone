"""
Diversity metrics for generated math expressions.

Measures how varied the model's outputs are across different targets.
"""

import re
import argparse
import json
from collections import Counter
from difflib import SequenceMatcher


def templatize(expr: str) -> str:
    """Replace all numbers with N to extract structural template."""
    return re.sub(r'\d+\.?\d*', 'N', expr)


def get_operators(expr: str) -> list[str]:
    """Extract operators from expression."""
    ops = []
    for ch in expr:
        if ch in '+-*/%':
            ops.append(ch)
    # Handle ** (power)
    count_dblstar = expr.count('**')
    if count_dblstar:
        # Remove single * that are part of **
        ops = [o for o in ops if o != '*']
        ops.extend(['*'] * (expr.count('*') - 2 * count_dblstar))
        ops.extend(['**'] * count_dblstar)
    # Handle ^
    ops.extend(['^'] * expr.count('^'))
    return ops


def get_operands(expr: str) -> list[str]:
    """Extract numeric literals from expression."""
    return re.findall(r'\d+\.?\d*', expr)


def pairwise_edit_distance(exprs: list[str]) -> float:
    """Average normalized edit distance between all pairs."""
    if len(exprs) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(exprs)):
        for j in range(i + 1, len(exprs)):
            ratio = SequenceMatcher(None, exprs[i], exprs[j]).ratio()
            total += 1.0 - ratio  # distance = 1 - similarity
            count += 1
    return total / count


def analyze(expressions: list[str], targets: list[int] | None = None) -> dict:
    """Compute diversity metrics for a list of expressions."""
    if not expressions:
        return {"error": "no expressions"}

    # 1. Template uniqueness
    templates = [templatize(e) for e in expressions]
    template_counts = Counter(templates)
    unique_templates = len(template_counts)
    most_common_template = template_counts.most_common(1)[0]
    template_dominance = most_common_template[1] / len(expressions)

    # 2. Pairwise edit distance
    avg_edit_dist = pairwise_edit_distance(expressions)

    # 3. Operand variety
    all_operands = []
    per_expr_unique_operands = []
    for expr in expressions:
        ops = get_operands(expr)
        all_operands.extend(ops)
        per_expr_unique_operands.append(len(set(ops)))
    unique_operands_total = len(set(all_operands))
    avg_unique_operands = sum(per_expr_unique_operands) / len(per_expr_unique_operands)

    # 4. Operator distribution
    all_ops = []
    for expr in expressions:
        all_ops.extend(get_operators(expr))
    op_counts = Counter(all_ops)
    total_ops = sum(op_counts.values())
    op_entropy = 0.0
    if total_ops > 0:
        import math
        for count in op_counts.values():
            p = count / total_ops
            if p > 0:
                op_entropy -= p * math.log2(p)

    # 5. Target containment (if targets provided)
    target_in_expr = 0
    if targets:
        for expr, tgt in zip(expressions, targets):
            if re.search(r'(?<!\d)' + re.escape(str(tgt)) + r'(?!\d)', expr):
                target_in_expr += 1

    return {
        "n_expressions": len(expressions),
        "unique_templates": unique_templates,
        "template_ratio": unique_templates / len(expressions),
        "most_common_template": most_common_template[0],
        "template_dominance": template_dominance,
        "avg_pairwise_edit_distance": round(avg_edit_dist, 4),
        "unique_operands_total": unique_operands_total,
        "avg_unique_operands_per_expr": round(avg_unique_operands, 2),
        "operator_distribution": dict(op_counts),
        "operator_entropy_bits": round(op_entropy, 3),
        "target_in_expr_count": target_in_expr,
        "target_in_expr_pct": round(target_in_expr / len(expressions), 2) if targets else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze expression diversity")
    parser.add_argument("--file", type=str, help="JSON file with list of {expr, target} dicts")
    parser.add_argument("--inline", type=str, nargs="+", help="Expressions to analyze inline")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            data = json.load(f)
        expressions = [d["expr"] for d in data if d.get("expr")]
        targets = [d["target"] for d in data if d.get("expr")]
    elif args.inline:
        expressions = args.inline
        targets = None
    else:
        parser.error("Provide --file or --inline")

    metrics = analyze(expressions, targets)
    print(json.dumps(metrics, indent=2))


# Hardcoded experiment data for quick comparison
EXPERIMENT_DATA = {
    "baseline": {
        "expressions": ["73", "5000", "256", "42", "100", "7", "9999", "1000", "2048", "500"],
        "targets": [73, 5000, 256, 42, 100, 7, 9999, 1000, 2048, 500],
    },
    "exp1_high_complexity": {
        "expressions": [
            "(7 - 1) / 1 + 1", "(42 - 1) / 1 + 1", "(73 - 1) / 1 + 1",
            "(100 - 1) / 1 + 1", "(256 - 1) / 1 + 1", "(500 - 1) / 1 + 1",
            "(1000 - 1) / 1 + 1", "(2048 - 1) / 1 + 1", "(5000 - 1) / 1 + 1",
            "(9999 - 1) / 1 + 1",
        ],
        "targets": [7, 42, 73, 100, 256, 500, 1000, 2048, 5000, 9999],
    },
    "exp2_ban_trivial": {
        "expressions": [
            "(7 + 1) / 1 - 1", "(42 + 1) / 1 - 1", "(73 + 1) / 1 - 1",
            "(100 + 1) / 1 - 1", "(256 + 1) / 1 - 1", "(500 + 1) / 1 - 1",
            "(1000 + 1) / 1 - 1", "(2048 + 1) / 1 - 1", "(5000 + 1) / 1 - 1",
            "(9999 + 1) / 1 - 1",
        ],
        "targets": [7, 42, 73, 100, 256, 500, 1000, 2048, 5000, 9999],
    },
    "exp4_no_target": {
        "expressions": [
            "4 * 1 + 2 - 3 / 2 + 1", "5 * 7 + 1 / 2 + 1", "7 * 10 + 3 + 1 / 2",
            "9 * 10 + 1 + 1", "2**8", "5 * 100 + 5 + 1",
            "7 * 111 + 1", "2**11", "5 * 100 + 5 + 1",
            "1000 * 9 + 1 + 1",
        ],
        "targets": [7, 42, 73, 100, 256, 500, 1000, 2048, 5000, 9999],
    },
}


def compare_all():
    """Compare diversity across all experiments."""
    print("=" * 70)
    print(f"{'Metric':<35} {'Baseline':>8} {'Exp1':>8} {'Exp2':>8} {'Exp4':>8}")
    print("=" * 70)

    results = {}
    for name, data in EXPERIMENT_DATA.items():
        results[name] = analyze(data["expressions"], data["targets"])

    metrics_to_show = [
        ("unique_templates", "Unique templates"),
        ("template_ratio", "Template ratio (1.0=all unique)"),
        ("template_dominance", "Most common template %"),
        ("avg_pairwise_edit_distance", "Avg pairwise edit dist"),
        ("unique_operands_total", "Unique operands (total)"),
        ("avg_unique_operands_per_expr", "Avg operands per expr"),
        ("operator_entropy_bits", "Operator entropy (bits)"),
        ("target_in_expr_pct", "Target in expr %"),
    ]

    for key, label in metrics_to_show:
        vals = []
        for name in ["baseline", "exp1_high_complexity", "exp2_ban_trivial", "exp4_no_target"]:
            v = results[name].get(key, "—")
            if isinstance(v, float):
                vals.append(f"{v:>8.3f}")
            else:
                vals.append(f"{v:>8}")
            
        print(f"{label:<35} {vals[0]} {vals[1]} {vals[2]} {vals[3]}")

    print("=" * 70)
    print()

    # Show dominant template per experiment
    for name in ["baseline", "exp1_high_complexity", "exp2_ban_trivial", "exp4_no_target"]:
        t = results[name]["most_common_template"]
        d = results[name]["template_dominance"]
        print(f"{name}: dominant template = '{t}' ({d:.0%})")


if __name__ == "__main__":
    compare_all()
