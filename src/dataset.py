"""
Dataset generation for the math expression RL task.

Generates prompts like:
    "Find a math expression that equals 73. Use +, -, *, / and parentheses.
     Write your answer inside <expr></expr> tags."

Target numbers range from simple (1-100) to harder (100-9999).
"""

import argparse
import json
import random

from datasets import Dataset


SYSTEM_PROMPT = (
    "You are a math assistant. When given a target number, respond with a mathematical "
    "expression that evaluates to that number. Use only integers, +, -, *, /, and parentheses. "
    "Put your expression inside <expr></expr> tags."
)

USER_TEMPLATE = "Find a math expression that equals {target}. Respond with ONLY <expr>YOUR_EXPRESSION</expr>."


def generate_targets(
    n: int = 10000,
    easy_frac: float = 0.4,
    medium_frac: float = 0.35,
    hard_frac: float = 0.25,
    seed: int = 42,
) -> list[int]:
    """
    Generate a list of target numbers with a curriculum distribution.

    Easy:   1-100
    Medium: 100-999
    Hard:   1000-9999
    """
    rng = random.Random(seed)

    n_easy = int(n * easy_frac)
    n_medium = int(n * medium_frac)
    n_hard = n - n_easy - n_medium

    targets = []
    targets.extend(rng.randint(1, 100) for _ in range(n_easy))
    targets.extend(rng.randint(100, 999) for _ in range(n_medium))
    targets.extend(rng.randint(1000, 9999) for _ in range(n_hard))

    rng.shuffle(targets)
    return targets


def build_dataset(n: int = 10000, seed: int = 42) -> Dataset:
    """Build a HuggingFace Dataset with prompts and targets."""
    targets = generate_targets(n=n, seed=seed)

    prompts = []
    for target in targets:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(target=target)},
        ]
        prompts.append(messages)

    return Dataset.from_dict({
        "prompt": prompts,
        "target": targets,
    })


def build_eval_dataset(n: int = 500, seed: int = 99) -> Dataset:
    """Build a separate eval dataset with different seed."""
    return build_dataset(n=n, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Generate datasets for math expression RL")
    parser.add_argument("--n-train", type=int, default=10000, help="Number of training examples")
    parser.add_argument("--n-eval", type=int, default=500, help="Number of eval examples")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ds = build_dataset(n=args.n_train, seed=args.seed)
    eval_ds = build_eval_dataset(n=args.n_eval, seed=args.seed + 57)

    train_ds.save_to_disk(f"{args.output_dir}/train")
    eval_ds.save_to_disk(f"{args.output_dir}/eval")

    # Also save stats
    train_targets = train_ds["target"]
    stats = {
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "train_target_range": [min(train_targets), max(train_targets)],
        "train_target_mean": sum(train_targets) / len(train_targets),
    }
    with open(f"{args.output_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved {len(train_ds)} training and {len(eval_ds)} eval examples to {args.output_dir}/")
    print(f"Stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
