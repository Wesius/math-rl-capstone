"""
Evaluation script for the trained model.

Generates expressions for eval targets and computes metrics:
- Accuracy (exact match)
- Closeness (relative error)
- Format compliance
- Expression diversity & complexity
- Difficulty breakdown (easy/medium/hard)
"""

import argparse
import json
import os
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import SYSTEM_PROMPT, USER_TEMPLATE, build_eval_dataset
from src.reward import extract_expression, safe_eval_expr


def load_model(model_path: str, base_model: str = "LiquidAI/LFM2-350M"):
    """Load the base model with LoRA adapter merged in."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # LoRA checkpoint — load base + adapter
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        # Full model checkpoint
        print(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def generate_expression(
    model,
    tokenizer,
    target: int,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    num_samples: int = 1,
) -> list[str]:
    """Generate expression(s) for a given target number."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(target=target)},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_samples,
        min_p=0.15,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

    results = []
    for output in outputs:
        # Decode only the new tokens
        new_tokens = output[input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(text)

    return results


def evaluate(
    model,
    tokenizer,
    targets: list[int],
    temperature: float = 0.3,
    num_samples: int = 1,
    verbose: bool = False,
) -> tuple[dict, list[dict]]:
    """Run evaluation over a list of targets."""
    results = []
    correct = 0
    total = len(targets)
    closeness_sum = 0.0
    format_ok = 0

    for i, target in enumerate(targets):
        completions = generate_expression(
            model, tokenizer, target,
            temperature=temperature,
            num_samples=num_samples,
        )

        for completion in completions:
            expr_str = extract_expression(completion)
            expr_result = safe_eval_expr(expr_str) if expr_str else None

            is_correct = expr_result is not None and abs(expr_result - target) < 1e-6
            has_format = "<expr>" in completion and "</expr>" in completion

            if is_correct:
                correct += 1
            if has_format:
                format_ok += 1

            # Closeness
            if expr_result is not None and target != 0:
                closeness = max(0.0, 1.0 - abs(expr_result - target) / abs(target))
            elif expr_result is not None and target == 0:
                closeness = 1.0 if abs(expr_result) < 1e-6 else 0.0
            else:
                closeness = 0.0
            closeness_sum += closeness

            # Count operators
            num_ops = 0
            if expr_str:
                num_ops = sum(expr_str.count(op) for op in ["+", "-", "*", "/", "%"])
                num_ops -= expr_str.count("**")  # Don't double-count

            result_entry = {
                "target": target,
                "completion": completion,
                "expression": expr_str,
                "evaluated": expr_result,
                "correct": is_correct,
                "closeness": closeness,
                "format_ok": has_format,
                "num_operators": num_ops,
            }
            results.append(result_entry)

            if verbose and i < 20:
                status = "OK" if is_correct else "WRONG"
                print(f"  [{status}] target={target}, expr={expr_str}, result={expr_result}")

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{total} ({correct}/{(i + 1) * num_samples} correct)")

    total_samples = total * num_samples
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    avg_closeness = closeness_sum / total_samples if total_samples > 0 else 0.0
    format_rate = format_ok / total_samples if total_samples > 0 else 0.0

    # Difficulty breakdown
    easy = [r for r in results if r["target"] <= 100]
    medium = [r for r in results if 100 < r["target"] <= 999]
    hard = [r for r in results if r["target"] > 999]

    def bucket_accuracy(bucket):
        if not bucket:
            return 0.0
        return sum(1 for r in bucket if r["correct"]) / len(bucket)

    # Expression diversity — how many unique expressions for same target
    expr_counter = Counter(r["expression"] for r in results if r["expression"])

    metrics = {
        "accuracy": accuracy,
        "avg_closeness": avg_closeness,
        "format_rate": format_rate,
        "total_samples": total_samples,
        "correct": correct,
        "accuracy_easy": bucket_accuracy(easy),
        "accuracy_medium": bucket_accuracy(medium),
        "accuracy_hard": bucket_accuracy(hard),
        "n_easy": len(easy),
        "n_medium": len(medium),
        "n_hard": len(hard),
        "unique_expressions": len(expr_counter),
        "avg_operators": (
            sum(r["num_operators"] for r in results) / total_samples
            if total_samples > 0 else 0.0
        ),
    }

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained math expression model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model/adapter")
    parser.add_argument("--base-model", type=str, default="LiquidAI/LFM2-350M")
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per target")
    parser.add_argument("--output-dir", type=str, default="output/eval")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--eval-base", action="store_true", help="Also evaluate base model (no adapter)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate trained model
    print("=" * 60)
    print("Evaluating TRAINED model")
    print("=" * 60)
    model, tokenizer = load_model(args.model_path, args.base_model)

    eval_ds = build_eval_dataset(n=args.n_eval)
    targets = eval_ds["target"]

    metrics, results = evaluate(
        model, tokenizer, targets,
        temperature=args.temperature,
        num_samples=args.num_samples,
        verbose=args.verbose,
    )

    print(f"\n{'=' * 40}")
    print("TRAINED MODEL RESULTS")
    print(f"{'=' * 40}")
    print(f"Accuracy:         {metrics['accuracy']:.1%}")
    print(f"Avg Closeness:    {metrics['avg_closeness']:.3f}")
    print(f"Format Rate:      {metrics['format_rate']:.1%}")
    print(f"Accuracy (easy):  {metrics['accuracy_easy']:.1%}")
    print(f"Accuracy (medium):{metrics['accuracy_medium']:.1%}")
    print(f"Accuracy (hard):  {metrics['accuracy_hard']:.1%}")
    print(f"Unique Exprs:     {metrics['unique_expressions']}")
    print(f"Avg Operators:    {metrics['avg_operators']:.1f}")

    # Save results
    with open(f"{args.output_dir}/metrics_trained.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{args.output_dir}/results_trained.json", "w") as f:
        json.dump(results[:100], f, indent=2)  # Save first 100 for inspection

    # Optionally evaluate base model for comparison
    if args.eval_base:
        print(f"\n{'=' * 60}")
        print("Evaluating BASE model (no RL training)")
        print("=" * 60)

        # Clear GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        base_model.eval()

        base_metrics, base_results = evaluate(
            base_model, tokenizer, targets,
            temperature=args.temperature,
            num_samples=args.num_samples,
            verbose=args.verbose,
        )

        print(f"\n{'=' * 40}")
        print("BASE MODEL RESULTS")
        print(f"{'=' * 40}")
        print(f"Accuracy:         {base_metrics['accuracy']:.1%}")
        print(f"Avg Closeness:    {base_metrics['avg_closeness']:.3f}")
        print(f"Format Rate:      {base_metrics['format_rate']:.1%}")
        print(f"Accuracy (easy):  {base_metrics['accuracy_easy']:.1%}")
        print(f"Accuracy (medium):{base_metrics['accuracy_medium']:.1%}")
        print(f"Accuracy (hard):  {base_metrics['accuracy_hard']:.1%}")

        with open(f"{args.output_dir}/metrics_base.json", "w") as f:
            json.dump(base_metrics, f, indent=2)
        with open(f"{args.output_dir}/results_base.json", "w") as f:
            json.dump(base_results[:100], f, indent=2)

        # Comparison
        print(f"\n{'=' * 40}")
        print("IMPROVEMENT (trained - base)")
        print(f"{'=' * 40}")
        print(f"Accuracy:  {metrics['accuracy'] - base_metrics['accuracy']:+.1%}")
        print(f"Closeness: {metrics['avg_closeness'] - base_metrics['avg_closeness']:+.3f}")
        print(f"Format:    {metrics['format_rate'] - base_metrics['format_rate']:+.1%}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
