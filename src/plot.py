"""
Plotting utilities for analysis and report figures.
Generates publication-quality plots for the LaTeX report.
"""

import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_accuracy_by_difficulty(metrics_trained: dict, metrics_base: dict | None, output_path: str):
    """Bar chart comparing accuracy across difficulty buckets."""
    categories = ["Easy\n(1-100)", "Medium\n(100-999)", "Hard\n(1000-9999)", "Overall"]
    trained_vals = [
        metrics_trained["accuracy_easy"],
        metrics_trained["accuracy_medium"],
        metrics_trained["accuracy_hard"],
        metrics_trained["accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(categories))
    width = 0.35

    if metrics_base:
        base_vals = [
            metrics_base["accuracy_easy"],
            metrics_base["accuracy_medium"],
            metrics_base["accuracy_hard"],
            metrics_base["accuracy"],
        ]
        ax.bar(x - width / 2, base_vals, width, label="Base LFM2-350M", color="#94a3b8", edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, trained_vals, width, label="GRPO Fine-tuned", color="#3b82f6", edgecolor="black", linewidth=0.5)
        ax.legend()
    else:
        ax.bar(x, trained_vals, width, label="GRPO Fine-tuned", color="#3b82f6", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Accuracy")
    ax.set_title("Expression Accuracy by Target Difficulty")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_reward_curves(wandb_export_path: str, output_path: str):
    """
    Plot training reward curves from a wandb CSV export.
    If no wandb export exists, generates a placeholder.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if os.path.exists(wandb_export_path):
        import pandas as pd
        df = pd.read_csv(wandb_export_path)

        # Plot total reward
        if "reward" in df.columns:
            ax = axes[0]
            ax.plot(df.index, df["reward"], color="#3b82f6", linewidth=1.5)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Mean Reward")
            ax.set_title("Total Reward During Training")
            ax.grid(alpha=0.3)

        # Plot individual rewards if available
        ax = axes[1]
        reward_cols = [c for c in df.columns if c.startswith("reward/") and c.endswith("/mean")]
        for col in reward_cols:
            label = col.replace("reward/", "").replace("/mean", "")
            ax.plot(df.index, df[col], linewidth=1.2, label=label)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Individual Reward Components")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        # Placeholder
        for ax in axes:
            ax.text(0.5, 0.5, "Run training to generate\nreward curves",
                    ha="center", va="center", fontsize=12, color="#94a3b8",
                    transform=ax.transAxes)
            ax.set_xlabel("Training Step")
        axes[0].set_title("Total Reward During Training")
        axes[1].set_title("Individual Reward Components")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_target_distribution(output_path: str):
    """Plot the training data target distribution."""
    from src.dataset import generate_targets

    targets = generate_targets(n=10000)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(targets, bins=50, color="#3b82f6", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Target Number")
    ax.set_ylabel("Count")
    ax.set_title("Training Data: Target Number Distribution")
    ax.grid(axis="y", alpha=0.3)

    # Add difficulty region annotations
    ax.axvspan(1, 100, alpha=0.1, color="green", label="Easy (1-100)")
    ax.axvspan(100, 999, alpha=0.1, color="orange", label="Medium (100-999)")
    ax.axvspan(999, 9999, alpha=0.1, color="red", label="Hard (1000-9999)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_analysis(results: list[dict], output_path: str):
    """Scatter plot of target vs evaluated result for incorrect answers."""
    incorrect = [r for r in results if not r["correct"] and r["evaluated"] is not None]

    if not incorrect:
        print("No incorrect results to plot")
        return

    targets = [r["target"] for r in incorrect]
    evaluated = [r["evaluated"] for r in incorrect]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, evaluated, alpha=0.4, s=15, color="#ef4444")
    # Perfect line
    lims = [min(min(targets), min(evaluated)), max(max(targets), max(evaluated))]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1, label="y = x (correct)")
    ax.set_xlabel("Target Number")
    ax.set_ylabel("Model's Expression Result")
    ax.set_title("Error Analysis: Incorrect Predictions")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for report")
    parser.add_argument("--eval-dir", type=str, default="output/eval")
    parser.add_argument("--wandb-csv", type=str, default="output/wandb_export.csv")
    parser.add_argument("--output-dir", type=str, default="report/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load metrics
    trained_path = os.path.join(args.eval_dir, "metrics_trained.json")
    base_path = os.path.join(args.eval_dir, "metrics_base.json")

    metrics_trained = None
    metrics_base = None

    if os.path.exists(trained_path):
        with open(trained_path) as f:
            metrics_trained = json.load(f)
    if os.path.exists(base_path):
        with open(base_path) as f:
            metrics_base = json.load(f)

    # Generate plots
    plot_target_distribution(os.path.join(args.output_dir, "target_distribution.pdf"))

    if metrics_trained:
        plot_accuracy_by_difficulty(
            metrics_trained, metrics_base,
            os.path.join(args.output_dir, "accuracy_by_difficulty.pdf"),
        )

    plot_reward_curves(args.wandb_csv, os.path.join(args.output_dir, "reward_curves.pdf"))

    # Error analysis
    results_path = os.path.join(args.eval_dir, "results_trained.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        plot_error_analysis(results, os.path.join(args.output_dir, "error_analysis.pdf"))

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
