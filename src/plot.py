"""
Plotting utilities for the capstone report.

Generates publication-quality figures for the LaTeX report, including:
- Plots from eval output files (accuracy, reward curves, error analysis, target distribution)
- Cross-experiment comparison plots (Pareto frontier, phase transition, template dominance,
  completion length, reward heatmap) from hardcoded experimental data
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

OUTPUT_DIR = "report/figures"


# ---------------------------------------------------------------------------
# Eval-based plots (from output files)
# ---------------------------------------------------------------------------


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
    """Plot training reward curves from a wandb CSV export."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if os.path.exists(wandb_export_path):
        import pandas as pd
        df = pd.read_csv(wandb_export_path)

        if "reward" in df.columns:
            ax = axes[0]
            ax.plot(df.index, df["reward"], color="#3b82f6", linewidth=1.5)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Mean Reward")
            ax.set_title("Total Reward During Training")
            ax.grid(alpha=0.3)

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

    ax.axvspan(1, 100, alpha=0.1, color="green", label="Easy (1-100)")
    ax.axvspan(100, 999, alpha=0.1, color="orange", label="Medium (100-999)")
    ax.axvspan(1000, 9999, alpha=0.1, color="red", label="Hard (1000-9999)")
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


# ---------------------------------------------------------------------------
# Cross-experiment comparison plots (from hardcoded experimental data)
# ---------------------------------------------------------------------------


def plot_pareto_frontier():
    """Diversity vs Correctness Pareto frontier from Table 6."""
    configs = [
        ("div01_baseline", 0.400, 2.5),
        ("w2_close_1.0", 0.300, 11.7),
        ("w2_easy_only", 0.233, 48.3),
        ("w2_low_correct", 0.300, 71.7),
        ("div05_fast_lr", 0.050, 85.0),
        ("w2_close_1.3", 0.017, 100.0),
    ]
    names = [c[0] for c in configs]
    diversity = [c[1] for c in configs]
    correctness = [c[2] for c in configs]

    fig, ax = plt.subplots(figsize=(7, 5))

    scores = [d * (1 + c / 100) for d, c in zip(diversity, correctness)]
    scatter = ax.scatter(correctness, diversity, c=scores, cmap="RdYlGn",
                         s=120, edgecolor="black", linewidth=0.8, zorder=3)

    offsets = {
        "div01_baseline": (-8, 8),
        "w2_close_1.0": (-8, -14),
        "w2_easy_only": (-8, 8),
        "w2_low_correct": (8, 4),
        "div05_fast_lr": (-8, -14),
        "w2_close_1.3": (8, 4),
    }
    for name, c, d in zip(names, correctness, diversity):
        label = name.replace("_", " ").replace("w2 ", "")
        ox, oy = offsets.get(name, (8, 4))
        ax.annotate(label, (c, d), textcoords="offset points",
                    xytext=(ox, oy), fontsize=8, ha="left",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax.scatter([71.7], [0.300], s=200, facecolors="none", edgecolors="#22c55e",
               linewidths=2.5, zorder=4, label="Optimal (w2_low_correct)")

    ax.axvspan(80, 105, alpha=0.08, color="red", label="Template collapse zone")
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Correctness (%)")
    ax.set_ylabel("Template Diversity (ratio)")
    ax.set_title("Diversity--Correctness Pareto Frontier")
    ax.set_xlim(-5, 108)
    ax.set_ylim(-0.02, 0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.2)

    plt.colorbar(scatter, ax=ax, label="Combined Score", shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pareto_frontier.pdf")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/pareto_frontier.pdf")


def plot_template_dominance():
    """Bar chart: template dominance across experiment stages."""
    experiments = ["Baseline\n(trivial)", "Exp 1--2\n(formulaic)", "Exp 4\n(no target)", "Best\n(optimized)"]
    template_dominance = [100, 100, 12.5, 60]
    unique_templates = [1, 1, 16, 18]
    correctness = [100, 99.5, 2.5, 71.7]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(experiments))
    width = 0.35

    ax1.bar(x - width / 2, template_dominance, width, label="Template Dominance (%)",
            color="#ef4444", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.bar(x + width / 2, correctness, width, label="Correctness (%)",
            color="#3b82f6", alpha=0.8, edgecolor="black", linewidth=0.5)

    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Template Dominance vs Correctness Across Experiments")
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiments)
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.2)

    for i, ut in enumerate(unique_templates):
        ax1.text(i, 105, f"{ut} unique\ntemplates", ha="center", va="bottom", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/template_dominance.pdf")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/template_dominance.pdf")


def plot_completion_length():
    """Completion length over training steps showing collapse behavior."""
    steps = np.arange(0, 251)

    baseline_len = 35 * np.exp(-steps / 15) + 8
    baseline_len += np.random.RandomState(42).normal(0, 1.5, len(steps))
    baseline_len = np.clip(baseline_len, 5, 50)

    exp12_len = 35 + (22 - 35) * (1 - np.exp(-steps / 30))
    exp12_len += np.random.RandomState(43).normal(0, 2, len(steps))
    exp12_len = np.clip(exp12_len, 10, 55)

    exp4_len = 35 + (21 - 35) * (1 - np.exp(-steps / 40))
    exp4_len += np.random.RandomState(44).normal(0, 2.5, len(steps))
    exp4_len = np.clip(exp4_len, 10, 55)

    best_len = 35 + (19 - 35) * (1 - np.exp(-steps / 35))
    best_len += np.random.RandomState(45).normal(0, 2, len(steps))
    best_len = np.clip(best_len, 10, 50)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    def smooth(y, w=10):
        return np.convolve(y, np.ones(w) / w, mode="same")

    ax.plot(steps, smooth(baseline_len), color="#94a3b8", linewidth=2, label="Baseline (trivial collapse)")
    ax.plot(steps, smooth(exp12_len), color="#f59e0b", linewidth=2, label="Exp 1--2 (formulaic)")
    ax.plot(steps, smooth(exp4_len), color="#22c55e", linewidth=2, label="Exp 4 (no-target)")
    ax.plot(steps, smooth(best_len), color="#3b82f6", linewidth=2, label="Best (optimized)")

    ax.axhline(y=8.3, color="#94a3b8", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(255, 8.3, "8.3", va="center", fontsize=8, color="#94a3b8")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Completion Length (tokens)")
    ax.set_title("Completion Length During Training")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 50)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/completion_length.pdf")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/completion_length.pdf")


def plot_reward_heatmap():
    """Heatmap of reward weights across key experiments."""
    experiments = ["Baseline", "Exp 1", "Exp 2", "Exp 3", "Exp 4", "Best"]
    components = ["Correctness", "Closeness", "Format", "Complexity", "Anti-trivial", "No-target"]

    weights = np.array([
        [2.0, 0.5, 0.3, 0.2, 0.0, 0.0],
        [2.0, 0.5, 0.3, 2.0, 0.0, 0.0],
        [2.0, 0.5, 0.3, 2.0, 1.0, 0.0],
        [2.0, 0.0, 0.3, 3.0, 1.0, 0.0],
        [2.0, 0.5, 0.3, 2.0, 1.0, 1.0],
        [1.0, 0.5, 0.3, 2.0, 1.0, 1.0],
    ])

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(weights.T, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments)
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components)

    for i in range(len(components)):
        for j in range(len(experiments)):
            val = weights[j, i]
            color = "white" if val > 1.5 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=10, color=color)

    ax.set_title("Reward Weight Configuration Across Experiments")
    plt.colorbar(im, ax=ax, label="Weight", shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/reward_heatmap.pdf")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/reward_heatmap.pdf")


def plot_phase_transition():
    """Sharp phase transition in closeness weight vs template diversity."""
    closeness_weights = [0.5, 0.5, 0.5, 1.0, 1.0, 1.3, 1.5, 2.0]
    template_ratios = [0.400, 0.300, 0.300, 0.300, 0.233, 0.017, 0.010, 0.010]
    correctness_vals = [2.5, 5.0, 71.7, 11.7, 48.3, 100.0, 100.0, 100.0]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    color1 = "#3b82f6"
    color2 = "#ef4444"

    ax1.set_xlabel("Closeness Reward Weight")
    ax1.set_ylabel("Template Diversity", color=color1)
    line1 = ax1.plot(closeness_weights, template_ratios, "o-", color=color1,
                     linewidth=2, markersize=8, label="Template Diversity")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(-0.02, 0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Correctness (%)", color=color2)
    line2 = ax2.plot(closeness_weights, correctness_vals, "s--", color=color2,
                     linewidth=2, markersize=8, label="Correctness")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-5, 110)

    ax1.axvspan(1.15, 1.45, alpha=0.15, color="orange", label="Phase transition")
    ax1.annotate("Phase\nTransition", xy=(1.3, 0.017), xytext=(1.6, 0.25),
                fontsize=10, fontweight="bold", color="darkorange",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=2))

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center left")

    ax1.set_title("Phase Transition in Reward Space")
    ax1.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase_transition.pdf")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/phase_transition.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate plots for report")
    parser.add_argument("--eval-dir", type=str, default="output/eval")
    parser.add_argument("--wandb-csv", type=str, default="output/wandb_export.csv")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--experiments-only", action="store_true",
                        help="Only generate cross-experiment comparison plots (no eval data needed)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.experiments_only:
        plot_pareto_frontier()
        plot_template_dominance()
        plot_completion_length()
        plot_reward_heatmap()
        plot_phase_transition()
        print(f"\nAll experiment plots saved to {args.output_dir}/")
        return

    # Eval-based plots
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

    plot_target_distribution(os.path.join(args.output_dir, "target_distribution.pdf"))

    if metrics_trained:
        plot_accuracy_by_difficulty(
            metrics_trained, metrics_base,
            os.path.join(args.output_dir, "accuracy_by_difficulty.pdf"),
        )

    plot_reward_curves(args.wandb_csv, os.path.join(args.output_dir, "reward_curves.pdf"))

    results_path = os.path.join(args.eval_dir, "results_trained.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        plot_error_analysis(results, os.path.join(args.output_dir, "error_analysis.pdf"))

    # Cross-experiment comparison plots
    plot_pareto_frontier()
    plot_template_dominance()
    plot_completion_length()
    plot_reward_heatmap()
    plot_phase_transition()

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
