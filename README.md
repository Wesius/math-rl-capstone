# Math Expression RL Capstone

RL fine-tuning of [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) (Liquid AI's hybrid conv+attention model) to generate mathematical expressions that evaluate to target numbers.

**Task**: Given a target number like `73`, the model outputs a math expression like `70 + 3` or `9 * 8 + 1`.

**Method**: Group Relative Policy Optimization (GRPO) with LoRA, using a verifiable reward function (`eval(expression) == target`).

## Project Structure

```
├── src/
│   ├── dataset.py     # Dataset generation (curriculum-distributed targets)
│   ├── reward.py      # Reward functions (correctness, closeness, format, complexity, no-target)
│   ├── train.py       # GRPO training script (configurable reward weights, anti-trivial, no-target)
│   ├── eval.py        # Evaluation and metrics
│   ├── diversity.py   # Template diversity analysis
│   └── plot.py        # All report figure generation (eval-based + cross-experiment)
├── report/
│   ├── report.tex     # LaTeX report (5-7 pages)
│   ├── references.bib # Bibliography
│   └── figures/       # Generated figures
└── output/            # Training checkpoints and eval results
```

## Setup

```bash
uv sync
```

## Usage

### Generate Dataset
```bash
uv run generate-data --n-train 10000 --n-eval 500
```

### Train
```bash
uv run train --model LiquidAI/LFM2-350M --epochs 3 --batch-size 4
```

With vLLM acceleration:
```bash
uv run train --model LiquidAI/LFM2-350M --use-vllm
```

### Evaluate
```bash
uv run eval --model-path output/grpo-lfm2-math/final --eval-base --verbose
```

### Generate Report Figures
```bash
uv run python -m src.plot --eval-dir output/eval
```

### Build Report
```bash
cd report && latexmk -pdf report.tex
```

## Reward Design

| Component   | Weight | Description                                |
|-------------|--------|--------------------------------------------|
| Correctness | 2.0    | Binary: eval(expr) == target               |
| Closeness   | 0.5    | Partial credit for being numerically close  |
| Format      | 0.3    | Uses `<expr></expr>` tags                  |
| Complexity  | 0.2    | Bonus for multi-operator expressions        |

## Key Details

- **Model**: LFM2-350M — 354M params, hybrid architecture (10 conv + 6 attention layers)
- **Training**: LoRA (rank 32, alpha 64) on attention + MLP layers
- **GRPO**: 8 completions per prompt, group-relative advantage normalization
- **Safe eval**: AST-based expression parser (no `eval()` calls)
