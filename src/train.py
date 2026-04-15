"""
GRPO training script for the math expression task.

Trains LFM2-350M (Liquid AI's hybrid conv+attention model) to generate
math expressions that evaluate to target numbers using Group Relative
Policy Optimization.

Supports configurable reward weights, anti-trivial mode, and no-target constraints.
"""

import argparse
import json

import torch
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from src.dataset import build_dataset, build_eval_dataset
from src.reward import (
    closeness_reward,
    complexity_reward,
    correctness_reward,
    format_reward,
    no_target_reward,
    nontrivial_reward,
)


# LoRA targets for LFM2 — the model uses a hybrid architecture with
# both attention and convolution blocks. We target the attention projections
# and the gated MLP layers.
LFM2_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def main():
    parser = argparse.ArgumentParser(description="GRPO training for math expression generation")
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2-350M", help="Model ID")
    parser.add_argument("--n-train", type=int, default=10000, help="Training set size")
    parser.add_argument("--n-eval", type=int, default=500, help="Eval set size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num-generations", type=int, default=8, help="Generations per prompt (G)")
    parser.add_argument("--max-completion-length", type=int, default=128, help="Max tokens per completion")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--output-dir", type=str, default="output/grpo-lfm2-math")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="math-rl-capstone")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bfloat16")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for generation")
    parser.add_argument(
        "--reward-weights",
        type=str,
        default="[2.0, 0.5, 0.3, 0.2]",
        help="JSON list of weights: [correctness, closeness, format, complexity, (ban-trivial), (no-target)]",
    )
    parser.add_argument(
        "--ban-trivial",
        action="store_true",
        help="Add nontrivial reward that penalizes single-number expressions",
    )
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Add reward that penalizes expressions containing the target number",
    )
    args = parser.parse_args()

    weights = json.loads(args.reward_weights)

    # Build datasets
    print("Building datasets...")
    train_dataset = build_dataset(n=args.n_train, seed=args.seed)
    eval_dataset = build_eval_dataset(n=args.n_eval, seed=args.seed + 57)
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Eval:  {len(eval_dataset)} examples")

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=LFM2_LORA_TARGETS,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # Build reward functions and weights
    reward_funcs = [correctness_reward, closeness_reward, format_reward, complexity_reward]
    reward_weights = list(weights[:4])

    if args.ban_trivial:
        reward_funcs.append(nontrivial_reward)
        reward_weights.append(weights[4] if len(weights) > 4 else 1.0)

    if args.no_target:
        reward_funcs.append(no_target_reward)
        idx = 5 if args.ban_trivial else 4
        reward_weights.append(weights[idx] if len(weights) > idx else 0.5)

    use_bf16 = not args.no_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # GRPO config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, args.num_generations),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        reward_weights=reward_weights,
        log_completions=True,
        num_completions_to_print=3,
        seed=args.seed,
        report_to="none" if args.no_wandb else "wandb",
        run_name="grpo-lfm2-350m-math" if not args.no_wandb else None,
        use_vllm=args.use_vllm,
        save_total_limit=3,
    )

    # Initialize trainer
    print("Starting GRPO training...")
    print(f"  Model:         {args.model}")
    print(f"  LoRA rank:     {args.lora_rank}")
    print(f"  Batch:         {args.batch_size} x {args.grad_accum} grad accum")
    print(f"  Generations:   {args.num_generations} per prompt")
    print(f"  LR:            {args.lr}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Reward weights: {reward_weights}")
    print(f"  Ban trivial:   {args.ban_trivial}")
    print(f"  No target:     {args.no_target}")
    print(f"  Output:        {args.output_dir}")
    print()

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
    )

    trainer.train()

    # Save final model
    print("Saving final model...")
    trainer.save_model(f"{args.output_dir}/final")
    print(f"Done! Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
