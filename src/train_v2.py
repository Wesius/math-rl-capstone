"""
GRPO training script v2 — supports configurable reward weights and anti-trivial mode.
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

LFM2_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2-350M")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="output/grpo-lfm2-math")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=9999)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--reward-weights", type=str, default="[2.0, 0.5, 0.3, 0.2]",
                        help="JSON list of weights: [correctness, closeness, format, complexity]")
    parser.add_argument("--ban-trivial", action="store_true",
                        help="Add nontrivial reward that penalizes single-number expressions")
    parser.add_argument("--no-target", action="store_true",
                        help="Add reward that penalizes expressions containing the target number")
    args = parser.parse_args()

    weights = json.loads(args.reward_weights)

    print(f"Experiment: {args.output_dir}")
    print(f"  Reward weights: {weights}")
    print(f"  Ban trivial: {args.ban_trivial}")

    train_dataset = build_dataset(n=args.n_train, seed=args.seed)
    eval_dataset = build_eval_dataset(n=args.n_eval, seed=args.seed + 57)
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

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
    reward_weights = weights[:4]

    if args.ban_trivial:
        reward_funcs.append(nontrivial_reward)
        reward_weights.append(weights[4] if len(weights) > 4 else 1.0)

    if args.no_target:
        reward_funcs.append(no_target_reward)
        reward_weights.append(weights[5] if len(weights) > 5 else 0.5)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

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
        num_completions_to_print=2,
        seed=args.seed,
        report_to="none" if args.no_wandb else "wandb",
        save_total_limit=1,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")
    print(f"Done! Saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
