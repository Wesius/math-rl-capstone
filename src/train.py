"""
GRPO training script for the math expression task.

Trains LFM2-350M (Liquid AI's hybrid conv+attention model) to generate
math expressions that evaluate to target numbers using Group Relative
Policy Optimization.
"""

import argparse

from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from src.dataset import build_dataset, build_eval_dataset
from src.reward import closeness_reward, complexity_reward, correctness_reward, format_reward


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
    parser.add_argument("--num-completions", type=int, default=8, help="Completions per prompt (G)")
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
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for generation")
    args = parser.parse_args()

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

    # GRPO config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        num_completions=args.num_completions,
        max_completion_length=args.max_completion_length,
        # Reward weighting: correctness dominates, closeness helps shape gradient,
        # format and complexity are minor bonuses
        reward_weights=[2.0, 0.5, 0.3, 0.2],
        log_completions=True,
        num_completions_to_print=3,
        seed=args.seed,
        report_to="none" if args.no_wandb else "wandb",
        run_name="grpo-lfm2-350m-math" if not args.no_wandb else None,
        use_vllm=args.use_vllm,
        save_total_limit=3,
    )

    # Initialize trainer
    print(f"Loading model: {args.model}")
    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[
            correctness_reward,
            closeness_reward,
            format_reward,
            complexity_reward,
        ],
        peft_config=peft_config,
    )

    print("Starting GRPO training...")
    print(f"  Model:       {args.model}")
    print(f"  LoRA rank:   {args.lora_rank}")
    print(f"  Batch:       {args.batch_size} x {args.grad_accum} grad accum")
    print(f"  Completions: {args.num_completions} per prompt")
    print(f"  LR:          {args.lr}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Output:      {args.output_dir}")
    print()

    trainer.train()

    # Save final model
    print("Saving final model...")
    trainer.save_model(f"{args.output_dir}/final")
    print(f"Done! Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
