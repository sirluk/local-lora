from __future__ import annotations

import argparse
import os
from pathlib import Path

from local_lora.glue import run_glue_task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune/evaluate on a single GLUE task with local LoRA variants.")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--task", type=str, required=True)

    p.add_argument(
        "--adapter_type",
        type=str,
        choices=["head_only", "full_ft", "vanilla_lora", "group_local", "bd_lora"],
        required=True,
    )
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--m", type=int, default=None, help="Group-local subrank per group (required for group_local).")
    p.add_argument("--alpha", type=float, default=None, help="Defaults to r (standard) or sqrt(r) (rs) if unset.")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--scaling_mode", type=str, choices=["standard", "rs"], default="standard")
    p.add_argument("--grouping_mode", type=str, choices=["contiguous", "random", "head_aligned"], default="contiguous")
    p.add_argument("--perm_seed", type=int, default=None)
    p.add_argument("--bd_n", type=int, default=8, help="BD-LoRA block count N (used for adapter_type=bd_lora).")
    p.add_argument(
        "--bd_row_factor",
        type=str,
        choices=["block_a", "block_b", "dense"],
        default="block_a",
        help="BD-LoRA ablation for row-parallel-like layers (o_proj, down_proj).",
    )
    p.add_argument("--max_length", type=int, default=256)

    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--results_csv", type=str, default=None)

    p.add_argument("--target_suffixes", type=str, default="q_proj,k_proj,v_proj,o_proj")

    p.add_argument("--wandb_mode", type=str, choices=["disabled", "online", "offline"], default="disabled")
    p.add_argument("--wandb_entity", type=str, default="hauzenberger")
    p.add_argument("--wandb_project", type=str, default="local-lora")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target_suffixes = tuple(s.strip() for s in args.target_suffixes.split(",") if s.strip())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    wandb_tags = tuple(s.strip() for s in args.wandb_tags.split(",") if s.strip())

    run_glue_task(
        model_name=args.model_name,
        task=args.task,
        output_dir=args.output_dir,
        adapter_type=args.adapter_type,
        r=args.r,
        m=args.m,
        alpha=args.alpha,
        dropout=args.dropout,
        scaling_mode=args.scaling_mode,
        grouping_mode=args.grouping_mode,
        perm_seed=args.perm_seed,
        bd_n=args.bd_n,
        bd_row_factor=args.bd_row_factor,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        bf16=args.bf16,
        fp16=args.fp16,
        results_csv=args.results_csv,
        wandb_mode=args.wandb_mode,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_group=args.wandb_group,
        wandb_tags=wandb_tags,
        target_suffixes=target_suffixes,
    )


if __name__ == "__main__":
    main()
