from __future__ import annotations

import argparse
import time
from pathlib import Path

from local_lora.glue import collect_projection_shapes, find_parameter_matched_r, run_glue_task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a GLUE suite comparing vanilla LoRA vs Group-Local LoRA.")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--tasks", type=str, default="sst2,mrpc,rte,qnli")
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--results_csv", type=str, default=None)

    p.add_argument(
        "--methods",
        type=str,
        default="head_only,vanilla_lora,group_local_equal,group_local_param",
        help=(
            "Comma-separated methods to run: head_only, vanilla_lora, group_local_equal, group_local_param. "
            "You can also use 'all' or 'group_local'."
        ),
    )
    p.add_argument("--r_base", type=int, default=16)
    p.add_argument("--m_values", type=str, default="16,8,4,2,1")

    p.add_argument("--alpha", type=float, default=None, help="Defaults to r for each run if unset.")
    p.add_argument("--dropout", type=float, default=0.0)
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

    p.add_argument("--target_suffixes", type=str, default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--max_r_search", type=int, default=512)

    p.add_argument("--wandb_mode", type=str, choices=["disabled", "online", "offline"], default="disabled")
    p.add_argument("--wandb_entity", type=str, default="hauzenberger")
    p.add_argument("--wandb_project", type=str, default="local-lora")
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    m_values = [int(x.strip()) for x in args.m_values.split(",") if x.strip()]
    target_suffixes = tuple(s.strip() for s in args.target_suffixes.split(",") if s.strip())
    wandb_tags = tuple(s.strip() for s in args.wandb_tags.split(",") if s.strip())

    methods_raw = [m.strip() for m in args.methods.split(",") if m.strip()]
    expanded: set[str] = set()
    for m in methods_raw:
        if m == "all":
            expanded.update({"head_only", "vanilla_lora", "group_local_equal", "group_local_param"})
        elif m == "group_local":
            expanded.update({"group_local_equal", "group_local_param"})
        else:
            expanded.add(m)
    valid = {"head_only", "vanilla_lora", "group_local_equal", "group_local_param"}
    unknown = sorted(expanded - valid)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid: {sorted(valid)} (or 'all'/'group_local').")

    run_head_only = "head_only" in expanded
    run_vanilla = "vanilla_lora" in expanded
    run_group_local_equal = "group_local_equal" in expanded
    run_group_local_param = "group_local_param" in expanded

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = args.results_csv or str(out_root / "results.csv")

    sum_d_in = None
    sum_d_out = None
    d_out_list = None
    if run_group_local_param:
        # Pre-compute shapes for parameter-matched search (same for all tasks).
        shapes = collect_projection_shapes(args.model_name, target_suffixes=target_suffixes)
        sum_d_in = sum(s["d_in"] for s in shapes)
        sum_d_out = sum(s["d_out"] for s in shapes)
        d_out_list = [s["d_out"] for s in shapes]

    ts = time.strftime("%Y%m%d_%H%M%S")
    wandb_group = args.wandb_group
    if args.wandb_mode != "disabled" and wandb_group is None:
        wandb_group = f"glue_suite_{ts}"

    def alpha_for(r: int) -> float:
        return float(args.alpha) if args.alpha is not None else float(r)

    for task in tasks:
        if run_head_only:
            run_glue_task(
                model_name=args.model_name,
                task=task,
                output_dir=str(out_root / f"{ts}_{task}_head_only"),
                adapter_type="head_only",
                r=args.r_base,
                m=None,
                alpha=alpha_for(args.r_base),
                dropout=args.dropout,
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
                results_csv=results_csv,
                wandb_mode=args.wandb_mode,
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                wandb_name=None,
                wandb_group=wandb_group,
                wandb_tags=wandb_tags,
                target_suffixes=target_suffixes,
            )

        if run_vanilla:
            run_glue_task(
                model_name=args.model_name,
                task=task,
                output_dir=str(out_root / f"{ts}_{task}_vanilla_lora_r{args.r_base}"),
                adapter_type="vanilla_lora",
                r=args.r_base,
                m=None,
                alpha=alpha_for(args.r_base),
                dropout=args.dropout,
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
                results_csv=results_csv,
                wandb_mode=args.wandb_mode,
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                wandb_name=None,
                wandb_group=wandb_group,
                wandb_tags=wandb_tags,
                target_suffixes=target_suffixes,
            )

        if run_group_local_equal:
            for m in m_values:
                run_glue_task(
                    model_name=args.model_name,
                    task=task,
                    output_dir=str(out_root / f"{ts}_{task}_group_local_equal_r{args.r_base}_m{m}"),
                    adapter_type="group_local",
                    r=args.r_base,
                    m=m,
                    alpha=alpha_for(args.r_base),
                    dropout=args.dropout,
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
                    results_csv=results_csv,
                    wandb_mode=args.wandb_mode,
                    wandb_entity=args.wandb_entity,
                    wandb_project=args.wandb_project,
                    wandb_name=None,
                    wandb_group=wandb_group,
                    wandb_tags=wandb_tags,
                    target_suffixes=target_suffixes,
                )

        if run_group_local_param:
            if sum_d_in is None or sum_d_out is None or d_out_list is None:
                raise RuntimeError("Internal error: parameter-matched shapes not computed.")
            for m in m_values:
                r_match, delta = find_parameter_matched_r(
                    sum_d_in=sum_d_in,
                    sum_d_out=sum_d_out,
                    r_base=args.r_base,
                    m=m,
                    d_out_list=d_out_list,
                    max_r=args.max_r_search,
                )
                run_glue_task(
                    model_name=args.model_name,
                    task=task,
                    output_dir=str(out_root / f"{ts}_{task}_group_local_param_match_r{r_match}_m{m}_d{delta}"),
                    adapter_type="group_local",
                    r=r_match,
                    m=m,
                    alpha=alpha_for(r_match),
                    dropout=args.dropout,
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
                    results_csv=results_csv,
                    wandb_mode=args.wandb_mode,
                    wandb_entity=args.wandb_entity,
                    wandb_project=args.wandb_project,
                    wandb_name=None,
                    wandb_group=wandb_group,
                    wandb_tags=wandb_tags,
                    target_suffixes=target_suffixes,
                )


if __name__ == "__main__":
    main()
