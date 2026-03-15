from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from local_lora.glue import collect_projection_shapes, find_parameter_matched_r, run_glue_task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a GLUE suite comparing LoRA variants (vanilla, Group-Local, BD-LoRA) plus baselines."
    )
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument(
        "--model_names",
        type=str,
        default=None,
        help="Optional comma-separated list of model names/paths to run (overrides --model_name).",
    )
    p.add_argument("--tasks", type=str, default="sst2,mrpc,rte,qnli")
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--results_csv", type=str, default=None)
    p.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional tag used in output subdirectory names (useful for DDP to keep ranks consistent).",
    )

    p.add_argument(
        "--methods",
        type=str,
        default="head_only,vanilla_lora,group_local_equal,group_local_param",
        help=(
            "Comma-separated methods to run: head_only, full_ft, vanilla_lora, bd_lora, "
            "group_local_equal, group_local_param. You can also use 'all' or 'group_local'."
        ),
    )
    p.add_argument("--r_base", type=int, default=16)
    p.add_argument("--m_values", type=str, default="16,8,4,2,1")
    p.add_argument("--bd_n_values", type=str, default="8,4", help="BD-LoRA block counts N to evaluate.")
    p.add_argument(
        "--bd_row_factor",
        type=str,
        choices=["block_a", "block_b", "dense"],
        default="block_a",
        help="BD-LoRA ablation for row-parallel-like layers (o_proj, down_proj).",
    )

    p.add_argument("--alpha", type=float, default=None, help="Defaults to r for each run if unset.")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--scaling_mode", type=str, choices=["standard", "rs"], default="standard")
    p.add_argument("--grouping_mode", type=str, choices=["contiguous", "random", "head_aligned"], default="contiguous")
    p.add_argument("--perm_seed", type=int, default=None)
    p.add_argument(
        "--torch_compile",
        action="store_true",
        help="Enable torch.compile for the model (uses HF Trainer integration when available).",
    )
    p.add_argument("--torch_compile_backend", type=str, default=None, help="Optional torch.compile backend.")
    p.add_argument("--torch_compile_mode", type=str, default=None, help="Optional torch.compile mode.")

    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default=None, help="Optional comma-separated list of seeds (overrides --seed).")

    p.add_argument(
        "--learning_rates",
        type=str,
        default=None,
        help="Optional comma-separated list of learning rates (overrides --learning_rate).",
    )
    p.add_argument(
        "--full_ft_learning_rates",
        type=str,
        default=None,
        help="Optional comma-separated list of learning rates for full_ft (overrides --learning_rates).",
    )
    p.add_argument(
        "--max_lengths",
        type=str,
        default=None,
        help="Optional comma-separated list of max_length values (overrides --max_length).",
    )
    p.add_argument(
        "--warmup_ratios",
        type=str,
        default=None,
        help="Optional comma-separated list of warmup_ratio values (overrides --warmup_ratio).",
    )
    p.add_argument(
        "--scaling_modes",
        type=str,
        default=None,
        help="Optional comma-separated list of scaling_mode values (overrides --scaling_mode).",
    )

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

    def split_csv(s: str) -> list[str]:
        return [x.strip() for x in s.split(",") if x.strip()]

    def sanitize_token(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

    def fmt_sci(x: float) -> str:
        # For learning rates; keeps directory names short and stable.
        return sanitize_token(f"{float(x):.0e}".replace("+", ""))

    def fmt_frac(x: float) -> str:
        return sanitize_token(f"{float(x):g}".replace(".", "p"))

    tasks = split_csv(args.tasks)
    m_values = [int(x) for x in split_csv(args.m_values)]
    bd_n_values = [int(x) for x in split_csv(args.bd_n_values)]
    target_suffixes = tuple(split_csv(args.target_suffixes))
    wandb_tags = tuple(split_csv(args.wandb_tags))
    compile_kwargs = dict(
        torch_compile=bool(args.torch_compile),
        torch_compile_backend=args.torch_compile_backend,
        torch_compile_mode=args.torch_compile_mode,
    )

    def is_global_rank_zero() -> bool:
        # torchrun sets RANK; fall back to SLURM_PROCID when present.
        for k in ("RANK", "SLURM_PROCID"):
            v = os.environ.get(k)
            if v is None:
                continue
            try:
                return int(v) == 0
            except Exception:
                pass
        return True

    def env_flag(name: str, default: str = "1") -> bool:
        v = os.environ.get(name, default).strip().lower()
        return v not in {"0", "false", "no", "off", ""}

    print_run_config = env_flag("LOCAL_LORA_PRINT_RUN_CONFIG", "1") and is_global_rank_zero()

    def fmt_alpha(r: int, scaling_mode: str) -> float:
        if args.alpha is not None:
            return float(args.alpha)
        return float(r) if scaling_mode == "standard" else float(math.sqrt(float(r)))

    def log_event(event: str, payload: dict) -> None:
        if not print_run_config:
            return
        out = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event}
        out.update(payload)
        print(json.dumps(out, sort_keys=True), flush=True)

    model_names = split_csv(args.model_names) if args.model_names else [args.model_name]
    seeds = [int(x) for x in split_csv(args.seeds)] if args.seeds else [int(args.seed)]

    learning_rates = [float(x) for x in split_csv(args.learning_rates)] if args.learning_rates else [float(args.learning_rate)]
    full_ft_learning_rates = (
        [float(x) for x in split_csv(args.full_ft_learning_rates)]
        if args.full_ft_learning_rates
        else list(learning_rates)
    )
    max_lengths = [int(x) for x in split_csv(args.max_lengths)] if args.max_lengths else [int(args.max_length)]
    warmup_ratios = [float(x) for x in split_csv(args.warmup_ratios)] if args.warmup_ratios else [float(args.warmup_ratio)]
    scaling_modes = split_csv(args.scaling_modes) if args.scaling_modes else [str(args.scaling_mode)]

    bad_scaling = sorted(set(scaling_modes) - {"standard", "rs"})
    if bad_scaling:
        raise ValueError(f"Invalid scaling_modes: {bad_scaling}. Valid: ['standard', 'rs'].")

    if args.alpha is None and len(scaling_modes) > 1:
        # With the current defaults in local_lora.glue.run_glue_task:
        # - standard: alpha defaults to r, scaling = alpha/r = 1
        # - rs: alpha defaults to sqrt(r), scaling = alpha/sqrt(r) = 1
        # So sweeping scaling_modes without an explicit alpha is (almost) a no-op.
        print(
            "WARNING: sweeping --scaling_modes with --alpha unset. With current defaults this makes the effective "
            "LoRA scaling ~1.0 for both 'standard' and 'rs', so the sweep is likely redundant. "
            "If you want to compare scaling modes, set --alpha explicitly (same alpha across modes) or drop one mode."
        )

    methods_raw = split_csv(args.methods)
    expanded: set[str] = set()
    for m in methods_raw:
        if m == "all":
            expanded.update(
                {"head_only", "full_ft", "vanilla_lora", "bd_lora", "group_local_equal", "group_local_param"}
            )
        elif m == "group_local":
            expanded.update({"group_local_equal", "group_local_param"})
        else:
            expanded.add(m)
    valid = {"head_only", "full_ft", "vanilla_lora", "bd_lora", "group_local_equal", "group_local_param"}
    unknown = sorted(expanded - valid)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid: {sorted(valid)} (or 'all'/'group_local').")

    run_head_only = "head_only" in expanded
    run_full_ft = "full_ft" in expanded
    run_vanilla = "vanilla_lora" in expanded
    run_bd_lora = "bd_lora" in expanded
    run_group_local_equal = "group_local_equal" in expanded
    run_group_local_param = "group_local_param" in expanded

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = args.results_csv or str(out_root / "results.csv")

    run_tag = args.run_tag or time.strftime("%Y%m%d_%H%M%S")
    ts = sanitize_token(run_tag) or "run"

    wandb_group = args.wandb_group
    if args.wandb_mode != "disabled" and wandb_group is None:
        wandb_group = f"glue_suite_{ts}"

    log_event(
        "suite_start",
        {
            "output_root": str(out_root),
            "results_csv": str(results_csv),
            "run_tag": str(run_tag),
            "model_names": list(model_names),
            "tasks": list(tasks),
            "seeds": list(seeds),
            "methods": str(args.methods),
            "learning_rates": list(learning_rates),
            "full_ft_learning_rates": list(full_ft_learning_rates),
            "max_lengths": list(max_lengths),
            "warmup_ratios": list(warmup_ratios),
            "scaling_modes": list(scaling_modes),
            "target_suffixes": list(target_suffixes),
            "grouping_mode": str(args.grouping_mode),
            "perm_seed": args.perm_seed,
            "bd_n_values": list(bd_n_values),
            "bd_row_factor": str(args.bd_row_factor),
            "m_values": list(m_values),
            "wandb_mode": str(args.wandb_mode),
            "wandb_group": str(wandb_group) if wandb_group is not None else None,
            "torch_compile": bool(args.torch_compile),
            "torch_compile_backend": args.torch_compile_backend,
            "torch_compile_mode": args.torch_compile_mode,
        },
    )

    def model_tag(model_name: str) -> str:
        p = Path(model_name)
        if p.exists():
            return sanitize_token(p.name)
        return sanitize_token(model_name.split("/")[-1])

    def make_out_dir(
        *,
        model_name: str,
        task: str,
        method_tag: str,
        seed: int,
        lr: float,
        max_length: int,
        warmup_ratio: float,
        scaling_mode: str,
    ) -> str:
        parts = [ts, model_tag(model_name), task, method_tag]
        if len(seeds) > 1:
            parts.append(f"seed{seed}")
        if (len(learning_rates) > 1) or (len(full_ft_learning_rates) > 1):
            parts.append(f"lr{fmt_sci(lr)}")
        if len(max_lengths) > 1:
            parts.append(f"len{int(max_length)}")
        if len(warmup_ratios) > 1:
            parts.append(f"wu{fmt_frac(warmup_ratio)}")
        if len(scaling_modes) > 1:
            parts.append(f"sc{scaling_mode}")
        return str(out_root / "_".join(parts))

    for model_name in model_names:
        sum_d_in = None
        sum_d_out = None
        d_out_list = None
        if run_group_local_param:
            shapes = collect_projection_shapes(model_name, target_suffixes=target_suffixes)
            sum_d_in = sum(s["d_in"] for s in shapes)
            sum_d_out = sum(s["d_out"] for s in shapes)
            d_out_list = [s["d_out"] for s in shapes]

        def run_one(
            *,
            task: str,
            seed: int,
            max_length: int,
            warmup_ratio: float,
            scaling_mode: str,
            lr: float,
            adapter_type: str,
            method_tag: str,
            r: int,
            m: int | None,
            bd_n: int | None,
            bd_row_factor: str,
            group_local_delta: int | None = None,
        ) -> None:
            output_dir = make_out_dir(
                model_name=model_name,
                task=task,
                method_tag=method_tag,
                seed=seed,
                lr=lr,
                max_length=max_length,
                warmup_ratio=warmup_ratio,
                scaling_mode=scaling_mode,
            )

            log_event(
                "run_start",
                {
                    "model_name": model_name,
                    "task": task,
                    "seed": int(seed),
                    "adapter_type": adapter_type,
                    "method_tag": method_tag,
                    "r": int(r),
                    "m": None if m is None else int(m),
                    "n": None if bd_n is None else int(bd_n),
                    "group_local_delta": None if group_local_delta is None else int(group_local_delta),
                    "bd_row_factor": bd_row_factor if adapter_type == "bd_lora" else None,
                    "scaling_mode": scaling_mode,
                    "alpha": fmt_alpha(int(r), scaling_mode),
                    "dropout": float(args.dropout),
                    "lr": float(lr),
                    "max_length": int(max_length),
                    "warmup_ratio": float(warmup_ratio),
                    "per_device_train_batch_size": int(args.per_device_train_batch_size),
                    "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
                    "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                    "max_train_samples": args.max_train_samples,
                    "max_eval_samples": args.max_eval_samples,
                    "bf16": bool(args.bf16),
                    "fp16": bool(args.fp16),
                    "target_suffixes": list(target_suffixes),
                    "grouping_mode": str(args.grouping_mode),
                    "perm_seed": args.perm_seed,
                    "torch_compile": bool(args.torch_compile),
                    "output_dir": output_dir,
                    "wandb_mode": str(args.wandb_mode),
                },
            )

            result = run_glue_task(
                model_name=model_name,
                task=task,
                output_dir=output_dir,
                adapter_type=adapter_type,
                r=r,
                m=m,
                alpha=args.alpha,
                dropout=args.dropout,
                scaling_mode=scaling_mode,
                grouping_mode=args.grouping_mode,
                perm_seed=args.perm_seed,
                bd_n=bd_n,
                bd_row_factor=bd_row_factor,
                max_length=max_length,
                learning_rate=lr,
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                weight_decay=args.weight_decay,
                warmup_ratio=warmup_ratio,
                seed=seed,
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
                **compile_kwargs,
                target_suffixes=target_suffixes,
            )

            cfg = (result or {}).get("config", {})
            tps = cfg.get("trainable_param_summary") or {}
            inj = cfg.get("injection_report") or {}
            injected = inj.get("injected") if isinstance(inj, dict) else None
            injected_adapter_params = None
            injected_layers = None
            if isinstance(injected, list):
                injected_layers = len(injected)
                try:
                    injected_adapter_params = sum(int(x.get("adapter_params", 0)) for x in injected if isinstance(x, dict))
                except Exception:
                    injected_adapter_params = None

            trainable_params = tps.get("trainable_params")
            adapter_trainable_params = tps.get("adapter_trainable_params")
            non_adapter_trainable_params = None
            try:
                if trainable_params is not None and adapter_trainable_params is not None:
                    non_adapter_trainable_params = int(trainable_params) - int(adapter_trainable_params)
            except Exception:
                non_adapter_trainable_params = None

            adapter_trainable_to_injected_ratio = None
            try:
                if (
                    injected_adapter_params is not None
                    and int(injected_adapter_params) > 0
                    and adapter_trainable_params is not None
                ):
                    adapter_trainable_to_injected_ratio = float(int(adapter_trainable_params)) / float(
                        int(injected_adapter_params)
                    )
            except Exception:
                adapter_trainable_to_injected_ratio = None

            split_scores = {}
            metrics = (result or {}).get("metrics", {}) or {}
            if isinstance(metrics, dict):
                for split, md in metrics.items():
                    if isinstance(md, dict) and "score" in md:
                        split_scores[str(split)] = float(md["score"])

            log_event(
                "run_end",
                {
                    "output_dir": cfg.get("output_dir", output_dir),
                    "adapter_type": adapter_type,
                    "method_tag": method_tag,
                    "train_time_s": cfg.get("train_time_s"),
                    "trainable_params": trainable_params,
                    "adapter_trainable_params": adapter_trainable_params,
                    "non_adapter_trainable_params": non_adapter_trainable_params,
                    "injected_layers": injected_layers,
                    "injected_adapter_params": injected_adapter_params,
                    "adapter_trainable_to_injected_ratio": adapter_trainable_to_injected_ratio,
                    "score_by_split": split_scores,
                },
            )

        for task in tasks:
            for seed in seeds:
                for max_length in max_lengths:
                    for warmup_ratio in warmup_ratios:
                        for scaling_mode in scaling_modes:
                            # Avoid redundant sweeps for non-LoRA baselines.
                            if run_head_only and scaling_mode == scaling_modes[0]:
                                for lr in learning_rates:
                                    run_one(
                                        task=task,
                                        seed=seed,
                                        max_length=max_length,
                                        warmup_ratio=warmup_ratio,
                                        scaling_mode=scaling_modes[0],
                                        lr=lr,
                                        adapter_type="head_only",
                                        method_tag="head_only",
                                        r=args.r_base,
                                        m=None,
                                        bd_n=None,
                                        bd_row_factor=args.bd_row_factor,
                                    )

                            if run_full_ft and scaling_mode == scaling_modes[0]:
                                for lr in full_ft_learning_rates:
                                    run_one(
                                        task=task,
                                        seed=seed,
                                        max_length=max_length,
                                        warmup_ratio=warmup_ratio,
                                        scaling_mode=scaling_modes[0],
                                        lr=lr,
                                        adapter_type="full_ft",
                                        method_tag="full_ft",
                                        r=args.r_base,
                                        m=None,
                                        bd_n=None,
                                        bd_row_factor=args.bd_row_factor,
                                    )

                            if run_vanilla:
                                for lr in learning_rates:
                                    run_one(
                                        task=task,
                                        seed=seed,
                                        max_length=max_length,
                                        warmup_ratio=warmup_ratio,
                                        scaling_mode=scaling_mode,
                                        lr=lr,
                                        adapter_type="vanilla_lora",
                                        method_tag=f"vanilla_lora_r{args.r_base}",
                                        r=args.r_base,
                                        m=None,
                                        bd_n=None,
                                        bd_row_factor=args.bd_row_factor,
                                    )

                            if run_bd_lora:
                                for n in bd_n_values:
                                    for lr in learning_rates:
                                        run_one(
                                            task=task,
                                            seed=seed,
                                            max_length=max_length,
                                            warmup_ratio=warmup_ratio,
                                            scaling_mode=scaling_mode,
                                            lr=lr,
                                            adapter_type="bd_lora",
                                            method_tag=f"bd_lora_r{args.r_base}_n{n}_row{args.bd_row_factor}",
                                            r=args.r_base,
                                            m=None,
                                            bd_n=n,
                                            bd_row_factor=args.bd_row_factor,
                                        )

                            if run_group_local_equal:
                                for m in m_values:
                                    for lr in learning_rates:
                                        run_one(
                                            task=task,
                                            seed=seed,
                                            max_length=max_length,
                                            warmup_ratio=warmup_ratio,
                                            scaling_mode=scaling_mode,
                                            lr=lr,
                                            adapter_type="group_local",
                                            method_tag=f"group_local_equal_r{args.r_base}_m{m}",
                                            r=args.r_base,
                                            m=m,
                                            bd_n=None,
                                            bd_row_factor=args.bd_row_factor,
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
                                    for lr in learning_rates:
                                        run_one(
                                            task=task,
                                            seed=seed,
                                            max_length=max_length,
                                            warmup_ratio=warmup_ratio,
                                            scaling_mode=scaling_mode,
                                            lr=lr,
                                            adapter_type="group_local",
                                            method_tag=f"group_local_param_r{r_match}_m{m}_d{delta}",
                                            r=r_match,
                                            m=m,
                                            bd_n=None,
                                            bd_row_factor=args.bd_row_factor,
                                            group_local_delta=delta,
                                        )

    log_event("suite_end", {"status": "ok"})


if __name__ == "__main__":
    main()
