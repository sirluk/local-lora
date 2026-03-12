from __future__ import annotations

import csv
import inspect
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .adapters import adapter_state_dict
from .inject import (
    freeze_model_params,
    inject_adapters,
    inject_bd_lora,
    trainable_param_summary,
    unfreeze_adapter_params,
    unfreeze_sequence_classification_head,
)


TASK_TO_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def _patch_transformers_output_capturing_torch_global() -> None:
    """
    Work around a transformers bug that can surface under torch.compile:

        NameError: name 'torch' is not defined

    Some transformers versions import `transformers.utils.output_capturing` in a
    way that leaves the module-level `torch` symbol undefined even when PyTorch
    is installed. The wrapper there later references `torch`, crashing at
    runtime. Injecting the already-imported torch module into that namespace is
    safe and fixes the issue.
    """

    try:
        import transformers.utils.output_capturing as output_capturing  # type: ignore
    except Exception:
        return

    if "torch" not in getattr(output_capturing, "__dict__", {}):
        output_capturing.torch = torch  # type: ignore[attr-defined]


def _raise_with_hf_auth_help(model_name: str, err: OSError) -> None:
    msg = str(err).lower()
    if "gated repo" not in msg and "gatedrepoerror" not in msg:
        return

    hf_home = os.environ.get("HF_HOME")
    hf_home_note = ""
    if hf_home:
        hf_home_note = (
            f"\nNote: HF_HOME is set to {hf_home!r}. If you override HF_HOME, a prior login token in "
            "~/.cache/huggingface may not be visible; export HF_TOKEN explicitly (or place a token at $HF_HOME/token)."
        )

    raise OSError(
        f"Cannot download {model_name!r} from Hugging Face (gated repo; authentication required).\n"
        "Request access to the model and set HF_TOKEN (or run `huggingface-cli login`) for this job."
        f"{hf_home_note}"
    ) from err


def glue_primary_score(task: str, metrics: Dict[str, float]) -> float:
    task = task.lower()
    if task == "cola":
        return float(metrics["matthews_correlation"])
    if task in {"sst2", "mnli", "qnli", "rte", "wnli"}:
        return float(metrics["accuracy"])
    if task in {"mrpc", "qqp"}:
        return 0.5 * (float(metrics["accuracy"]) + float(metrics["f1"]))
    if task == "stsb":
        return 0.5 * (float(metrics["pearson"]) + float(metrics["spearmanr"]))
    raise ValueError(f"Unknown GLUE task for primary score: {task}")


def _num_labels_from_dataset(ds: DatasetDict) -> int:
    feat = ds["train"].features["label"]
    if isinstance(feat, ClassLabel):
        return int(feat.num_classes)
    return 1


def _tokenize_dataset(
    ds: DatasetDict,
    tokenizer,
    task: str,
    max_length: int,
) -> DatasetDict:
    key1, key2 = TASK_TO_KEYS[task]

    def tok(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        if key2 is None:
            return tokenizer(batch[key1], truncation=True, max_length=max_length)
        return tokenizer(batch[key1], batch[key2], truncation=True, max_length=max_length)

    remove_cols = [c for c in ds["train"].column_names if c not in {key1, key2, "label"}]
    return ds.map(tok, batched=True, remove_columns=remove_cols)


def _limit_samples(dataset: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    if max_samples is None:
        return dataset
    max_samples_i = int(max_samples)
    if max_samples_i <= 0:
        return dataset.select([])
    if len(dataset) <= max_samples_i:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples_i))


def _ensure_pad_token(tokenizer, model) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _build_compute_metrics(task: str):
    metric = load_metric("glue", task)

    def compute(eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        if task == "stsb":
            preds = logits.squeeze()
        else:
            preds = logits.argmax(axis=-1)
        out = metric.compute(predictions=preds, references=labels)
        out["score"] = glue_primary_score(task, out)
        return out

    return compute


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _append_results_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = path.exists()
    needs_header = (not exists) or (path.stat().st_size == 0)
    if exists:
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header and header != fieldnames:
            raise ValueError(
                f"results_csv schema mismatch for {str(path)!r}. Existing header has {len(header)} columns, "
                f"but this run wants {len(fieldnames)} columns. Use a fresh results_csv path (or delete the old file)."
            )
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def run_glue_task(
    *,
    model_name: str,
    task: str,
    output_dir: str,
    adapter_type: str,
    r: int,
    m: Optional[int],
    alpha: Optional[float],
    dropout: float,
    scaling_mode: str = "standard",
    grouping_mode: str = "contiguous",
    perm_seed: Optional[int] = None,
    bd_n: Optional[int] = None,
    bd_row_factor: str = "block_a",
    torch_compile: bool = False,
    torch_compile_backend: Optional[str] = None,
    torch_compile_mode: Optional[str] = None,
    max_length: int,
    learning_rate: float,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
    bf16: bool,
    fp16: bool,
    results_csv: Optional[str],
    wandb_mode: str = "disabled",
    wandb_entity: str = "hauzenberger",
    wandb_project: str = "local-lora",
    wandb_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_tags: Sequence[str] = (),
    target_suffixes: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> Dict[str, Any]:
    task = task.lower()
    if task not in TASK_TO_KEYS:
        raise ValueError(f"Unknown GLUE task: {task}")

    if torch_compile:
        _patch_transformers_output_capturing_torch_global()

    adapter_type = adapter_type.lower()
    scaling_mode = scaling_mode.lower()
    grouping_mode = grouping_mode.lower()

    if scaling_mode not in {"standard", "rs"}:
        raise ValueError(f"scaling_mode must be one of standard/rs, got {scaling_mode!r}")
    if grouping_mode not in {"contiguous", "random", "head_aligned"}:
        raise ValueError(f"grouping_mode must be one of contiguous/random/head_aligned, got {grouping_mode!r}")

    if alpha is None:
        alpha_v = float(r) if scaling_mode == "standard" else float(math.sqrt(float(r)))
    else:
        alpha_v = float(alpha)

    if grouping_mode == "random" and perm_seed is None:
        # Default to a fixed permutation across seeds unless explicitly overridden.
        perm_seed = 0

    set_seed(seed)

    ds = load_dataset("glue", task)
    num_labels = _num_labels_from_dataset(ds)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    except OSError as e:
        _raise_with_hf_auth_help(model_name, e)
        raise

    if task == "stsb":
        model.config.problem_type = "regression"

    _ensure_pad_token(tokenizer, model)
    model.config.use_cache = False

    head_dim = None
    if grouping_mode == "head_aligned":
        head_dim = getattr(model.config, "head_dim", None)
        if head_dim is None:
            hidden = getattr(model.config, "hidden_size", None)
            n_heads = getattr(model.config, "num_attention_heads", None)
            if isinstance(hidden, int) and isinstance(n_heads, int) and n_heads > 0 and (hidden % n_heads) == 0:
                head_dim = hidden // n_heads
        if head_dim is None:
            raise ValueError(
                "grouping_mode='head_aligned' requires model.config.head_dim or "
                "(hidden_size and num_attention_heads with hidden_size % num_attention_heads == 0)."
            )

    injection_report = None
    if adapter_type in {"head_only", "full_ft"}:
        pass
    elif adapter_type == "vanilla_lora":
        injection_report = inject_adapters(
            model,
            adapter_type="vanilla_lora",
            r=r,
            m=None,
            alpha=alpha_v,
            dropout=dropout,
            scaling_mode=scaling_mode,
            grouping_mode=grouping_mode,
            perm_seed=perm_seed,
            head_dim=int(head_dim) if head_dim is not None else None,
            target_suffixes=target_suffixes,
        )
    elif adapter_type == "group_local":
        if m is None:
            raise ValueError("m is required for adapter_type=group_local")
        injection_report = inject_adapters(
            model,
            adapter_type="group_local",
            r=r,
            m=m,
            alpha=alpha_v,
            dropout=dropout,
            scaling_mode=scaling_mode,
            grouping_mode=grouping_mode,
            perm_seed=perm_seed,
            head_dim=int(head_dim) if head_dim is not None else None,
            target_suffixes=target_suffixes,
        )
    elif adapter_type == "bd_lora":
        if bd_n is None:
            raise ValueError("bd_n is required for adapter_type=bd_lora")
        bd_row_factor = bd_row_factor.lower()
        if bd_row_factor not in {"block_a", "block_b", "dense"}:
            raise ValueError(f"bd_row_factor must be one of block_a/block_b/dense, got {bd_row_factor!r}")
        injection_report = inject_bd_lora(
            model,
            r=r,
            n=int(bd_n),
            alpha=alpha_v,
            dropout=dropout,
            scaling_mode=scaling_mode,
            grouping_mode=grouping_mode,
            perm_seed=perm_seed,
            bd_row_factor=bd_row_factor,
            target_suffixes=target_suffixes,
            head_dim=int(head_dim) if head_dim is not None else None,
        )
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")

    m_eff = int(m) if m is not None else (int(r) // int(bd_n) if adapter_type == "bd_lora" and bd_n else None)

    if adapter_type != "full_ft":
        freeze_model_params(model)
        unfreeze_sequence_classification_head(model)
        if adapter_type in {"vanilla_lora", "group_local", "bd_lora"}:
            unfreeze_adapter_params(model)

    tok_ds = _tokenize_dataset(ds, tokenizer=tokenizer, task=task, max_length=max_length)

    train_ds = _limit_samples(tok_ds["train"], max_train_samples, seed=seed)

    if task == "mnli":
        eval_splits = ["validation_matched", "validation_mismatched"]
    else:
        eval_splits = ["validation"]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = _build_compute_metrics(task)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if wandb_mode not in {"disabled", "online", "offline"}:
        raise ValueError(f"wandb_mode must be one of disabled/online/offline, got {wandb_mode}")

    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_params = ta_sig.parameters

    eval_strategy_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    args_kwargs: Dict[str, Any] = {
        "output_dir": str(out_dir / "trainer_out"),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "per_device_eval_batch_size": int(per_device_eval_batch_size),
        "learning_rate": float(learning_rate),
        "num_train_epochs": float(num_train_epochs),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "weight_decay": float(weight_decay),
        "warmup_ratio": float(warmup_ratio),
        "logging_strategy": "steps",
        "logging_steps": 50,
        eval_strategy_key: "no",
        "save_strategy": "no",
        "report_to": (["wandb"] if wandb_mode != "disabled" else []),
        "run_name": (wandb_name or out_dir.name) if wandb_mode != "disabled" else None,
        "bf16": bool(bf16),
        "fp16": bool(fp16),
        "seed": int(seed),
    }

    # torch.compile support:
    manual_compile = False
    if torch_compile:
        if "torch_compile" in ta_params:
            # Prefer HF-native compilation (handles device placement / DDP integration).
            args_kwargs["torch_compile"] = True
            if torch_compile_backend is not None and "torch_compile_backend" in ta_params:
                args_kwargs["torch_compile_backend"] = str(torch_compile_backend)
            if torch_compile_mode is not None and "torch_compile_mode" in ta_params:
                args_kwargs["torch_compile_mode"] = str(torch_compile_mode)
        else:
            manual_compile = True

    args = TrainingArguments(**args_kwargs)

    if manual_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile requested, but this PyTorch build has no torch.compile.")
        compile_kwargs = {}
        if torch_compile_backend is not None:
            compile_kwargs["backend"] = str(torch_compile_backend)
        if torch_compile_mode is not None:
            compile_kwargs["mode"] = str(torch_compile_mode)
        model = torch.compile(model, **compile_kwargs)  # type: ignore[attr-defined]

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    is_main = trainer.is_world_process_zero()

    wandb_run = None
    if wandb_mode != "disabled" and is_main:
        try:
            import wandb  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("wandb is not installed. `pip install wandb` to enable W&B logging.") from e

        run_name = wandb_name or out_dir.name
        base_tags = ["glue", task, adapter_type]
        if adapter_type in {"vanilla_lora", "group_local", "bd_lora"}:
            base_tags.append(f"r{int(r)}")
        if adapter_type == "group_local" and m is not None:
            base_tags.append(f"m{int(m)}")
        if adapter_type == "bd_lora" and bd_n is not None:
            base_tags.append(f"n{int(bd_n)}")
            base_tags.append(f"row_{bd_row_factor}")
        if scaling_mode != "standard":
            base_tags.append(f"scaling_{scaling_mode}")
        if grouping_mode != "contiguous":
            base_tags.append(f"grouping_{grouping_mode}")
        merged_tags = [t for t in list(wandb_tags) + base_tags if t]
        tags_dedup = list(dict.fromkeys(merged_tags))

        cfg_pre = {
            "model_name": model_name,
            "task": task,
            "adapter_type": adapter_type,
            "r": int(r),
            "m": m_eff,
            "n": int(bd_n) if adapter_type == "bd_lora" and bd_n is not None else None,
            "bd_row_factor": bd_row_factor if adapter_type == "bd_lora" else None,
            "scaling_mode": scaling_mode,
            "grouping_mode": grouping_mode,
            "perm_seed": int(perm_seed) if perm_seed is not None else None,
            "torch_compile": bool(torch_compile),
            "torch_compile_backend": str(torch_compile_backend) if torch_compile_backend is not None else None,
            "torch_compile_mode": str(torch_compile_mode) if torch_compile_mode is not None else None,
            "alpha": float(alpha_v),
            "dropout": float(dropout),
            "max_length": int(max_length),
            "learning_rate": float(learning_rate),
            "num_train_epochs": float(num_train_epochs),
            "per_device_train_batch_size": int(per_device_train_batch_size),
            "per_device_eval_batch_size": int(per_device_eval_batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "weight_decay": float(weight_decay),
            "warmup_ratio": float(warmup_ratio),
            "seed": int(seed),
            "max_train_samples": int(max_train_samples) if max_train_samples is not None else None,
            "max_eval_samples": int(max_eval_samples) if max_eval_samples is not None else None,
            "bf16": bool(bf16),
            "fp16": bool(fp16),
            "target_suffixes": list(target_suffixes),
            "trainable_param_summary": trainable_param_summary(model),
            "injection_report": asdict(injection_report) if injection_report is not None else None,
        }

        wandb_run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=run_name,
            group=wandb_group,
            tags=tags_dedup,
            config=cfg_pre,
            mode=wandb_mode,
            reinit=True,
        )

    try:
        t0 = time.time()
        trainer.train()
        train_time_s = time.time() - t0

        all_metrics: Dict[str, Dict[str, float]] = {}
        for split in eval_splits:
            eval_ds = _limit_samples(tok_ds[split], max_eval_samples, seed=seed)
            metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{split}")
            # Trainer prefixes keys with "eval_<split>_"
            prefix = f"eval_{split}_"
            cleaned = {k.replace(prefix, ""): float(v) for k, v in metrics.items() if k.startswith(prefix)}
            all_metrics[split] = cleaned

        if is_main:
            adapter_sd = adapter_state_dict(model)
            torch.save(adapter_sd, out_dir / "adapter.pt")

        cfg = {
            "model_name": model_name,
            "task": task,
            "adapter_type": adapter_type,
            "r": int(r),
            "m": m_eff,
            "n": int(bd_n) if adapter_type == "bd_lora" and bd_n is not None else None,
            "bd_row_factor": bd_row_factor if adapter_type == "bd_lora" else None,
            "scaling_mode": scaling_mode,
            "grouping_mode": grouping_mode,
            "perm_seed": int(perm_seed) if perm_seed is not None else None,
            "torch_compile": bool(torch_compile),
            "torch_compile_backend": str(torch_compile_backend) if torch_compile_backend is not None else None,
            "torch_compile_mode": str(torch_compile_mode) if torch_compile_mode is not None else None,
            "alpha": float(alpha_v),
            "dropout": float(dropout),
            "max_length": int(max_length),
            "learning_rate": float(learning_rate),
            "num_train_epochs": float(num_train_epochs),
            "per_device_train_batch_size": int(per_device_train_batch_size),
            "per_device_eval_batch_size": int(per_device_eval_batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "weight_decay": float(weight_decay),
            "warmup_ratio": float(warmup_ratio),
            "seed": int(seed),
            "max_train_samples": int(max_train_samples) if max_train_samples is not None else None,
            "max_eval_samples": int(max_eval_samples) if max_eval_samples is not None else None,
            "bf16": bool(bf16),
            "fp16": bool(fp16),
            "target_suffixes": list(target_suffixes),
            "train_time_s": float(train_time_s),
            "trainable_param_summary": trainable_param_summary(model),
            "injection_report": asdict(injection_report) if injection_report is not None else None,
        }
        if is_main:
            _write_json(out_dir / "config.json", cfg)
            _write_json(out_dir / "metrics.json", all_metrics)

        # Append results row(s)
        if is_main and results_csv is not None:
            results_path = Path(results_csv)
            run_id = out_dir.name
            ts = int(time.time())

            for split, metrics in all_metrics.items():
                row = {
                    "timestamp": ts,
                    "run_id": run_id,
                    "model": model_name,
                    "task": task,
                    "split": split,
                    "adapter_type": adapter_type,
                    "r": int(r),
                    "m": int(m_eff) if m_eff is not None else "",
                    "n": int(bd_n) if adapter_type == "bd_lora" and bd_n is not None else "",
                    "bd_row_factor": bd_row_factor if adapter_type == "bd_lora" else "",
                    "scaling_mode": scaling_mode,
                    "grouping_mode": grouping_mode,
                    "perm_seed": int(perm_seed) if perm_seed is not None else "",
                    "torch_compile": int(1 if torch_compile else 0),
                    "torch_compile_backend": str(torch_compile_backend) if torch_compile_backend is not None else "",
                    "torch_compile_mode": str(torch_compile_mode) if torch_compile_mode is not None else "",
                    "target_suffixes_json": json.dumps(list(target_suffixes)),
                    "alpha": float(alpha_v),
                    "dropout": float(dropout),
                    "learning_rate": float(learning_rate),
                    "num_train_epochs": float(num_train_epochs),
                    "max_length": int(max_length),
                    "warmup_ratio": float(warmup_ratio),
                    "weight_decay": float(weight_decay),
                    "max_train_samples": int(max_train_samples) if max_train_samples is not None else "",
                    "max_eval_samples": int(max_eval_samples) if max_eval_samples is not None else "",
                    "seed": int(seed),
                    "score": float(metrics.get("score", float("nan"))),
                    "metrics_json": json.dumps(metrics, sort_keys=True),
                    "train_time_s": float(train_time_s),
                    "trainable_params": int(cfg["trainable_param_summary"]["trainable_params"]),
                    "adapter_trainable_params": int(cfg["trainable_param_summary"]["adapter_trainable_params"]),
                }
                _append_results_csv(results_path, row)

        if wandb_run is not None and is_main:
            import wandb  # type: ignore

            wandb.log({"train/train_time_s": float(train_time_s)})

        return {
            "output_dir": str(out_dir),
            "metrics": all_metrics,
            "config": cfg,
        }
    finally:
        if wandb_run is not None:
            import wandb  # type: ignore

            wandb.finish()

    raise RuntimeError("Unreachable")


def collect_projection_shapes(model_name: str, target_suffixes: Sequence[str]) -> List[Dict[str, int]]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel

    # Instantiate on meta device to avoid allocating model weights (we only need module shapes).
    try:
        config = AutoConfig.from_pretrained(model_name)
    except OSError as e:
        _raise_with_hf_auth_help(model_name, e)
        raise

    with init_empty_weights():
        model = AutoModel.from_config(config)
    shapes = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(name.endswith(suf) for suf in target_suffixes):
            shapes.append({"d_in": int(module.in_features), "d_out": int(module.out_features)})
    return shapes


def find_parameter_matched_r(
    *,
    sum_d_in: int,
    sum_d_out: int,
    r_base: int,
    m: int,
    d_out_list: Sequence[int],
    max_r: int = 512,
) -> Tuple[int, int]:
    pv = int(r_base) * (int(sum_d_in) + int(sum_d_out))

    best_r: Optional[int] = None
    best_delta: Optional[int] = None

    for r in range(1, int(max_r) + 1):
        if r % m != 0:
            continue
        g = r // m
        if g <= 0:
            continue
        if any((d_out % g) != 0 for d_out in d_out_list):
            continue

        pg = int(r) * int(sum_d_in) + int(m) * int(sum_d_out)
        delta = abs(pg - pv)
        if best_delta is None or delta < best_delta:
            best_r = r
            best_delta = delta

    if best_r is None or best_delta is None:
        raise RuntimeError(f"No valid r found for m={m} up to max_r={max_r}")

    return int(best_r), int(best_delta)
