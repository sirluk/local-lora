from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# NOTE: This parser intentionally does not import wandb. We parse offline `.wandb`
# protobuf-ish records using a lightweight key/value_json extractor (see
# report_stage1_wandb.py for the original version).

CFG_KEYS: Tuple[str, ...] = (
    "model_name",
    "task",
    "adapter_type",
    "r",
    "m",
    "n",
    "bd_row_factor",
    "scaling_mode",
    "grouping_mode",
    "perm_seed",
    "torch_compile",
    "torch_compile_backend",
    "torch_compile_mode",
    "alpha",
    "dropout",
    "max_length",
    "learning_rate",
    "num_train_epochs",
    "warmup_ratio",
    "weight_decay",
    "seed",
    "max_train_samples",
    "max_eval_samples",
    "bf16",
    "fp16",
    "target_suffixes",
    "trainable_param_summary",
    "injection_report",
)

METRIC_KEYS: Tuple[str, ...] = (
    # Non-MNLI tasks:
    "eval/validation_score",
    # MNLI split scores:
    "eval/validation_matched_score",
    "eval/validation_mismatched_score",
    # Runtime:
    "train/train_time_s",
)

GROUP_LOCAL_PARAM_RE = re.compile(rb"group_local_param_r(\d+)_m(\d+)_d(\d+)")
GROUP_LOCAL_EQUAL_RE = re.compile(rb"group_local_equal_r(\d+)_m(\d+)")


def _read_varint(data: bytes, i: int) -> Tuple[int, int]:
    shift = 0
    out = 0
    while True:
        if i >= len(data):
            raise ValueError("varint out of range")
        b = data[i]
        i += 1
        out |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return out, i
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _extract_value_json(data: bytes, key: str, *, prefer_last: bool = False) -> Optional[str]:
    """
    Extract the W&B internal `value_json` (field 16, tag 0x82 0x01) for a given key.

    Works for both:
    - config items: 0x0a <len(key)> <key> 0x82 0x01 <len(value_json)> <value_json...>
    - history/summary items: 0x12 <len(key)> <key> 0x82 0x01 <len(value_json)> <value_json...>
    """

    kb = key.encode("utf-8")
    if len(kb) >= 128:
        raise ValueError(f"Key too long for this parser: {key!r}")

    patterns = (
        b"\x0a" + bytes([len(kb)]) + kb + b"\x82\x01",
        b"\x12" + bytes([len(kb)]) + kb + b"\x82\x01",
    )

    idxs: List[int] = []
    for pat in patterns:
        if prefer_last:
            j = data.rfind(pat)
            if j != -1:
                idxs.append(j)
        else:
            j = data.find(pat)
            if j != -1:
                idxs.append(j)

    if not idxs:
        return None

    idx = max(idxs) if prefer_last else min(idxs)
    # value length starts immediately after the pattern
    i = idx + len(patterns[0])  # both patterns are same length for a given key
    try:
        vlen, vstart = _read_varint(data, i)
    except Exception:
        return None

    vend = vstart + int(vlen)
    if vend > len(data) or vlen < 0:
        return None
    try:
        return data[vstart:vend].decode("utf-8")
    except Exception:
        return None


def _extract_value_bytes(data: bytes, key: str, *, prefer_last: bool = False) -> Optional[bytes]:
    """
    Like _extract_value_json, but returns raw bytes instead of decoding UTF-8.

    Some large W&B config blobs can contain non-UTF8 bytes (observed for Qwen
    injection_report), so we need this for robust parsing.
    """

    kb = key.encode("utf-8")
    if len(kb) >= 128:
        raise ValueError(f"Key too long for this parser: {key!r}")

    patterns = (
        b"\x0a" + bytes([len(kb)]) + kb + b"\x82\x01",
        b"\x12" + bytes([len(kb)]) + kb + b"\x82\x01",
    )

    idxs: List[int] = []
    for pat in patterns:
        j = data.rfind(pat) if prefer_last else data.find(pat)
        if j != -1:
            idxs.append(j)

    if not idxs:
        return None

    idx = max(idxs) if prefer_last else min(idxs)
    i = idx + len(patterns[0])
    try:
        vlen, vstart = _read_varint(data, i)
    except Exception:
        return None

    vend = vstart + int(vlen)
    if vend > len(data) or vlen < 0:
        return None
    return data[vstart:vend]


def _extract_json(data: bytes, key: str, *, prefer_last: bool = False) -> Any:
    v = _extract_value_json(data, key, prefer_last=prefer_last)
    if v is None:
        return None
    try:
        return json.loads(v)
    except Exception:
        return v


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(statistics.mean(xs))


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(statistics.stdev(xs))


def _fmt_float(x: Any, *, digits: int = 4) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "nan"
    return f"{xf:.{digits}f}".rstrip("0").rstrip(".")


def _target_set(target_suffixes: Any) -> str:
    if not isinstance(target_suffixes, list):
        return "unknown"
    if target_suffixes == ["q_proj", "v_proj"]:
        return "attn_only"
    if target_suffixes == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        return "attn_mlp"
    return "custom"


def _model_short_name(model_name: str) -> str:
    # HF cache paths:
    #   .../models--ORG--NAME/snapshots/<hash>
    m = re.search(r"models--[^/]+--([^/]+)/snapshots/", model_name)
    if m is not None:
        return m.group(1)
    # HF repo ids:
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


@dataclass(frozen=True)
class RunRow:
    path: str
    # config
    model_name: str
    model_short: str
    task: str
    seed: int
    adapter_type: str
    method: str
    group_local_kind: Optional[str]
    group_local_delta: Optional[int]
    r: Optional[int]
    m: Optional[int]
    n: Optional[int]
    bd_row_factor: Optional[str]
    scaling_mode: str
    grouping_mode: str
    perm_seed: Optional[int]
    target_set: str
    learning_rate: float
    max_length: int
    warmup_ratio: float
    trainable_params: Optional[int]
    adapter_trainable_params: Optional[int]
    injection_adapter_params_total: Optional[int]
    # metrics
    eval_validation_score: Optional[float]
    eval_validation_matched_score: Optional[float]
    eval_validation_mismatched_score: Optional[float]
    task_score: Optional[float]
    train_time_s: Optional[float]


def _method_label(
    *,
    adapter_type: str,
    group_local_kind: Optional[str],
    group_local_delta: Optional[int],
    r: Optional[int],
    m: Optional[int],
    n: Optional[int],
    bd_row_factor: Optional[str],
) -> str:
    if adapter_type == "group_local":
        if group_local_kind == "equal":
            return f"group_local_equal_r{r}_m{m}"
        if group_local_kind == "param":
            d = "" if group_local_delta is None else str(int(group_local_delta))
            return f"group_local_param_r{r}_m{m}_d{d}"
        return f"group_local_r{r}_m{m}"
    if adapter_type == "vanilla_lora":
        return f"vanilla_lora_r{r}"
    if adapter_type == "bd_lora":
        return f"bd_lora_r{r}_n{n}_row{bd_row_factor}"
    return adapter_type


def parse_run_file(path: Path) -> RunRow:
    data = path.read_bytes()

    cfg: Dict[str, Any] = {}
    for k in CFG_KEYS:
        cfg[k] = _extract_json(data, k, prefer_last=False)

    metrics: Dict[str, Any] = {}
    for k in METRIC_KEYS:
        metrics[k] = _extract_json(data, k, prefer_last=True)

    adapter_type = str(cfg.get("adapter_type") or "").strip().lower()
    task = str(cfg.get("task") or "").strip().lower()

    gl_kind = None
    gl_delta = None
    if adapter_type == "group_local":
        m_param = GROUP_LOCAL_PARAM_RE.search(data)
        m_equal = GROUP_LOCAL_EQUAL_RE.search(data)
        if m_param is not None:
            gl_kind = "param"
            try:
                gl_delta = int(m_param.group(3))
            except Exception:
                gl_delta = None
        elif m_equal is not None:
            gl_kind = "equal"
        else:
            gl_kind = "unknown"

    tgt_suffixes = cfg.get("target_suffixes")
    tgt_set = _target_set(tgt_suffixes)

    trainable_params = None
    adapter_trainable_params = None
    trainable_summary = cfg.get("trainable_param_summary")
    if isinstance(trainable_summary, dict):
        trainable_params = _as_int(trainable_summary.get("trainable_params"))
        adapter_trainable_params = _as_int(trainable_summary.get("adapter_trainable_params"))

    inj = cfg.get("injection_report")
    inj_adapter_params_total = None
    if isinstance(inj, dict):
        try:
            injected = inj.get("injected", [])
            if isinstance(injected, list):
                inj_adapter_params_total = sum(int(x.get("adapter_params", 0)) for x in injected if isinstance(x, dict))
        except Exception:
            inj_adapter_params_total = None
    # Fallback: some runs contain non-UTF8 bytes in the injection_report blob, so
    # our JSON decode path can fail. Still recover adapter params total by regex.
    if inj_adapter_params_total is None:
        inj_bytes = _extract_value_bytes(data, "injection_report", prefer_last=False)
        if inj_bytes is not None:
            try:
                nums = [int(m.group(1)) for m in re.finditer(rb'"adapter_params"\s*:\s*(\d+)', inj_bytes)]
                if nums:
                    inj_adapter_params_total = sum(nums)
            except Exception:
                inj_adapter_params_total = None

    eval_validation_score = _as_float(metrics.get("eval/validation_score"))
    eval_validation_matched_score = _as_float(metrics.get("eval/validation_matched_score"))
    eval_validation_mismatched_score = _as_float(metrics.get("eval/validation_mismatched_score"))

    task_score = eval_validation_score
    if task == "mnli":
        if eval_validation_matched_score is not None and eval_validation_mismatched_score is not None:
            task_score = 0.5 * (float(eval_validation_matched_score) + float(eval_validation_mismatched_score))
        else:
            task_score = None

    r_v = _as_int(cfg.get("r"))
    m_v = _as_int(cfg.get("m"))
    n_v = _as_int(cfg.get("n"))
    bd_row_factor = str(cfg.get("bd_row_factor")) if cfg.get("bd_row_factor") is not None else None

    method = _method_label(
        adapter_type=adapter_type,
        group_local_kind=gl_kind,
        group_local_delta=gl_delta,
        r=r_v,
        m=m_v,
        n=n_v,
        bd_row_factor=bd_row_factor,
    )

    model_name = str(cfg.get("model_name") or "")
    return RunRow(
        path=str(path),
        model_name=model_name,
        model_short=_model_short_name(model_name),
        task=task,
        seed=int(cfg.get("seed") or 0),
        adapter_type=adapter_type,
        method=method,
        group_local_kind=gl_kind,
        group_local_delta=_as_int(gl_delta),
        r=r_v,
        m=m_v,
        n=n_v,
        bd_row_factor=bd_row_factor,
        scaling_mode=str(cfg.get("scaling_mode") or ""),
        grouping_mode=str(cfg.get("grouping_mode") or ""),
        perm_seed=_as_int(cfg.get("perm_seed")),
        target_set=tgt_set,
        learning_rate=float(cfg.get("learning_rate") or 0.0),
        max_length=int(cfg.get("max_length") or 0),
        warmup_ratio=float(cfg.get("warmup_ratio") or 0.0),
        trainable_params=trainable_params,
        adapter_trainable_params=adapter_trainable_params,
        injection_adapter_params_total=_as_int(inj_adapter_params_total),
        eval_validation_score=eval_validation_score,
        eval_validation_matched_score=eval_validation_matched_score,
        eval_validation_mismatched_score=eval_validation_mismatched_score,
        task_score=_as_float(task_score),
        train_time_s=_as_float(metrics.get("train/train_time_s")),
    )


def _write_csv(path: Path, rows: Sequence[RunRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "path",
                "model_short",
                "model_name",
                "task",
                "seed",
                "adapter_type",
                "method",
                "group_local_kind",
                "group_local_delta",
                "r",
                "m",
                "n",
                "bd_row_factor",
                "target_set",
                "scaling_mode",
                "grouping_mode",
                "perm_seed",
                "learning_rate",
                "max_length",
                "warmup_ratio",
                "trainable_params",
                "adapter_trainable_params",
                "injection_adapter_params_total",
                "eval_validation_score",
                "eval_validation_matched_score",
                "eval_validation_mismatched_score",
                "task_score",
                "train_time_s",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.path,
                    r.model_short,
                    r.model_name,
                    r.task,
                    int(r.seed),
                    r.adapter_type,
                    r.method,
                    r.group_local_kind or "",
                    "" if r.group_local_delta is None else int(r.group_local_delta),
                    "" if r.r is None else int(r.r),
                    "" if r.m is None else int(r.m),
                    "" if r.n is None else int(r.n),
                    "" if r.bd_row_factor is None else r.bd_row_factor,
                    r.target_set,
                    r.scaling_mode,
                    r.grouping_mode,
                    "" if r.perm_seed is None else int(r.perm_seed),
                    float(r.learning_rate),
                    int(r.max_length),
                    float(r.warmup_ratio),
                    "" if r.trainable_params is None else int(r.trainable_params),
                    "" if r.adapter_trainable_params is None else int(r.adapter_trainable_params),
                    "" if r.injection_adapter_params_total is None else int(r.injection_adapter_params_total),
                    "" if r.eval_validation_score is None else float(r.eval_validation_score),
                    "" if r.eval_validation_matched_score is None else float(r.eval_validation_matched_score),
                    "" if r.eval_validation_mismatched_score is None else float(r.eval_validation_mismatched_score),
                    "" if r.task_score is None else float(r.task_score),
                    "" if r.train_time_s is None else float(r.train_time_s),
                ]
            )


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    h = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [h, sep]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _method_sort_key(method: str) -> Tuple[int, str]:
    if method.startswith("head_only"):
        return (0, method)
    if method.startswith("vanilla_lora"):
        return (1, method)
    if method.startswith("group_local_equal"):
        return (2, method)
    if method.startswith("group_local_param"):
        return (3, method)
    if method.startswith("bd_lora"):
        return (4, method)
    if method.startswith("full_ft"):
        return (5, method)
    return (6, method)


@dataclass(frozen=True)
class SeedSummary:
    model_short: str
    method: str
    seed: int
    tasks_present_excl_wnli: int
    avg_excl_wnli: float
    wnli: Optional[float]
    mean_train_time_s: Optional[float]


@dataclass(frozen=True)
class MethodSummary:
    model_short: str
    method: str
    seeds: List[int]
    complete_seeds: int
    total_seeds: int
    avg_mean: float
    avg_sd: float
    wnli_mean: Optional[float]
    mean_train_time_s: Optional[float]
    adapter_trainable_params: Optional[int]
    injection_adapter_params_total: Optional[int]
    adapter_to_injected_ratio: Optional[float]


def summarize(
    *,
    runs: Sequence[RunRow],
    tasks: Sequence[str],
) -> Tuple[List[SeedSummary], List[MethodSummary], Dict[Tuple[str, str], Dict[str, List[float]]]]:
    tasks_l = [t.lower() for t in tasks]
    tasks_excl_wnli = [t for t in tasks_l if t != "wnli"]
    required = len(tasks_excl_wnli)

    # (model, method, seed) -> task -> list[score]
    bucket: Dict[Tuple[str, str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    time_bucket: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)

    # method-level param counts (constant within method; collect all for robust median)
    method_params: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)

    for r in runs:
        key = (r.model_short, r.method, int(r.seed))
        if r.task_score is not None and not math.isnan(float(r.task_score)):
            bucket[key][r.task].append(float(r.task_score))
        if r.train_time_s is not None and not math.isnan(float(r.train_time_s)):
            time_bucket[key].append(float(r.train_time_s))

        if r.adapter_trainable_params is not None and r.injection_adapter_params_total is not None:
            method_params[(r.model_short, r.method)].append(
                (int(r.adapter_trainable_params), int(r.injection_adapter_params_total))
            )

    seed_summaries: List[SeedSummary] = []
    for (model_short, method, seed), by_task in bucket.items():
        # Aggregate duplicates by mean (should usually be exactly 1 entry)
        task_to_score = {t: _mean(vals) for t, vals in by_task.items() if vals}
        present_scores = [task_to_score.get(t) for t in tasks_excl_wnli]
        present = [s for s in present_scores if s is not None and not math.isnan(float(s))]
        avg = _mean(present)
        wnli = task_to_score.get("wnli")
        mean_tt = _mean(time_bucket.get((model_short, method, seed), []))
        seed_summaries.append(
            SeedSummary(
                model_short=model_short,
                method=method,
                seed=int(seed),
                tasks_present_excl_wnli=len(present),
                avg_excl_wnli=float(avg),
                wnli=None if wnli is None else float(wnli),
                mean_train_time_s=None if math.isnan(mean_tt) else float(mean_tt),
            )
        )

    # Group across seeds -> method summaries
    by_method: Dict[Tuple[str, str], List[SeedSummary]] = defaultdict(list)
    for s in seed_summaries:
        by_method[(s.model_short, s.method)].append(s)

    method_summaries: List[MethodSummary] = []
    for (model_short, method), ss in by_method.items():
        ss_sorted = sorted(ss, key=lambda x: int(x.seed))
        # Prefer "complete" seeds (all tasks excl wnli). If none are complete, fall back to whatever exists.
        complete = [x for x in ss_sorted if x.tasks_present_excl_wnli >= required]
        used = complete if complete else ss_sorted

        avgs = [x.avg_excl_wnli for x in used if not math.isnan(float(x.avg_excl_wnli))]
        avg_mean = _mean(avgs)
        avg_sd = _stdev(avgs)

        wnli_vals = [x.wnli for x in used if x.wnli is not None and not math.isnan(float(x.wnli))]
        wnli_mean = None if not wnli_vals else float(_mean([float(x) for x in wnli_vals]))

        tt_vals = [x.mean_train_time_s for x in used if x.mean_train_time_s is not None and not math.isnan(float(x.mean_train_time_s))]
        tt_mean = None if not tt_vals else float(_mean([float(x) for x in tt_vals]))

        # Params: take median across collected runs for stability.
        params = method_params.get((model_short, method), [])
        adapter_p = None
        injected_p = None
        ratio = None
        if params:
            adapter_list = sorted(a for a, _ in params)
            injected_list = sorted(b for _, b in params)
            adapter_p = int(adapter_list[len(adapter_list) // 2])
            injected_p = int(injected_list[len(injected_list) // 2])
            if injected_p > 0:
                ratio = float(adapter_p) / float(injected_p)

        method_summaries.append(
            MethodSummary(
                model_short=model_short,
                method=method,
                seeds=[int(x.seed) for x in ss_sorted],
                complete_seeds=len(complete),
                total_seeds=len(ss_sorted),
                avg_mean=float(avg_mean),
                avg_sd=float(avg_sd),
                wnli_mean=wnli_mean,
                mean_train_time_s=tt_mean,
                adapter_trainable_params=adapter_p,
                injection_adapter_params_total=injected_p,
                adapter_to_injected_ratio=ratio,
            )
        )

    # Per-task scores for report matrix:
    # (model, method) -> task -> list[score] across seeds
    per_task: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r.task_score is None or math.isnan(float(r.task_score)):
            continue
        per_task[(r.model_short, r.method)][r.task].append(float(r.task_score))

    return seed_summaries, method_summaries, per_task


def build_report(
    *,
    runs: Sequence[RunRow],
    method_summaries: Sequence[MethodSummary],
    per_task: Dict[Tuple[str, str], Dict[str, List[float]]],
    tasks: Sequence[str],
    out_path: Path,
    wandb_dir: Path,
) -> Dict[str, Any]:
    tasks_l = [t.lower() for t in tasks]
    tasks_excl_wnli = [t for t in tasks_l if t != "wnli"]
    required = len(tasks_excl_wnli)

    models = sorted({r.model_short for r in runs})
    methods_all = sorted({r.method for r in runs}, key=_method_sort_key)
    seeds_all = sorted({int(r.seed) for r in runs})

    # Detect "recipe" uniqueness (helps sanity-check Stage 2 is actually fixed).
    def uniq(attr: str) -> List[str]:
        vals = sorted({str(getattr(r, attr)) for r in runs})
        return vals

    recipe = {
        "target_set": uniq("target_set"),
        "scaling_mode": uniq("scaling_mode"),
        "grouping_mode": uniq("grouping_mode"),
        "perm_seed": uniq("perm_seed"),
        "max_length": uniq("max_length"),
        "warmup_ratio": uniq("warmup_ratio"),
        "learning_rate": sorted({float(r.learning_rate) for r in runs}),
    }

    lines: List[str] = []
    lines.append("# Stage 2 (Final GLUE) – Offline W&B Report")
    lines.append("")
    lines.append(f"Parsed {len(runs)} offline run files from `{str(wandb_dir)}/`.")
    lines.append("")
    lines.append(f"- Models: {', '.join(models)}")
    lines.append(f"- Tasks: {', '.join(tasks_l)}")
    lines.append(f"- Seeds observed: {', '.join(str(s) for s in seeds_all)}")
    lines.append(f"- Primary metric: mean over GLUE tasks excluding WNLI (MNLI = mean of matched/mismatched).")
    lines.append("")
    lines.append("## Recipe Check (Unique Values Found)")
    lines.append("```json")
    lines.append(json.dumps(recipe, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")

    # Index method summaries by model
    by_model: Dict[str, List[MethodSummary]] = defaultdict(list)
    for ms in method_summaries:
        by_model[ms.model_short].append(ms)

    # For delta vs vanilla, pick the first vanilla method per model.
    vanilla_by_model: Dict[str, Optional[MethodSummary]] = {}
    for m in models:
        v = next((x for x in by_model[m] if x.method.startswith("vanilla_lora")), None)
        vanilla_by_model[m] = v

    for model in models:
        lines.append(f"## Model: {model}")
        lines.append("")

        summaries = sorted(by_model.get(model, []), key=lambda x: _method_sort_key(x.method))
        vanilla = vanilla_by_model.get(model)
        v_mean = vanilla.avg_mean if vanilla is not None and not math.isnan(float(vanilla.avg_mean)) else None

        rows: List[List[str]] = []
        for s in summaries:
            delta = ""
            if v_mean is not None and not math.isnan(float(s.avg_mean)):
                delta = _fmt_float(float(s.avg_mean) - float(v_mean), digits=4)
                if delta and not delta.startswith("-"):
                    delta = "+" + delta

            complete = f"{s.complete_seeds}/{s.total_seeds}"
            if s.complete_seeds < s.total_seeds and s.complete_seeds < required:
                complete = f"{complete} (incomplete)"

            rows.append(
                [
                    s.method,
                    ",".join(str(x) for x in s.seeds),
                    complete,
                    f"{_fmt_float(s.avg_mean, digits=4)} ± {_fmt_float(s.avg_sd, digits=4)}",
                    delta,
                    "" if s.wnli_mean is None else _fmt_float(s.wnli_mean, digits=4),
                    "" if s.mean_train_time_s is None else _fmt_float(s.mean_train_time_s, digits=2),
                ]
            )

        lines.append(
            _md_table(
                [
                    "method",
                    "seeds",
                    f"complete_seeds (need {required})",
                    "avg_glue_excl_wnli (mean ± sd)",
                    "Δ vs vanilla",
                    "wnli_mean",
                    "mean_train_time_s",
                ],
                rows,
            )
        )
        lines.append("")

        # Sanity check param counting for adapter methods.
        param_rows: List[List[str]] = []
        for s in summaries:
            if not s.method.startswith(("vanilla_lora", "group_local", "bd_lora")):
                continue
            ratio = ""
            flag = ""
            if s.adapter_to_injected_ratio is not None:
                ratio = _fmt_float(s.adapter_to_injected_ratio, digits=3)
                try:
                    flag = "OK" if float(s.adapter_to_injected_ratio) <= 1.5 else "WARN"
                except Exception:
                    flag = ""
            param_rows.append(
                [
                    s.method,
                    "" if s.adapter_trainable_params is None else str(int(s.adapter_trainable_params)),
                    "" if s.injection_adapter_params_total is None else str(int(s.injection_adapter_params_total)),
                    ratio,
                    flag,
                ]
            )

        if param_rows:
            lines.append("### Adapter Param Sanity")
            lines.append(
                _md_table(
                    ["method", "adapter_trainable_params", "injected_adapter_params", "ratio", "flag"],
                    param_rows,
                )
            )
            lines.append("")

        # Per-task matrix
        lines.append("### Per-Task Scores (mean ± sd over available seeds)")
        matrix_rows: List[List[str]] = []
        for task in tasks_l:
            row = [task]
            for s in summaries:
                vals = per_task.get((model, s.method), {}).get(task, [])
                if not vals:
                    row.append("")
                else:
                    row.append(f"{_fmt_float(_mean(vals), digits=4)}±{_fmt_float(_stdev(vals), digits=4)}")
            matrix_rows.append(row)
        lines.append(_md_table(["task"] + [s.method for s in summaries], matrix_rows))
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    return {
        "wandb_dir": str(wandb_dir),
        "parsed_run_files": len(runs),
        "models": models,
        "tasks": tasks_l,
        "seeds": seeds_all,
        "recipe_uniques": recipe,
        "paths": {
            "report_md": str(out_path),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 report from offline W&B .wandb files (no network).")
    p.add_argument(
        "--wandb_dir",
        type=str,
        default="wandb_stage2/wandb",
        help="Directory with offline-run-*/run-*.wandb",
    )
    p.add_argument("--out_dir", type=str, default="reports/stage2")
    p.add_argument("--tasks", type=str, default="cola,sst2,mrpc,qqp,stsb,mnli,qnli,rte,wnli")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on parsed runs (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wandb_dir = Path(args.wandb_dir)
    out_dir = Path(args.out_dir)
    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]

    paths = sorted(wandb_dir.glob("offline-run-*/run-*.wandb"))
    if args.limit is not None:
        paths = paths[: int(args.limit)]

    runs: List[RunRow] = []
    for i, p in enumerate(paths):
        runs.append(parse_run_file(p))
        if (i + 1) % 200 == 0:
            print(json.dumps({"parsed": i + 1}, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "stage2_runs.csv"
    _write_csv(runs_csv, runs)

    seed_summaries, method_summaries, per_task = summarize(runs=runs, tasks=tasks)
    report_path = out_dir / "stage2_report.md"
    summary_obj = build_report(
        runs=runs,
        method_summaries=method_summaries,
        per_task=per_task,
        tasks=tasks,
        out_path=report_path,
        wandb_dir=wandb_dir,
    )
    summary_path = report_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary_obj, indent=2, sort_keys=True) + "\n")

    # Also emit a compact stdout summary for quick CLI usage.
    print(json.dumps(summary_obj, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
