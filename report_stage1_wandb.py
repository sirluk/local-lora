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
    "eval/validation_score",
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


def _extract_json(data: bytes, key: str, *, prefer_last: bool = False) -> Any:
    v = _extract_value_json(data, key, prefer_last=prefer_last)
    if v is None:
        return None
    try:
        return json.loads(v)
    except Exception:
        return v


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
    if abs(xf) >= 1e4 or (abs(xf) > 0 and abs(xf) < 1e-3):
        return f"{xf:.{digits}g}"
    return f"{xf:.{digits}f}".rstrip("0").rstrip(".")


def _fmt_lr(x: Any) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except Exception:
        return str(x)
    return f"{xf:.0e}".replace("+", "")


def _target_set(target_suffixes: Any) -> str:
    if not isinstance(target_suffixes, list):
        return "unknown"
    if target_suffixes == ["q_proj", "v_proj"]:
        return "attn_only"
    if target_suffixes == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        return "attn_mlp"
    return "custom"


@dataclass(frozen=True)
class RunRow:
    path: str
    # config
    model_name: str
    task: str
    adapter_type: str
    r: Optional[int]
    m: Optional[int]
    n: Optional[int]
    bd_row_factor: Optional[str]
    scaling_mode: str
    grouping_mode: str
    max_length: int
    warmup_ratio: float
    learning_rate: float
    seed: int
    target_set: str
    target_suffixes_json: str
    group_local_kind: Optional[str]
    group_local_delta: Optional[int]
    trainable_params: Optional[int]
    adapter_trainable_params: Optional[int]
    injection_adapter_params_total: Optional[int]
    # metrics
    eval_score: Optional[float]
    train_time_s: Optional[float]


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


def parse_run_file(path: Path) -> RunRow:
    data = path.read_bytes()

    cfg: Dict[str, Any] = {}
    for k in CFG_KEYS:
        cfg[k] = _extract_json(data, k, prefer_last=False)

    metrics: Dict[str, Any] = {}
    for k in METRIC_KEYS:
        metrics[k] = _extract_json(data, k, prefer_last=True)

    adapter_type = str(cfg.get("adapter_type") or "").strip().lower()

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

    trainable_summary = cfg.get("trainable_param_summary")
    trainable_params = None
    adapter_trainable_params = None
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

    return RunRow(
        path=str(path),
        model_name=str(cfg.get("model_name") or ""),
        task=str(cfg.get("task") or "").lower(),
        adapter_type=adapter_type,
        r=_as_int(cfg.get("r")),
        m=_as_int(cfg.get("m")),
        n=_as_int(cfg.get("n")),
        bd_row_factor=(str(cfg.get("bd_row_factor")) if cfg.get("bd_row_factor") is not None else None),
        scaling_mode=str(cfg.get("scaling_mode") or ""),
        grouping_mode=str(cfg.get("grouping_mode") or ""),
        max_length=int(cfg.get("max_length") or 0),
        warmup_ratio=float(cfg.get("warmup_ratio") or 0.0),
        learning_rate=float(cfg.get("learning_rate") or 0.0),
        seed=int(cfg.get("seed") or 0),
        target_set=tgt_set,
        target_suffixes_json=json.dumps(tgt_suffixes, sort_keys=False),
        group_local_kind=gl_kind,
        group_local_delta=gl_delta,
        trainable_params=trainable_params,
        adapter_trainable_params=adapter_trainable_params,
        injection_adapter_params_total=_as_int(inj_adapter_params_total),
        eval_score=_as_float(metrics.get("eval/validation_score")),
        train_time_s=_as_float(metrics.get("train/train_time_s")),
    )


def _write_csv(path: Path, rows: Sequence[RunRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "path",
                "model_name",
                "task",
                "adapter_type",
                "group_local_kind",
                "group_local_delta",
                "r",
                "m",
                "n",
                "bd_row_factor",
                "target_set",
                "target_suffixes_json",
                "scaling_mode",
                "grouping_mode",
                "learning_rate",
                "max_length",
                "warmup_ratio",
                "seed",
                "trainable_params",
                "adapter_trainable_params",
                "injection_adapter_params_total",
                "eval_score",
                "train_time_s",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.path,
                    r.model_name,
                    r.task,
                    r.adapter_type,
                    r.group_local_kind or "",
                    "" if r.group_local_delta is None else int(r.group_local_delta),
                    "" if r.r is None else int(r.r),
                    "" if r.m is None else int(r.m),
                    "" if r.n is None else int(r.n),
                    "" if r.bd_row_factor is None else r.bd_row_factor,
                    r.target_set,
                    r.target_suffixes_json,
                    r.scaling_mode,
                    r.grouping_mode,
                    float(r.learning_rate),
                    int(r.max_length),
                    float(r.warmup_ratio),
                    int(r.seed),
                    "" if r.trainable_params is None else int(r.trainable_params),
                    "" if r.adapter_trainable_params is None else int(r.adapter_trainable_params),
                    "" if r.injection_adapter_params_total is None else int(r.injection_adapter_params_total),
                    "" if r.eval_score is None else float(r.eval_score),
                    "" if r.train_time_s is None else float(r.train_time_s),
                ]
            )


@dataclass(frozen=True)
class ConfigSummary:
    adapter_type: str
    group_local_kind: Optional[str]
    group_local_delta: Optional[int]
    r: Optional[int]
    m: Optional[int]
    n: Optional[int]
    bd_row_factor: Optional[str]
    target_set: str
    scaling_mode: str
    learning_rate: float
    max_length: int
    warmup_ratio: float
    seed: int
    tasks_present: int
    avg_score: float
    task_scores: Dict[str, float]
    task_repl: Dict[str, int]


def summarize_configs(runs: Sequence[RunRow], *, tasks: Sequence[str]) -> List[ConfigSummary]:
    tasks_l = [t.lower() for t in tasks]
    tasks_set = set(tasks_l)

    # key -> task -> list[score]
    buckets: Dict[
        Tuple[Any, ...],
        Dict[str, List[float]],
    ] = defaultdict(lambda: defaultdict(list))

    for r in runs:
        if r.task not in tasks_set:
            continue
        if r.eval_score is None or (isinstance(r.eval_score, float) and math.isnan(float(r.eval_score))):
            continue

        key = (
            r.adapter_type,
            r.group_local_kind,
            r.group_local_delta,
            r.r,
            r.m,
            r.n,
            r.bd_row_factor,
            r.target_set,
            r.scaling_mode,
            float(r.learning_rate),
            int(r.max_length),
            float(r.warmup_ratio),
            int(r.seed),
        )
        buckets[key][r.task].append(float(r.eval_score))

    out: List[ConfigSummary] = []
    for key, by_task in buckets.items():
        (
            adapter_type,
            group_local_kind,
            group_local_delta,
            r_v,
            m_v,
            n_v,
            bd_row_factor,
            target_set,
            scaling_mode,
            learning_rate,
            max_length,
            warmup_ratio,
            seed,
        ) = key

        task_scores: Dict[str, float] = {}
        task_repl: Dict[str, int] = {}
        for t, vals in by_task.items():
            task_scores[t] = _mean(vals)
            task_repl[t] = len(vals)

        # Avg across the requested tasks, but only those present.
        present_scores = [task_scores[t] for t in tasks_l if t in task_scores]
        avg = _mean(present_scores)

        out.append(
            ConfigSummary(
                adapter_type=str(adapter_type),
                group_local_kind=str(group_local_kind) if group_local_kind is not None else None,
                group_local_delta=_as_int(group_local_delta),
                r=r_v,
                m=m_v,
                n=n_v,
                bd_row_factor=str(bd_row_factor) if bd_row_factor is not None else None,
                target_set=str(target_set),
                scaling_mode=str(scaling_mode),
                learning_rate=float(learning_rate),
                max_length=int(max_length),
                warmup_ratio=float(warmup_ratio),
                seed=int(seed),
                tasks_present=len(task_scores),
                avg_score=float(avg),
                task_scores=task_scores,
                task_repl=task_repl,
            )
        )

    return out


def _pick_best(
    summaries: Sequence[ConfigSummary],
    *,
    predicate,
    tasks_required: int,
    top_k: int = 5,
) -> List[ConfigSummary]:
    cand = [s for s in summaries if predicate(s) and s.tasks_present >= tasks_required]
    cand.sort(key=lambda s: (float("-inf") if math.isnan(s.avg_score) else s.avg_score), reverse=True)
    return cand[:top_k]


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    h = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [h, sep]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def build_report(
    *,
    runs: Sequence[RunRow],
    summaries: Sequence[ConfigSummary],
    tasks: Sequence[str],
    runs_csv: Path,
    out_path: Path,
) -> Dict[str, Any]:
    tasks_l = [t.lower() for t in tasks]
    tasks_required = len(tasks_l)

    # Stage-1: choose best vanilla_lora r=16 for each target set, then pick winner.
    vanilla_best: Dict[str, Optional[ConfigSummary]] = {}
    for tgt in ["attn_only", "attn_mlp"]:
        best = _pick_best(
            summaries,
            predicate=lambda s, tgt=tgt: s.adapter_type == "vanilla_lora" and s.r == 16 and s.target_set == tgt,
            tasks_required=tasks_required,
            top_k=1,
        )
        vanilla_best[tgt] = best[0] if best else None

    chosen_target_set = None
    chosen_recipe = None
    if vanilla_best["attn_only"] is not None or vanilla_best["attn_mlp"] is not None:
        # Prefer the higher avg; tie-break to attn_only (simpler) if within 1e-6.
        a = vanilla_best["attn_only"].avg_score if vanilla_best["attn_only"] is not None else float("-inf")
        b = vanilla_best["attn_mlp"].avg_score if vanilla_best["attn_mlp"] is not None else float("-inf")
        if b > a + 1e-6:
            chosen_target_set = "attn_mlp"
            chosen_recipe = vanilla_best["attn_mlp"]
        else:
            chosen_target_set = "attn_only"
            chosen_recipe = vanilla_best["attn_only"]

    if chosen_recipe is None:
        chosen_target_set = "unknown"
        chosen_recipe = None

    # Given chosen recipe hyperparams (from vanilla), pick best full_ft lr (under same len/wu),
    # plus best group_local_equal/param m and best bd_lora N, all under the same recipe.
    def same_recipe(s: ConfigSummary) -> bool:
        if chosen_recipe is None:
            return False
        return (
            s.target_set == chosen_recipe.target_set
            and s.scaling_mode == chosen_recipe.scaling_mode
            and s.max_length == chosen_recipe.max_length
            and abs(s.warmup_ratio - chosen_recipe.warmup_ratio) < 1e-12
            and abs(s.learning_rate - chosen_recipe.learning_rate) < 1e-12
            and s.seed == chosen_recipe.seed
        )

    # full_ft doesn't meaningfully depend on scaling_mode / target_set; but Stage 2 will pass one scaling_mode anyway.
    # Here we keep max_length and warmup_ratio fixed to match the recipe, and sweep lr.
    full_ft_best = _pick_best(
        summaries,
        predicate=lambda s: s.adapter_type == "full_ft"
        and (chosen_recipe is None or (s.max_length == chosen_recipe.max_length and abs(s.warmup_ratio - chosen_recipe.warmup_ratio) < 1e-12))
        and s.seed == (0 if chosen_recipe is None else chosen_recipe.seed),
        tasks_required=tasks_required,
        top_k=1,
    )
    full_ft_best = full_ft_best[0] if full_ft_best else None

    # head_only: pick best lr under same len/wu
    head_only_best = _pick_best(
        summaries,
        predicate=lambda s: s.adapter_type == "head_only"
        and (chosen_recipe is None or (s.max_length == chosen_recipe.max_length and abs(s.warmup_ratio - chosen_recipe.warmup_ratio) < 1e-12))
        and s.seed == (0 if chosen_recipe is None else chosen_recipe.seed),
        tasks_required=tasks_required,
        top_k=1,
    )
    head_only_best = head_only_best[0] if head_only_best else None

    gl_equal_best = _pick_best(
        summaries,
        predicate=lambda s: s.adapter_type == "group_local" and s.group_local_kind == "equal" and s.r == 16 and same_recipe(s),
        tasks_required=tasks_required,
        top_k=1,
    )
    gl_equal_best = gl_equal_best[0] if gl_equal_best else None

    gl_param_best = _pick_best(
        summaries,
        predicate=lambda s: s.adapter_type == "group_local" and s.group_local_kind == "param" and same_recipe(s),
        tasks_required=tasks_required,
        top_k=1,
    )
    gl_param_best = gl_param_best[0] if gl_param_best else None

    bd_best = _pick_best(
        summaries,
        predicate=lambda s: s.adapter_type == "bd_lora" and s.r == 16 and same_recipe(s),
        tasks_required=tasks_required,
        top_k=2,  # show n=4 vs n=8 if both exist
    )

    # Vanilla top configs per target set
    vanilla_top_attn_only = _pick_best(
        summaries,
        predicate=lambda s: s.adapter_type == "vanilla_lora" and s.r == 16 and s.target_set == "attn_only",
        tasks_required=tasks_required,
        top_k=5,
    )
    vanilla_top_attn_mlp = _pick_best(
        summaries,
        predicate=lambda s: s.adapter_type == "vanilla_lora" and s.r == 16 and s.target_set == "attn_mlp",
        tasks_required=tasks_required,
        top_k=5,
    )

    # Coverage stats
    by_target: Dict[str, int] = defaultdict(int)
    by_method: Dict[str, int] = defaultdict(int)
    by_task: Dict[str, int] = defaultdict(int)
    for r in runs:
        by_target[r.target_set] += 1
        k = r.adapter_type
        if r.adapter_type == "group_local" and r.group_local_kind:
            k = f"group_local_{r.group_local_kind}"
        if r.adapter_type == "bd_lora" and r.n is not None:
            k = f"bd_lora_n{r.n}"
        by_method[k] += 1
        by_task[r.task] += 1

    # Build markdown report
    lines: List[str] = []
    lines.append("# Stage 1 Report (W&B offline logs)")
    lines.append("")
    lines.append("This report summarizes Stage 1 (protocol lock sweep) using the offline W&B run files under `wandb_stage1/`.")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Tasks: {', '.join(tasks_l)}")
    lines.append("- Metric: `eval/validation_score` (per-task primary GLUE score as logged by the Trainer)")
    lines.append("")
    lines.append("## Data Coverage")
    lines.append(f"- Parsed runs: {len(runs)}")
    lines.append(f"- Target sets: {', '.join(f'{k}={v}' for k, v in sorted(by_target.items()))}")
    lines.append(f"- Methods: {', '.join(f'{k}={v}' for k, v in sorted(by_method.items()))}")
    lines.append(f"- Task counts: {', '.join(f'{k}={v}' for k, v in sorted(by_task.items()))}")
    lines.append("")

    # Sanity-check adapter trainable param counts vs injection_report adapter params.
    lines.append("## Sanity Check (Trainable Params)")
    lines.append(
        "For adapter methods, `injection_report.adapter_params_total` is the number of low-rank adapter parameters "
        "that were injected, while `trainable_param_summary.adapter_trainable_params` is what actually ended up "
        "trainable in the run. If the latter is much larger, it usually means base projection weights were "
        "accidentally unfrozen."
    )
    lines.append("")

    # Pick one representative run per method under the chosen recipe (if available).
    rep_rows: List[List[str]] = []
    if chosen_recipe is not None:
        def match_recipe(r: RunRow) -> bool:
            return (
                r.target_set == chosen_recipe.target_set
                and r.scaling_mode == chosen_recipe.scaling_mode
                and r.max_length == chosen_recipe.max_length
                and abs(r.warmup_ratio - chosen_recipe.warmup_ratio) < 1e-12
                and abs(r.learning_rate - chosen_recipe.learning_rate) < 1e-12
                and r.seed == chosen_recipe.seed
            )

        want = [
            ("vanilla_lora", None),
            ("group_local", "equal"),
            ("group_local", "param"),
            ("bd_lora", None),
        ]
        for adapter_type, gl_kind in want:
            rr = next(
                (
                    r
                    for r in runs
                    if r.adapter_type == adapter_type
                    and (gl_kind is None or r.group_local_kind == gl_kind)
                    and r.task == "sst2"
                    and match_recipe(r)
                ),
                None,
            )
            label = adapter_type if gl_kind is None else f"{adapter_type}_{gl_kind}"
            if rr is None:
                rep_rows.append([label, "", "", "", ""])
                continue

            inj = rr.injection_adapter_params_total
            trainable = rr.adapter_trainable_params
            ratio = ""
            if isinstance(inj, int) and inj > 0 and isinstance(trainable, int):
                ratio = _fmt_float(float(trainable) / float(inj), digits=3)

            rep_rows.append(
                [
                    label,
                    "" if trainable is None else str(int(trainable)),
                    "" if inj is None else str(int(inj)),
                    ratio,
                    "OK" if ratio == "" else ("OK" if float(ratio) <= 1.5 else "WARN"),
                ]
            )

    if rep_rows:
        lines.append(_md_table(["method", "adapter_trainable_params", "injected_adapter_params", "ratio", "flag"], rep_rows))
        lines.append("")
        # If vanilla looks suspicious, emit an explicit warning.
        for row in rep_rows:
            if row and row[0] == "vanilla_lora" and row[-1] == "WARN":
                lines.append(
                    "Warning: These Stage-1 runs appear to have trained far more parameters than the injected LoRA "
                    "adapters (ratio >> 1). This indicates the wrapped base Linear weights were likely unfrozen "
                    "during training, so the sweep is closer to partial fine-tuning than vanilla LoRA."
                )
                lines.append(
                    "In this repo, the likely cause was `unfreeze_adapter_params()` recursing into `module.base`; "
                    "this has been fixed to only unfreeze adapter parameters (see `local_lora/inject.py`)."
                )
                lines.append(
                    "Recommendation: re-run Stage 1 (and Stage 2) after the fix if you need true LoRA vs Group-Local LoRA comparisons."
                )
                lines.append("")
                break
    else:
        lines.append("Skipped: could not select representative runs (recipe not determined).")
        lines.append("")
    lines.append("## Recommendation (Stage 2 Recipe)")
    if chosen_recipe is None:
        lines.append("Could not determine a vanilla LoRA best config (missing complete 4-task coverage).")
    else:
        lines.append(
            "Chosen based on best Avg (mean over the Stage-1 task subset) for `vanilla_lora r=16`:"
        )
        lines.append("")
        lines.append(
            _md_table(
                ["target_set", "scaling", "lr", "max_length", "warmup", "avg", "cola", "mrpc", "rte", "sst2"],
                [
                    [
                        chosen_recipe.target_set,
                        chosen_recipe.scaling_mode,
                        _fmt_lr(chosen_recipe.learning_rate),
                        str(chosen_recipe.max_length),
                        _fmt_float(chosen_recipe.warmup_ratio, digits=3),
                        _fmt_float(chosen_recipe.avg_score),
                        _fmt_float(chosen_recipe.task_scores.get("cola")),
                        _fmt_float(chosen_recipe.task_scores.get("mrpc")),
                        _fmt_float(chosen_recipe.task_scores.get("rte")),
                        _fmt_float(chosen_recipe.task_scores.get("sst2")),
                    ]
                ],
            )
        )
        if "attn_mlp" not in by_target:
            lines.append("")
            lines.append(
                "Note: no `attn_mlp` runs were found in this export, so the target-set choice cannot be compared here."
            )

    lines.append("")
    lines.append("## Vanilla LoRA (r=16) – Top Configs")
    if vanilla_top_attn_only:
        rows = []
        for s in vanilla_top_attn_only:
            rows.append(
                [
                    s.target_set,
                    s.scaling_mode,
                    _fmt_lr(s.learning_rate),
                    str(s.max_length),
                    _fmt_float(s.warmup_ratio, digits=3),
                    _fmt_float(s.avg_score),
                    _fmt_float(s.task_scores.get("cola")),
                    _fmt_float(s.task_scores.get("mrpc")),
                    _fmt_float(s.task_scores.get("rte")),
                    _fmt_float(s.task_scores.get("sst2")),
                ]
            )
        lines.append("### attn_only")
        lines.append(
            _md_table(
                ["target_set", "scaling", "lr", "max_length", "warmup", "avg", "cola", "mrpc", "rte", "sst2"],
                rows,
            )
        )
        lines.append("")
    if vanilla_top_attn_mlp:
        rows = []
        for s in vanilla_top_attn_mlp:
            rows.append(
                [
                    s.target_set,
                    s.scaling_mode,
                    _fmt_lr(s.learning_rate),
                    str(s.max_length),
                    _fmt_float(s.warmup_ratio, digits=3),
                    _fmt_float(s.avg_score),
                    _fmt_float(s.task_scores.get("cola")),
                    _fmt_float(s.task_scores.get("mrpc")),
                    _fmt_float(s.task_scores.get("rte")),
                    _fmt_float(s.task_scores.get("sst2")),
                ]
            )
        lines.append("### attn_mlp")
        lines.append(
            _md_table(
                ["target_set", "scaling", "lr", "max_length", "warmup", "avg", "cola", "mrpc", "rte", "sst2"],
                rows,
            )
        )
        lines.append("")

    lines.append("## Best Methods Under The Chosen Recipe")
    if chosen_recipe is None:
        lines.append("Skipped: recipe not determined.")
    else:
        rows: List[List[str]] = []

        def add_row(label: str, s: Optional[ConfigSummary]) -> None:
            if s is None:
                rows.append([label, "", "", "", "", "", "", "", "", ""])
                return
            rows.append(
                [
                    label,
                    s.scaling_mode,
                    _fmt_lr(s.learning_rate),
                    str(s.max_length),
                    _fmt_float(s.warmup_ratio, digits=3),
                    _fmt_float(s.avg_score),
                    _fmt_float(s.task_scores.get("cola")),
                    _fmt_float(s.task_scores.get("mrpc")),
                    _fmt_float(s.task_scores.get("rte")),
                    _fmt_float(s.task_scores.get("sst2")),
                ]
            )

        add_row("head_only (best lr)", head_only_best)
        add_row("full_ft (best lr)", full_ft_best)
        add_row("vanilla_lora r=16", chosen_recipe)
        if gl_equal_best is not None:
            add_row(f"group_local_equal r=16 m={gl_equal_best.m}", gl_equal_best)
        else:
            add_row("group_local_equal (missing)", None)
        if gl_param_best is not None:
            delta_note = "" if gl_param_best.group_local_delta is None else f" d={gl_param_best.group_local_delta}"
            add_row(f"group_local_param r={gl_param_best.r} m={gl_param_best.m}{delta_note}", gl_param_best)
        else:
            add_row("group_local_param (missing)", None)

        if bd_best:
            for s in bd_best:
                add_row(f"bd_lora r=16 n={s.n}", s)
        else:
            add_row("bd_lora (missing)", None)

        lines.append(
            _md_table(
                ["method", "scaling", "lr", "max_length", "warmup", "avg", "cola", "mrpc", "rte", "sst2"],
                rows,
            )
        )

        # Group-local sweeps under the chosen recipe
        lines.append("")
        lines.append("## Group-Local Sweep Under The Chosen Recipe")

        gl_equal_all = _pick_best(
            summaries,
            predicate=lambda s: s.adapter_type == "group_local"
            and s.group_local_kind == "equal"
            and s.r == 16
            and same_recipe(s),
            tasks_required=tasks_required,
            top_k=1000,
        )
        gl_equal_all.sort(key=lambda s: (9999 if s.m is None else int(s.m)))

        if gl_equal_all:
            rows2: List[List[str]] = []
            for s in gl_equal_all:
                rows2.append(
                    [
                        str(s.m) if s.m is not None else "",
                        _fmt_float(s.avg_score),
                        _fmt_float(s.task_scores.get("cola")),
                        _fmt_float(s.task_scores.get("mrpc")),
                        _fmt_float(s.task_scores.get("rte")),
                        _fmt_float(s.task_scores.get("sst2")),
                    ]
                )
            lines.append("### group_local_equal (r=16)")
            lines.append(_md_table(["m", "avg", "cola", "mrpc", "rte", "sst2"], rows2))
            lines.append("")
        else:
            lines.append("No `group_local_equal` configs matched the chosen recipe with full task coverage.")
            lines.append("")

        gl_param_all = _pick_best(
            summaries,
            predicate=lambda s: s.adapter_type == "group_local" and s.group_local_kind == "param" and same_recipe(s),
            tasks_required=tasks_required,
            top_k=1000,
        )
        gl_param_all.sort(key=lambda s: (9999 if s.m is None else int(s.m)))

        if gl_param_all:
            rows3: List[List[str]] = []
            for s in gl_param_all:
                rows3.append(
                    [
                        str(s.m) if s.m is not None else "",
                        str(s.r) if s.r is not None else "",
                        str(s.group_local_delta) if s.group_local_delta is not None else "",
                        _fmt_float(s.avg_score),
                        _fmt_float(s.task_scores.get("cola")),
                        _fmt_float(s.task_scores.get("mrpc")),
                        _fmt_float(s.task_scores.get("rte")),
                        _fmt_float(s.task_scores.get("sst2")),
                    ]
                )
            lines.append("### group_local_param")
            lines.append(_md_table(["m", "r", "delta", "avg", "cola", "mrpc", "rte", "sst2"], rows3))
            lines.append("")
        else:
            lines.append("No `group_local_param` configs matched the chosen recipe with full task coverage.")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    return {
        "chosen_target_set": chosen_target_set,
        "chosen_recipe": None
        if chosen_recipe is None
        else {
            "target_set": chosen_recipe.target_set,
            "scaling_mode": chosen_recipe.scaling_mode,
            "learning_rate": chosen_recipe.learning_rate,
            "max_length": chosen_recipe.max_length,
            "warmup_ratio": chosen_recipe.warmup_ratio,
            "avg_score": chosen_recipe.avg_score,
            "task_scores": chosen_recipe.task_scores,
        },
        "best_full_ft": None
        if full_ft_best is None
        else {
            "learning_rate": full_ft_best.learning_rate,
            "avg_score": full_ft_best.avg_score,
            "task_scores": full_ft_best.task_scores,
        },
        "best_group_local_equal": None
        if gl_equal_best is None
        else {
            "r": gl_equal_best.r,
            "m": gl_equal_best.m,
            "avg_score": gl_equal_best.avg_score,
            "task_scores": gl_equal_best.task_scores,
        },
        "best_group_local_param": None
        if gl_param_best is None
        else {
            "r": gl_param_best.r,
            "m": gl_param_best.m,
            "avg_score": gl_param_best.avg_score,
            "task_scores": gl_param_best.task_scores,
        },
        "best_bd_lora": [
            {
                "n": s.n,
                "avg_score": s.avg_score,
                "task_scores": s.task_scores,
            }
            for s in bd_best
        ],
        "paths": {
            "runs_csv": str(runs_csv),
            "report_md": str(out_path),
            "summary_json": str(out_path.with_suffix(".summary.json")),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 report from offline W&B .wandb files (no network).")
    p.add_argument("--wandb_dir", type=str, default="wandb_stage1", help="Directory with offline-run-*/run-*.wandb")
    p.add_argument("--out_dir", type=str, default="reports/stage1")
    p.add_argument("--tasks", type=str, default="sst2,mrpc,rte,cola")
    p.add_argument("--seed", type=int, default=0)
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

    rows: List[RunRow] = []
    for i, p in enumerate(paths):
        r = parse_run_file(p)
        if r.seed != int(args.seed):
            continue
        if r.task not in set(tasks):
            continue
        rows.append(r)
        if (i + 1) % 200 == 0:
            print(json.dumps({"parsed": i + 1, "kept": len(rows)}, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "stage1_runs.csv"
    _write_csv(runs_csv, rows)

    summaries = summarize_configs(rows, tasks=tasks)
    report_path = out_dir / "stage1_report.md"
    summary_obj = build_report(runs=rows, summaries=summaries, tasks=tasks, runs_csv=runs_csv, out_path=report_path)
    (report_path.with_suffix(".summary.json")).write_text(json.dumps(summary_obj, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary_obj, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
