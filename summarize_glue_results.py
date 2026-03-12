from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CONFIG_FIELDS: Tuple[str, ...] = (
    "model",
    "adapter_type",
    "r",
    "m",
    "n",
    "bd_row_factor",
    "scaling_mode",
    "grouping_mode",
    "perm_seed",
    "target_suffixes_json",
    "alpha",
    "dropout",
    "learning_rate",
    "num_train_epochs",
    "max_length",
    "warmup_ratio",
    "weight_decay",
    "max_train_samples",
    "max_eval_samples",
)


def _none_if_empty(v: str) -> Optional[str]:
    v = (v or "").strip()
    return v if v else None


def _int_or_none(v: str) -> Optional[int]:
    v2 = _none_if_empty(v)
    return int(v2) if v2 is not None else None


def _float_or_none(v: str) -> Optional[float]:
    v2 = _none_if_empty(v)
    return float(v2) if v2 is not None else None


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize GLUE results.csv into per-task mean/std and GLUE avg.")
    p.add_argument("--results_csv", type=str, required=True)
    p.add_argument("--out_csv", type=str, default=None, help="Optional output CSV path (defaults next to results_csv).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.results_csv)
    out_path = Path(args.out_csv) if args.out_csv is not None else in_path.with_name(in_path.stem + "_summary.csv")

    rows: List[Dict[str, Any]] = []
    with in_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # config_key -> task -> seed -> split -> score
    scores: Dict[Tuple[Any, ...], Dict[str, Dict[int, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    for r in rows:
        cfg: List[Any] = []
        for k in CONFIG_FIELDS:
            if k in {"r", "m", "n", "perm_seed", "max_length"}:
                cfg.append(_int_or_none(r.get(k, "")))
            elif k in {"alpha", "dropout", "learning_rate", "num_train_epochs", "warmup_ratio", "weight_decay"}:
                cfg.append(_float_or_none(r.get(k, "")))
            else:
                cfg.append(_none_if_empty(r.get(k, "")))
        cfg_key = tuple(cfg)

        task = str(r["task"]).lower()
        split = str(r["split"])
        seed = int(r["seed"])
        score = float(r["score"])
        if math.isnan(score):
            continue
        scores[cfg_key][task][seed][split] = score

    out_rows: List[Dict[str, Any]] = []

    for cfg_key, by_task in scores.items():
        cfg_dict = dict(zip(CONFIG_FIELDS, cfg_key))

        # task -> list[score_per_seed]
        task_scores: Dict[str, List[float]] = {}
        # seed -> list[task_scores]
        glue_scores_by_seed: Dict[int, List[float]] = defaultdict(list)

        for task, by_seed in sorted(by_task.items()):
            per_seed: List[float] = []
            for seed, by_split in sorted(by_seed.items()):
                if task == "mnli":
                    m = by_split.get("validation_matched")
                    mm = by_split.get("validation_mismatched")
                    if m is None or mm is None:
                        continue
                    s = 0.5 * (float(m) + float(mm))
                else:
                    s = by_split.get("validation")
                    if s is None:
                        continue
                    s = float(s)
                per_seed.append(float(s))
                if task != "wnli":
                    glue_scores_by_seed[int(seed)].append(float(s))

            if not per_seed:
                continue

            task_scores[task] = per_seed
            mean_v, std_v = _mean_std(per_seed)
            out_rows.append(
                {
                    **cfg_dict,
                    "task": task,
                    "n_seeds": len(per_seed),
                    "score_mean": mean_v,
                    "score_std": std_v,
                }
            )

        # GLUE avg (exclude WNLI) per seed then mean/std across seeds
        glue_per_seed: List[float] = []
        for seed, vals in sorted(glue_scores_by_seed.items()):
            if not vals:
                continue
            glue_per_seed.append(float(statistics.mean(vals)))

        if glue_per_seed:
            mean_v, std_v = _mean_std(glue_per_seed)
            out_rows.append(
                {
                    **cfg_dict,
                    "task": "glue_avg_excl_wnli",
                    "n_seeds": len(glue_per_seed),
                    "score_mean": mean_v,
                    "score_std": std_v,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        fieldnames = list(CONFIG_FIELDS) + ["task", "n_seeds", "score_mean", "score_std"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(json.dumps({"in_csv": str(in_path), "out_csv": str(out_path), "rows": len(out_rows)}, indent=2))


if __name__ == "__main__":
    main()

