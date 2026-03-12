from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_TASKS = "cola,sst2,mrpc,rte,qnli,qqp,mnli,stsb,wnli"


def _setdefault_env(name: str, value: str) -> None:
    if os.environ.get(name) is None:
        os.environ[name] = value


def _apply_hf_home(hf_home: str | None) -> None:
    if not hf_home:
        return
    os.environ["HF_HOME"] = hf_home
    _setdefault_env("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))
    _setdefault_env("HF_DATASETS_CACHE", str(Path(hf_home) / "datasets"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prefetch GLUE datasets + metrics into the Hugging Face cache.\n\n"
            "Run this once on a node WITH internet (login / datamover), then set HF_*_OFFLINE=1 in your training job."
        )
    )
    p.add_argument("--tasks", type=str, default=DEFAULT_TASKS, help="Comma-separated GLUE tasks to prefetch.")
    p.add_argument(
        "--hf_home",
        type=str,
        default=os.environ.get("HF_HOME"),
        help="Where to store HF caches (must be on a filesystem visible to compute nodes).",
    )
    p.add_argument(
        "--offline_check",
        action="store_true",
        help="After prefetching, re-load everything with HF_HUB_OFFLINE=1 to verify the cache is sufficient.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _apply_hf_home(args.hf_home)
    _setdefault_env("HF_HUB_DISABLE_TELEMETRY", "1")

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        print(f"HF_HOME={hf_home}")
    print(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE', '')}")
    print(f"TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE', '')}")

    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise SystemExit("No tasks specified.")

    from datasets import load_dataset
    from evaluate import load as load_metric

    for task in tasks:
        print(f"\n== Prefetch: {task} ==")
        load_dataset("glue", task)
        load_metric("glue", task)

    if args.offline_check:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("\n== Offline check (HF_HUB_OFFLINE=1) ==")
        for task in tasks:
            load_dataset("glue", task)
            load_metric("glue", task)

    print("\nDone.")


if __name__ == "__main__":
    main()

