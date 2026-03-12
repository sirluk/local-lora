# local-lora agent notes

## Purpose
This repo benchmarks **vanilla LoRA** vs **Group-Local LoRA** on **GLUE** (fine-tune + eval on validation splits).

## Entrypoints
- `train_glue.py` – run a single GLUE task.
- `run_glue_suite.py` – run multiple tasks + method grid; use `--methods` to limit what runs.

## Tests
Run unit tests (no network/model downloads):
```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

## Cluster/SLURM
- Scripts live under `slurm/`.
- Logs go to `slurm_logs/` (directory must exist before `sbatch`, hence tracked via `.gitkeep`).
- SLURM scripts source `~/.bashrc`; avoid `set -u` failures by disabling nounset during sourcing.

## Caches / tokens
Large downloads come from Hugging Face; in jobs prefer setting:
- `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`
- `HF_TOKEN` if the model is gated

## W&B
Optional; enable via `--wandb_mode online|offline`. Defaults:
- entity `hauzenberger`
- project `local-lora`

