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
- Leonardo compute nodes have **no outbound internet**; rely on pre-fetched Hugging Face caches + offline mode env vars (see below).
- For 1 node / 4 GPUs on Leonardo there are both:
  - 4× independent workers (task-per-GPU) scripts, and
  - 4‑GPU DDP-per-task scripts (min GPU-hours; tasks run sequentially via `torchrun`).
  - task-per-GPU: `slurm/leonardo/leonardo_*_quick_tasks_1n4g.sh`
  - DDP-per-task: `slurm/leonardo/leonardo_*_ddp_tasks_1n4g.sh`

## Caches / tokens
Large downloads come from Hugging Face; in jobs prefer setting:
- `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`
- `HF_TOKEN` if the model is gated

### Offline HF caches (Leonardo)
If the job has no network access, set:
- `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`

Prefetch GLUE once on a node **with** internet into a shared cache directory:
- `python prefetch_hf_glue.py --tasks cola,sst2,mrpc,rte --offline_check`

### DDP notes (Leonardo)
DDP scripts use `torchrun --nproc_per_node=4` and will scale the *global* batch size:
- global train batch = `per_device_train_batch_size * 4 * gradient_accumulation_steps`
Adjust `PER_DEVICE_TRAIN_BS` / `GRAD_ACCUM` in the SLURM scripts if you want to keep the global batch constant.

## W&B
Optional; enable via `--wandb_mode online|offline`. Defaults:
- entity `hauzenberger`
- project `local-lora`
