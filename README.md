# Group-Local LoRA (local-lora)

This repo implements:

- **Vanilla LoRA** (`LoRALinear`)
- **Group-Local LoRA** (`GroupLocalLoRALinear`)

…plus utilities to inject adapters into Llama attention projections and run **GLUE** fine-tuning/evaluation via Hugging Face `Trainer` + `evaluate`.

## Environment

This code assumes the conda env `torch210` (already present on this machine) which includes `torch`, `transformers`, `datasets`, and `evaluate`.

## Quickstart (CPU smoke test)

```bash
/nfs-gpu/xlstm-distillation/miniconda3/bin/conda run --no-capture-output -n torch210 \
  python train_glue.py \
  --task sst2 \
  --adapter_type head_only \
  --max_train_samples 100 \
  --max_eval_samples 100 \
  --output_dir runs/smoke_head_only_sst2
```

## Run a small GLUE suite

```bash
/nfs-gpu/xlstm-distillation/miniconda3/bin/conda run --no-capture-output -n torch210 \
  python run_glue_suite.py \
  --tasks sst2,mrpc,rte,qnli \
  --output_root runs/glue_suite
```

Notes:
- Downloading `meta-llama/Llama-3.2-1B-Instruct` requires Hugging Face access. Set `HF_TOKEN` if needed.
- If you override `HF_HOME` (the SLURM scripts do), a prior `huggingface-cli login` token in `~/.cache/huggingface/token` may not be picked up unless `HF_TOKEN` is exported (the SLURM scripts try to auto-load it).
- The scripts write per-run artifacts to each run directory and append a row to `results.csv`.

## Offline / no-internet compute nodes (e.g. Leonardo)

If your training nodes have **no outbound internet**, you must **prefetch** the GLUE dataset + metric once on a node *with* internet (login/datamover), into a cache directory that is visible from compute nodes.

Example (prefetch GLUE into the same cache layout used by the SLURM scripts):

```bash
export HF_HOME="$PWD/runs/hf_home"
python prefetch_hf_glue.py --tasks cola,sst2,mrpc,rte --offline_check
```

Then run your job with:
- `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE` pointing to that shared cache location, and
- `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (the Leonardo SLURM scripts default these to `1`).

## Weights & Biases (optional)

This repo can log runs to W&B via the built-in Transformers integration.

1) Install:

```bash
pip install wandb
```

2) Enable logging by passing `--wandb_mode online` (or `offline`). Defaults:
- entity: `hauzenberger`
- project: `local-lora`

Example:

```bash
python run_glue_suite.py --output_root runs/glue_suite --wandb_mode online
```
