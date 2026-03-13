# ACL Experiment Plan: Group-Local LoRA vs BD-LoRA (arXiv:2510.23346)

This repo supports the baselines + ablations needed for an ACL-style paper:

- `head_only` (sanity / probe)
- `full_ft` (upper bound baseline)
- `vanilla_lora`
- `group_local` (equal-r and parameter-matched)
- `bd_lora` (Block-Diagonal LoRA baseline; mapping from arXiv:2510.23346)

GLUE is run on **validation** splits; report **WNLI separately** and compute **Avg GLUE excluding WNLI**.
For MNLI, average `validation_matched` and `validation_mismatched`.

## Leonardo SLURM scripts (DDP, 1 node / 4 GPUs)
Scripts live in `slurm/leonardo/` and assume **offline** execution (HF caches required).

- Stage 0 smoke: `slurm/leonardo/leonardo_acl_smoke_ddp_1n4g.sh`
- Stage 1 protocol lock (attn-only): `slurm/leonardo/leonardo_acl_protocol_lock_attn_only_ddp_1n4g.sh`
- Stage 1 protocol lock (attn+mlp): `slurm/leonardo/leonardo_acl_protocol_lock_attn_mlp_ddp_1n4g.sh`
- Stage 2 final GLUE: `slurm/leonardo/leonardo_acl_final_glue_ddp_1n4g.sh`
- Ablation A1 row-factor: `slurm/leonardo/leonardo_acl_ablation_row_factor_ddp_1n4g.sh`
- Ablation A2 granularity: `slurm/leonardo/leonardo_acl_ablation_granularity_ddp_1n4g.sh`
- Ablation A3 grouping modes: `slurm/leonardo/leonardo_acl_ablation_grouping_modes_ddp_1n4g.sh`
- Ablation A4 low-data: `slurm/leonardo/leonardo_acl_ablation_low_data_ddp_1n4g.sh`

Submit example:
```bash
sbatch slurm/leonardo/leonardo_acl_protocol_lock_attn_only_ddp_1n4g.sh
```

Override key variables (example):
```bash
sbatch --export=ALL,OUTPUT_ROOT=/path/to/runs,MODEL_NAME_3B=/path/to/3b/snapshot,LORA_LR=1e-4,FT_LR=2e-5 \
  slurm/leonardo/leonardo_acl_final_glue_ddp_1n4g.sh
```

Optional speed knob (all Leonardo scripts support this):
```bash
sbatch --export=ALL,TORCH_COMPILE=1,TORCH_COMPILE_BACKEND=inductor,TORCH_COMPILE_MODE=reduce-overhead \
  slurm/leonardo/leonardo_acl_protocol_lock_attn_only_ddp_1n4g.sh
```

## Stage 0 — Sanity + instrumentation (fast)
Run a couple of short runs to confirm:
- adapters inject correctly (1B + 3B)
- trainable/adaptor parameter counts look right
- `results.csv` rows are appended

Leonardo: `sbatch slurm/leonardo/leonardo_acl_smoke_ddp_1n4g.sh`

Example:
```bash
conda run -n torch210 python train_glue.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --task sst2 \
  --adapter_type bd_lora \
  --r 16 --bd_n 8 --bd_row_factor block_a \
  --max_train_samples 2000 --max_eval_samples 1000 \
  --bf16 \
  --output_dir runs/smoke_bd_lora_sst2
```

## Stage 1 — Protocol lock sweep (1B only, seed=0)
Tasks: `{sst2, mrpc, rte, cola}`.

Leonardo:
- attention-only: `sbatch slurm/leonardo/leonardo_acl_protocol_lock_attn_only_ddp_1n4g.sh`
- attention+mlp: `sbatch slurm/leonardo/leonardo_acl_protocol_lock_attn_mlp_ddp_1n4g.sh`
- split launcher (safer for strict walltime limits): `bash slurm/leonardo/launch_acl_protocol_lock_split.sh`

If your per-job walltime is tight (for example 2 days on Leonardo), prefer the split launcher.
It submits one smaller job per hyperparameter setting instead of serializing the full Stage 1 grid
inside a single `torchrun` allocation. Use `VARIANT=attn_only` or `VARIANT=attn_mlp` and
optionally `DRY_RUN=1` to inspect the generated submissions first.

Grid (shared where applicable):
- LoRA-like LR: `{1e-5, 2e-5, 5e-5, 1e-4}`
- Full-FT LR: `{5e-6, 1e-5, 2e-5}`
- `max_length ∈ {128, 256}`
- `warmup_ratio ∈ {0.0, 0.05}`
- `scaling_mode ∈ {standard, rs}` (compare once; pick default globally)
- Adapter targets:
  - attention-only: `q_proj,v_proj`
  - attention+mlp: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

Command template:
```bash
conda run -n torch210 python run_glue_suite.py \
  --methods all \
  --model_names meta-llama/Llama-3.2-1B-Instruct \
  --tasks sst2,mrpc,rte,cola \
  --seeds 0 \
  --learning_rates 1e-5,2e-5,5e-5,1e-4 \
  --full_ft_learning_rates 5e-6,1e-5,2e-5 \
  --max_lengths 128,256 \
  --warmup_ratios 0.0,0.05 \
  --scaling_modes standard,rs \
  --bd_n_values 4,8 \
  --bd_row_factor block_a \
  --grouping_mode contiguous \
  --target_suffixes q_proj,v_proj \
  --bf16 \
  --output_root runs/protocol_lock_attn_only
```

Repeat with:
```bash
--target_suffixes q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
--output_root runs/protocol_lock_attn_mlp
```

Selection rules (make these explicit in the paper):
- Choose one “paper-default” target set (`attn_only` vs `attn_mlp`) based on Avg GLUE (excl. WNLI) for `vanilla_lora r=16` (seed=0).
- Choose the best `group_local` configuration(s):
  - **equal-r**: best `m` for `r=16`
  - **param-matched**: best `m` among the parameter-matched runs
- For BD-LoRA choose `N ∈ {4,8}` (default report `N=8` unless `N=4` clearly wins).

Summarize:
```bash
python summarize_glue_results.py --results_csv runs/protocol_lock_attn_only/results.csv
python summarize_glue_results.py --results_csv runs/protocol_lock_attn_mlp/results.csv
```

## Stage 2 — Final GLUE (1B + 3B, seeds={0,1,2})
Run on full GLUE (report WNLI separately):
`cola,sst2,mrpc,qqp,stsb,mnli,qnli,rte,wnli`

Use the fixed recipe from Stage 1 (single LR/max_length/warmup/scaling/targets choice).

Leonardo: `sbatch slurm/leonardo/leonardo_acl_final_glue_ddp_1n4g.sh`

Example (replace `<...>` with chosen settings):
```bash
conda run -n torch210 python run_glue_suite.py \
  --methods all \
  --model_names meta-llama/Llama-3.2-1B-Instruct,meta-llama/Llama-3.2-3B-Instruct \
  --tasks cola,sst2,mrpc,qqp,stsb,mnli,qnli,rte,wnli \
  --seeds 0,1,2 \
  --learning_rates <lora_lr> \
  --full_ft_learning_rates <ft_lr> \
  --max_lengths <max_length> \
  --warmup_ratios <warmup_ratio> \
  --scaling_modes <standard_or_rs> \
  --bd_n_values <4_or_8> \
  --bd_row_factor block_a \
  --grouping_mode contiguous \
  --target_suffixes <chosen_suffixes> \
  --bf16 \
  --output_root runs/final_glue
```

Then:
```bash
python summarize_glue_results.py --results_csv runs/final_glue/results.csv
```

## “Different from Block-LoRA” ablations (paper figures)

### A1) Row-factor choice (key BD-LoRA differentiator)
On `{mrpc, rte, cola}` (1B; seeds 0/1/2), compare for BD-LoRA row-parallel-like layers (`o_proj`, `down_proj`):
- `--bd_row_factor block_a` (paper baseline mapping)
- `--bd_row_factor block_b`
- `--bd_row_factor dense`

Leonardo: `sbatch slurm/leonardo/leonardo_acl_ablation_row_factor_ddp_1n4g.sh`

Template:
```bash
conda run -n torch210 python run_glue_suite.py \
  --methods bd_lora \
  --model_names meta-llama/Llama-3.2-1B-Instruct \
  --tasks mrpc,rte,cola \
  --seeds 0,1,2 \
  --bd_n_values 4,8 \
  --bd_row_factor block_a \
  --scaling_modes <chosen> \
  --learning_rates <chosen> \
  --max_lengths <chosen> \
  --warmup_ratios <chosen> \
  --target_suffixes <chosen_suffixes> \
  --bf16 \
  --output_root runs/abl_row_factor_block_a
```
Repeat for `block_b` / `dense`.

### A2) Granularity (N / m sweep at matched budget)
- BD-LoRA: `--bd_n_values 1,2,4,8,16` (requires `r % N == 0` with current implementation)
- Group-local: `--m_values 16,8,4,2,1` and use `group_local_param` (parameter-matched) to compare fairly.

Leonardo: `sbatch slurm/leonardo/leonardo_acl_ablation_granularity_ddp_1n4g.sh`

### A3) Head-aligned vs random grouping
Use:
- `--grouping_mode head_aligned` (requires model head_dim inference; errors if divisibility doesn’t hold)
- `--grouping_mode random --perm_seed 0` as the control

Leonardo: `sbatch slurm/leonardo/leonardo_acl_ablation_grouping_modes_ddp_1n4g.sh`

### A4) Low-data regime
Tasks `{rte, mrpc, cola}` with `--max_train_samples 128|512|2048|<unset>` (full).
Compare `vanilla_lora`, `group_local`, `bd_lora`, `full_ft`.

Leonardo: `sbatch slurm/leonardo/leonardo_acl_ablation_low_data_ddp_1n4g.sh`

## Practical notes
- `results.csv` schema is versioned implicitly: if you ran older code, use a **fresh** `--results_csv` (or delete the old file).
- Offline nodes (Leonardo): prefetch GLUE + set offline env vars (see `README.md` / `AGENTS.md`).
