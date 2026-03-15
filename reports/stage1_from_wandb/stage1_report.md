# Stage 1 Report (W&B offline logs)

This report summarizes Stage 1 (protocol lock sweep) using the offline W&B run files under `wandb/`.

## Scope
- Tasks: sst2, mrpc, rte, cola
- Metric: `eval/validation_score` (per-task primary GLUE score as logged by the Trainer)

## Data Coverage
- Parsed runs: 14
- Target sets: custom=14
- Methods: bd_lora_n8=1, full_ft=1, group_local_equal=5, group_local_param=5, head_only=1, vanilla_lora=1
- Task counts: sst2=14

## Sanity Check (Trainable Params)
For adapter methods, `injection_report.adapter_params_total` is the number of low-rank adapter parameters that were injected, while `trainable_param_summary.adapter_trainable_params` is what actually ended up trainable in the run. If the latter is much larger, it usually means base projection weights were accidentally unfrozen.

Skipped: could not select representative runs (recipe not determined).

## Recommendation (Stage 2 Recipe)
Could not determine a vanilla LoRA best config (missing complete 4-task coverage).

## Vanilla LoRA (r=16) – Top Configs
## Best Methods Under The Chosen Recipe
Skipped: recipe not determined.
