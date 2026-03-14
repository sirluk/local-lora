# Stage 1 Report (W&B offline logs)

This report summarizes Stage 1 (protocol lock sweep) using the offline W&B run files under `wandb_stage1/`.

## Scope
- Tasks: sst2, mrpc, rte, cola
- Metric: `eval/validation_score` (per-task primary GLUE score as logged by the Trainer)

## Data Coverage
- Parsed runs: 1776
- Target sets: attn_only=1776
- Methods: bd_lora_n4=128, bd_lora_n8=128, full_ft=48, group_local_equal=640, group_local_param=640, head_only=64, vanilla_lora=128
- Task counts: cola=444, mrpc=444, rte=444, sst2=444

## Sanity Check (Trainable Params)
For adapter methods, `injection_report.adapter_params_total` is the number of low-rank adapter parameters that were injected, while `trainable_param_summary.adapter_trainable_params` is what actually ended up trainable in the run. If the latter is much larger, it usually means base projection weights were accidentally unfrozen.

| method | adapter_trainable_params | injected_adapter_params | ratio | flag |
| --- | --- | --- | --- | --- |
| vanilla_lora | 85590016 | 1703936 | 50.231 | WARN |
| group_local_equal | 85590016 | 1703936 | 50.231 | WARN |
| group_local_param | 85590016 | 1703936 | 50.231 | WARN |
| bd_lora | 85016576 | 1130496 | 75.203 | WARN |

Warning: These Stage-1 runs appear to have trained far more parameters than the injected LoRA adapters (ratio >> 1). This indicates the wrapped base Linear weights were likely unfrozen during training, so the sweep is closer to partial fine-tuning than vanilla LoRA.
In this repo, the likely cause was `unfreeze_adapter_params()` recursing into `module.base`; this has been fixed to only unfreeze adapter parameters (see `local_lora/inject.py`).
Recommendation: re-run Stage 1 (and Stage 2) after the fix if you need true LoRA vs Group-Local LoRA comparisons.

## Recommendation (Stage 2 Recipe)
Chosen based on best Avg (mean over the Stage-1 task subset) for `vanilla_lora r=16`:

| target_set | scaling | lr | max_length | warmup | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_only | rs | 5e-05 | 256 | 0.05 | 0.8309 | 0.6182 | 0.8944 | 0.8592 | 0.9518 |

Note: no `attn_mlp` runs were found in this export, so the target-set choice cannot be compared here.

## Vanilla LoRA (r=16) – Top Configs
### attn_only
| target_set | scaling | lr | max_length | warmup | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_only | rs | 5e-05 | 256 | 0.05 | 0.8309 | 0.6182 | 0.8944 | 0.8592 | 0.9518 |
| attn_only | standard | 5e-05 | 256 | 0.05 | 0.83 | 0.6182 | 0.8944 | 0.8556 | 0.9518 |
| attn_only | standard | 5e-05 | 128 | 0 | 0.8288 | 0.6153 | 0.8881 | 0.852 | 0.9599 |
| attn_only | rs | 5e-05 | 128 | 0 | 0.8288 | 0.6153 | 0.8881 | 0.852 | 0.9599 |
| attn_only | rs | 5e-05 | 256 | 0 | 0.8279 | 0.6153 | 0.8881 | 0.8484 | 0.9599 |

## Best Methods Under The Chosen Recipe
| method | scaling | lr | max_length | warmup | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head_only (best lr) | standard | 1e-04 | 256 | 0.05 | 0.5976 | 0.2789 | 0.7316 | 0.4982 | 0.8819 |
| full_ft (best lr) | standard | 2e-05 | 256 | 0.05 | 0.8258 | 0.6136 | 0.8907 | 0.8448 | 0.9541 |
| vanilla_lora r=16 | rs | 5e-05 | 256 | 0.05 | 0.8309 | 0.6182 | 0.8944 | 0.8592 | 0.9518 |
| group_local_equal r=16 m=1 | rs | 5e-05 | 256 | 0.05 | 0.8324 | 0.6414 | 0.8903 | 0.8448 | 0.953 |
| group_local_param r=16 m=4 d=491520 | rs | 5e-05 | 256 | 0.05 | 0.8281 | 0.6172 | 0.8986 | 0.8412 | 0.9553 |
| bd_lora r=16 n=4 | rs | 5e-05 | 256 | 0.05 | 0.8299 | 0.6172 | 0.8986 | 0.8484 | 0.9553 |
| bd_lora r=16 n=8 | rs | 5e-05 | 256 | 0.05 | 0.8261 | 0.6246 | 0.8903 | 0.8412 | 0.9484 |

## Group-Local Sweep Under The Chosen Recipe
### group_local_equal (r=16)
| m | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8324 | 0.6414 | 0.8903 | 0.8448 | 0.953 |
| 2 | 0.8261 | 0.6246 | 0.8903 | 0.8412 | 0.9484 |
| 4 | 0.829 | 0.6172 | 0.8986 | 0.8448 | 0.9553 |
| 8 | 0.823 | 0.5995 | 0.8946 | 0.8448 | 0.953 |
| 16 | 0.6769 | 0.0029 | 0.8986 | 0.852 | 0.9541 |

### group_local_param
| m | r | delta | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 434176 | 0.8266 | 0.6238 | 0.8881 | 0.8448 | 0.9495 |
| 2 | 32 | 475136 | 0.8274 | 0.6162 | 0.892 | 0.8448 | 0.9564 |
| 4 | 16 | 491520 | 0.8281 | 0.6172 | 0.8986 | 0.8412 | 0.9553 |
| 8 | 16 | 327680 | 0.823 | 0.5995 | 0.8946 | 0.8448 | 0.953 |
| 16 | 16 | 0 | 0.6778 | 0.0029 | 0.8986 | 0.8556 | 0.9541 |

