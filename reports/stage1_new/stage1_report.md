# Stage 1 Report (W&B offline logs)

This report summarizes Stage 1 (protocol lock sweep) using the offline W&B run files under `wandb_stage1_new/`.

## Scope
- Tasks: sst2, mrpc, rte, cola
- Metric: `eval/validation_score` (per-task primary GLUE score as logged by the Trainer)

## Data Coverage
- Parsed runs: 1664
- Target sets: attn_only=1664
- Methods: bd_lora_n4=128, bd_lora_n8=128, group_local_equal=640, group_local_param=640, vanilla_lora=128
- Task counts: cola=416, mrpc=416, rte=416, sst2=416

## Sanity Check (Trainable Params)
For adapter methods, `injection_report.adapter_params_total` is the number of low-rank adapter parameters that were injected, while `trainable_param_summary.adapter_trainable_params` is what actually ended up trainable in the run. If the latter is much larger, it usually means base projection weights were accidentally unfrozen.

| method | adapter_trainable_params | injected_adapter_params | ratio | flag |
| --- | --- | --- | --- | --- |
| vanilla_lora | 1703936 | 1703936 | 1 | OK |
| group_local_equal | 1703936 | 1703936 | 1 | OK |
| group_local_param | 1703936 | 1703936 | 1 | OK |
| bd_lora | 1130496 | 1130496 | 1 | OK |

## Recommendation (Stage 2 Recipe)
Chosen based on best Avg (mean over the Stage-1 task subset) for `vanilla_lora r=16`:

| target_set | scaling | lr | max_length | warmup | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_only | rs | 1e-04 | 256 | 0.05 | 0.8125 | 0.6182 | 0.8808 | 0.8014 | 0.9495 |

Note: no `attn_mlp` runs were found in this export, so the target-set choice cannot be compared here.

## Vanilla LoRA (r=16) – Top Configs
### attn_only
| target_set | scaling | lr | max_length | warmup | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_only | rs | 1e-04 | 256 | 0.05 | 0.8125 | 0.6182 | 0.8808 | 0.8014 | 0.9495 |
| attn_only | standard | 1e-04 | 256 | 0.05 | 0.8116 | 0.6182 | 0.8808 | 0.7978 | 0.9495 |
| attn_only | standard | 1e-04 | 128 | 0.05 | 0.8107 | 0.6182 | 0.8808 | 0.7942 | 0.9495 |
| attn_only | rs | 1e-04 | 128 | 0.05 | 0.8107 | 0.6182 | 0.8808 | 0.7942 | 0.9495 |
| attn_only | standard | 1e-04 | 128 | 0 | 0.8095 | 0.6213 | 0.8742 | 0.7906 | 0.9518 |

## Best Methods Under The Chosen Recipe
| method | scaling | lr | max_length | warmup | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head_only (best lr) |  |  |  |  |  |  |  |  |  |
| full_ft (best lr) |  |  |  |  |  |  |  |  |  |
| vanilla_lora r=16 | rs | 1e-04 | 256 | 0.05 | 0.8125 | 0.6182 | 0.8808 | 0.8014 | 0.9495 |
| group_local_equal r=16 m=16 | rs | 1e-04 | 256 | 0.05 | 0.8111 | 0.611 | 0.8787 | 0.8051 | 0.9495 |
| group_local_param r=16 m=16 d=0 | rs | 1e-04 | 256 | 0.05 | 0.8102 | 0.611 | 0.8787 | 0.8014 | 0.9495 |
| bd_lora r=16 n=4 | rs | 1e-04 | 256 | 0.05 | 0.7716 | 0.5558 | 0.8525 | 0.7365 | 0.9415 |
| bd_lora r=16 n=8 | rs | 1e-04 | 256 | 0.05 | 0.7409 | 0.5532 | 0.8418 | 0.6318 | 0.9369 |

## Group-Local Sweep Under The Chosen Recipe
### group_local_equal (r=16)
| m | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.71 | 0.5369 | 0.7958 | 0.5704 | 0.9369 |
| 2 | 0.7409 | 0.5532 | 0.8418 | 0.6318 | 0.9369 |
| 4 | 0.7716 | 0.5558 | 0.8525 | 0.7365 | 0.9415 |
| 8 | 0.7908 | 0.5869 | 0.8519 | 0.7762 | 0.9484 |
| 16 | 0.8111 | 0.611 | 0.8787 | 0.8051 | 0.9495 |

### group_local_param
| m | r | delta | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 434176 | 0.7012 | 0.5211 | 0.7919 | 0.556 | 0.9358 |
| 2 | 32 | 475136 | 0.7338 | 0.5548 | 0.8298 | 0.6101 | 0.9404 |
| 4 | 16 | 491520 | 0.7716 | 0.5558 | 0.8525 | 0.7365 | 0.9415 |
| 8 | 16 | 327680 | 0.7908 | 0.5869 | 0.8519 | 0.7762 | 0.9484 |
| 16 | 16 | 0 | 0.8102 | 0.611 | 0.8787 | 0.8014 | 0.9495 |

