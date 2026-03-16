# Stage 1 Report (W&B offline logs)

This report summarizes Stage 1 (protocol lock sweep) using the offline W&B run files under `wandb_stage1_mlp/`.

## Scope
- Tasks: sst2, mrpc, rte, cola
- Metric: `eval/validation_score` (per-task primary GLUE score as logged by the Trainer)

## Data Coverage
- Parsed runs: 1776
- Target sets: attn_mlp=1776
- Methods: bd_lora_n4=128, bd_lora_n8=128, full_ft=48, group_local_equal=640, group_local_param=640, head_only=64, vanilla_lora=128
- Task counts: cola=444, mrpc=444, rte=444, sst2=444

## Sanity Check (Trainable Params)
For adapter methods, `injection_report.adapter_params_total` is the number of low-rank adapter parameters that were injected, while `trainable_param_summary.adapter_trainable_params` is what actually ended up trainable in the run. If the latter is much larger, it usually means base projection weights were accidentally unfrozen.

| method | adapter_trainable_params | injected_adapter_params | ratio | flag |
| --- | --- | --- | --- | --- |
| vanilla_lora | 11272192 | 11272192 | 1 | OK |
| group_local_equal | 11272192 | 11272192 | 1 | OK |
| group_local_param | 11272192 | 11272192 | 1 | OK |
| bd_lora | 4620288 | 4620288 | 1 | OK |

## Recommendation (Stage 2 Recipe)
Chosen based on best Avg (mean over the Stage-1 task subset) for `vanilla_lora r=16`:

| target_set | scaling | lr | max_length | warmup | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_mlp | rs | 1e-04 | 128 | 0.05 | 0.8303 | 0.6319 | 0.8795 | 0.852 | 0.9576 |

## Vanilla LoRA (r=16) – Top Configs
### attn_mlp
| target_set | scaling | lr | max_length | warmup | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_mlp | rs | 1e-04 | 128 | 0.05 | 0.8303 | 0.6319 | 0.8795 | 0.852 | 0.9576 |
| attn_mlp | standard | 1e-04 | 128 | 0.05 | 0.8303 | 0.6319 | 0.8795 | 0.852 | 0.9576 |
| attn_mlp | rs | 1e-04 | 256 | 0 | 0.8296 | 0.6279 | 0.8775 | 0.852 | 0.961 |
| attn_mlp | standard | 1e-04 | 256 | 0 | 0.8296 | 0.6279 | 0.8775 | 0.852 | 0.961 |
| attn_mlp | standard | 1e-04 | 256 | 0.05 | 0.8294 | 0.6319 | 0.8795 | 0.8484 | 0.9576 |

## Best Methods Under The Chosen Recipe
| method | scaling | lr | max_length | warmup | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head_only (best lr) | standard | 1e-04 | 128 | 0.05 | 0.5962 | 0.2779 | 0.7316 | 0.4946 | 0.8807 |
| full_ft (best lr) | standard | 2e-05 | 128 | 0.05 | 0.8271 | 0.6174 | 0.8911 | 0.8412 | 0.9587 |
| vanilla_lora r=16 | rs | 1e-04 | 128 | 0.05 | 0.8303 | 0.6319 | 0.8795 | 0.852 | 0.9576 |
| group_local_equal r=16 m=16 | rs | 1e-04 | 128 | 0.05 | 0.8307 | 0.6284 | 0.8838 | 0.852 | 0.9587 |
| group_local_param r=16 m=16 d=0 | rs | 1e-04 | 128 | 0.05 | 0.8307 | 0.6284 | 0.8838 | 0.852 | 0.9587 |
| bd_lora r=16 n=4 | rs | 1e-04 | 128 | 0.05 | 0.8129 | 0.6197 | 0.8801 | 0.7978 | 0.9541 |
| bd_lora r=16 n=8 | rs | 1e-04 | 128 | 0.05 | 0.7967 | 0.5941 | 0.8682 | 0.7762 | 0.9484 |

## Group-Local Sweep Under The Chosen Recipe
### group_local_equal (r=16)
| m | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.783 | 0.5946 | 0.8576 | 0.7292 | 0.9507 |
| 2 | 0.8035 | 0.5998 | 0.8742 | 0.787 | 0.953 |
| 4 | 0.8164 | 0.6223 | 0.8804 | 0.8123 | 0.9507 |
| 8 | 0.8243 | 0.6436 | 0.8822 | 0.8159 | 0.9553 |
| 16 | 0.8307 | 0.6284 | 0.8838 | 0.852 | 0.9587 |

### group_local_param
| m | r | delta | avg | cola | mrpc | rte | sst2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 409600 | 0.7859 | 0.6039 | 0.8574 | 0.7329 | 0.9495 |
| 2 | 32 | 32768 | 0.8052 | 0.6141 | 0.8762 | 0.7798 | 0.9507 |
| 4 | 32 | 720896 | 0.8176 | 0.6406 | 0.874 | 0.8051 | 0.9507 |
| 8 | 32 | 2228224 | 0.8271 | 0.6355 | 0.8907 | 0.8303 | 0.9518 |
| 16 | 16 | 0 | 0.8307 | 0.6284 | 0.8838 | 0.852 | 0.9587 |

