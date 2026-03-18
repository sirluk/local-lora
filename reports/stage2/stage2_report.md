# Stage 2 (Final GLUE) – Offline W&B Report

Parsed 228 offline run files from `wandb_stage2/wandb/`.

- Models: Llama-3.2-1B-Instruct, Qwen3-4B-Instruct-2507
- Tasks: cola, sst2, mrpc, qqp, stsb, mnli, qnli, rte, wnli
- Seeds observed: 0, 1, 2
- Primary metric: mean over GLUE tasks excluding WNLI (MNLI = mean of matched/mismatched).

## Recipe Check (Unique Values Found)
```json
{
  "grouping_mode": [
    "contiguous"
  ],
  "learning_rate": [
    2e-05,
    0.0001
  ],
  "max_length": [
    "128"
  ],
  "perm_seed": [
    "None"
  ],
  "scaling_mode": [
    "rs"
  ],
  "target_set": [
    "attn_mlp"
  ],
  "warmup_ratio": [
    "0.05"
  ]
}
```

## Model: Llama-3.2-1B-Instruct

| method | seeds | complete_seeds (need 8) | avg_glue_excl_wnli (mean ± sd) | Δ vs vanilla | wnli_mean | mean_train_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| head_only | 0,1,2 | 3/3 | 0.592 ± 0.0022 | -0.2744 | 0.5587 | 293.78 |
| vanilla_lora_r16 | 0,1,2 | 3/3 | 0.8664 ± 0.0011 | +0 | 0.3709 | 1694.11 |
| group_local_equal_r16_m16 | 0,1,2 | 3/3 | 0.8672 ± 0.0011 | +0.0008 | 0.3662 | 1979.35 |
| group_local_param_r16_m16_d0 | 0,1,2 | 3/3 | 0.8672 ± 0.0011 | +0.0008 | 0.3662 | 1945.06 |
| bd_lora_r16_n4_rowblock_a | 0,1,2 | 3/3 | 0.8565 ± 0.0023 | -0.0099 | 0.4977 | 2058.75 |
| full_ft | 0,1,2 | 3/3 | 0.8694 ± 0.0033 | +0.0031 | 0.3192 | 2136.14 |

### Adapter Param Sanity
| method | adapter_trainable_params | injected_adapter_params | ratio | flag |
| --- | --- | --- | --- | --- |
| vanilla_lora_r16 | 11272192 | 11272192 | 1 | OK |
| group_local_equal_r16_m16 | 11272192 | 11272192 | 1 | OK |
| group_local_param_r16_m16_d0 | 11272192 | 11272192 | 1 | OK |
| bd_lora_r16_n4_rowblock_a | 5570560 | 5570560 | 1 | OK |

### Per-Task Scores (mean ± sd over available seeds)
| task | head_only | vanilla_lora_r16 | group_local_equal_r16_m16 | group_local_param_r16_m16_d0 | bd_lora_r16_n4_rowblock_a | full_ft |
| --- | --- | --- | --- | --- | --- | --- |
| cola | 0.2658±0.0155 | 0.629±0.0057 | 0.6283±0.0021 | 0.6283±0.0021 | 0.6342±0.0196 | 0.6374±0.0218 |
| sst2 | 0.8911±0.0091 | 0.9587±0.003 | 0.9591±0.0052 | 0.9591±0.0052 | 0.9557±0.0026 | 0.9541±0.0023 |
| mrpc | 0.7405±0.01 | 0.8777±0.0023 | 0.8829±0.003 | 0.8829±0.003 | 0.8773±0.0071 | 0.8942±0.0099 |
| qqp | 0.7349±0.004 | 0.8904±0.0004 | 0.8903±0.0003 | 0.8903±0.0003 | 0.8811±0.0008 | 0.8973±0.001 |
| stsb | 0.3177±0.0423 | 0.904±0.0017 | 0.9035±0.0014 | 0.9035±0.0014 | 0.8899±0.0038 | 0.9068±0.0021 |
| mnli | 0.5592±0.0115 | 0.8915±0.0012 | 0.8916±0.0014 | 0.8916±0.0014 | 0.887±0.0003 | 0.8909±0.0012 |
| qnli | 0.6972±0.0034 | 0.9372±0.0011 | 0.937±0.0004 | 0.937±0.0004 | 0.9326±0.0019 | 0.9372±0.0015 |
| rte | 0.5295±0.0328 | 0.8424±0.0083 | 0.8448±0.0063 | 0.8448±0.0063 | 0.7942±0.0063 | 0.8375±0.0096 |
| wnli | 0.5587±0.0533 | 0.3709±0.0695 | 0.3662±0.0614 | 0.3662±0.0614 | 0.4977±0.0215 | 0.3192±0.0215 |

## Model: Qwen3-4B-Instruct-2507

| method | seeds | complete_seeds (need 8) | avg_glue_excl_wnli (mean ± sd) | Δ vs vanilla | wnli_mean | mean_train_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| head_only | 1,2 | 2/2 | 0.6707 ± 0.0019 | -0.2062 | 0.4648 | 576.88 |
| vanilla_lora_r16 | 2 | 0/1 (incomplete) | 0.877 ± 0 | +0 |  | 3571.75 |
| group_local_equal_r16_m16 | 2 | 0/1 (incomplete) | 0.8742 ± 0 | -0.0027 |  | 3979.44 |
| group_local_param_r16_m16_d0 | 2 | 0/1 (incomplete) | 0.8742 ± 0 | -0.0027 |  | 3977.83 |
| bd_lora_r16_n4_rowblock_a | 2 | 0/1 (incomplete) | 0.8612 ± 0 | -0.0157 |  | 4200.04 |
| full_ft | 0,1,2 | 3/3 | 0.8867 ± 0.0026 | +0.0098 | 0.7418 | 6510.35 |

### Adapter Param Sanity
| method | adapter_trainable_params | injected_adapter_params | ratio | flag |
| --- | --- | --- | --- | --- |
| vanilla_lora_r16 | 33030144 | 32833555 | 1.006 | OK |
| group_local_equal_r16_m16 | 33030144 | 32776211 | 1.008 | OK |
| group_local_param_r16_m16_d0 | 33030144 | 32776211 | 1.008 | OK |
| bd_lora_r16_n4_rowblock_a | 15998976 | 15919111 | 1.005 | OK |

### Per-Task Scores (mean ± sd over available seeds)
| task | head_only | vanilla_lora_r16 | group_local_equal_r16_m16 | group_local_param_r16_m16_d0 | bd_lora_r16_n4_rowblock_a | full_ft |
| --- | --- | --- | --- | --- | --- | --- |
| cola | 0.2749±0.0132 | 0.6992±0 | 0.6828±0 | 0.6828±0 | 0.6549±0 | 0.6586±0.0008 |
| sst2 | 0.8847±0.0041 | 0.9644±0 | 0.9622±0 | 0.9622±0 | 0.9622±0 | 0.9656±0.0011 |
| mrpc | 0.7711±0.0069 | 0.9182±0 | 0.9163±0 | 0.9163±0 | 0.9076±0 | 0.9041±0.0045 |
| qqp | 0.8015±0.0024 | 0.8946±0 | 0.8951±0 | 0.8951±0 | 0.8886±0 | 0.9034±0.0006 |
| stsb | 0.5016±0.02 | 0.9084±0 | 0.9147±0 | 0.9147±0 | 0.8929±0 | 0.9055±0.011 |
| mnli | 0.6831±0.0007 |  |  |  |  | 0.9104±0.0009 |
| qnli | 0.8407±0.0028 |  |  |  |  | 0.9582±0.0017 |
| rte | 0.6083±0.0026 |  |  |  |  | 0.8881±0.0165 |
| wnli | 0.4648±0.0398 |  |  |  |  | 0.7418±0.0215 |

