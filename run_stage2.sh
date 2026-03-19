# optional, only if these tasks are not already in the shared HF cache
HF_HOME="$PWD/runs/hf_home" TRANSFORMERS_CACHE="$HF_HOME/transformers" HF_DATASETS_CACHE="$HF_HOME/datasets" \
python /Users/lukas.hauzenberger/git_repos/local-lora/prefetch_hf_glue.py \
  --tasks cola,sst2,mrpc,qqp,stsb,mnli,qnli,rte,wnli --offline_check

export WANDB_MODE_ARG=offline
export WANDB_DIR=wandb_stage2_final_mlp

export MODEL_NAME_3B="/leonardo_work/EUHPC_D31_132/models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"   # set this (otherwise it runs 1B only)

export LORA_LR=1e-4
export FT_LR=2e-5
export MAX_LENGTH=128
export WARMUP_RATIO=0.05
export SCALING_MODE=rs
export TARGET_SUFFIXES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

export BD_N_VALUES=4
export M_VALUES="16,4"
export GROUP_LOCAL_EQUAL_M_VALUES="16"
export GROUP_LOCAL_PARAM_M_VALUES="4"
export TASKS_PER_JOB=1

export TORCH_COMPILE=1
export OUTPUT_ROOT_BASE="runs/leonardo_acl_final_glue_attn_mlp_$(date +%Y%m%d_%H%M%S)"
export JOB_TIME=1-00:00:00

DRY_RUN=1 bash slurm/leonardo/launch_acl_final_glue_split.sh
bash slurm/leonardo/launch_acl_final_glue_split.sh
