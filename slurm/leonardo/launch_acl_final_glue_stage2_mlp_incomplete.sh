#!/bin/bash
###############################################################################
# local-lora – Leonardo Booster
# Re-submit only the missing Stage 2 MLP runs found in wandb_stage2_mlp.
#
# Missing expected configs from the original sweep:
# - 3B / stsb / seed {0,1}: bd_lora, group_local_equal, group_local_param
# - 3B / stsb / seed 2    : group_local_param
# - 3B / {mnli,qnli,rte,wnli} / seed {0,1,2}: all adapter methods
#
# This corresponds to 55 missing configurations across 15 SLURM jobs.
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/leonardo_acl_final_glue_ddp_1n4g.sh}"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "Base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

MODEL_NAME_3B="${MODEL_NAME_3B:-/leonardo_work/EUHPC_D31_132/models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
MODEL_NAMES="${MODEL_NAMES:-${MODEL_NAME_3B}}"

JOB_TIME="${JOB_TIME:-1-00:00:00}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-aclf2r}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-runs/leonardo_acl_final_glue_attn_mlp_rerun_incomplete_$(date +%Y%m%d_%H%M%S)}"
SBATCH_QOS="${SBATCH_QOS:-}"
DRY_RUN="${DRY_RUN:-0}"

# Keep the same Stage 2 recipe by default, but write new offline runs to a
# separate directory unless the caller overrides these values.
export WANDB_MODE_ARG="${WANDB_MODE_ARG:-offline}"
export WANDB_DIR="${WANDB_DIR:-wandb_stage2_final_mlp_rerun_incomplete}"

export LORA_LR="${LORA_LR:-1e-4}"
export FT_LR="${FT_LR:-2e-5}"
export MAX_LENGTH="${MAX_LENGTH:-128}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export SCALING_MODE="${SCALING_MODE:-rs}"
export TARGET_SUFFIXES="${TARGET_SUFFIXES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

export M_VALUES="${M_VALUES:-16,4}"
export GROUP_LOCAL_EQUAL_M_VALUES="${GROUP_LOCAL_EQUAL_M_VALUES:-16}"
export GROUP_LOCAL_PARAM_M_VALUES="${GROUP_LOCAL_PARAM_M_VALUES:-4}"
export BD_N_VALUES="${BD_N_VALUES:-4}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"

sanitize_token() {
  printf '%s' "$1" | tr -c '[:alnum:]._-' '_'
}

model_tag() {
  local model_name="$1"
  local base_name
  base_name="$(basename "${model_name}")"
  sanitize_token "${base_name}"
}

submit_job() {
  local task="$1"
  local seed="$2"
  local methods="$3"
  local method_group="$4"

  local tag_model
  tag_model="$(model_tag "${MODEL_NAMES}")"

  local tag="rerun_${tag_model}_seed${seed}_${task}_${method_group}"
  local output_root="${OUTPUT_ROOT_BASE}/${tag}"
  local job_name="${JOB_NAME_PREFIX}-${tag}"
  local export_spec="ALL,TASKS,MODEL_NAMES,SEEDS,OUTPUT_ROOT,METHODS,LORA_LR,FT_LR,MAX_LENGTH,WARMUP_RATIO,SCALING_MODE,TARGET_SUFFIXES,M_VALUES,GROUP_LOCAL_EQUAL_M_VALUES,GROUP_LOCAL_PARAM_M_VALUES,BD_N_VALUES,BD_ROW_FACTOR"
  local -a sbatch_args=(
    --parsable
    --time "${JOB_TIME}"
    --job-name "${job_name}"
    --export="${export_spec}"
  )

  if [ -n "${SBATCH_QOS}" ]; then
    sbatch_args+=(--qos "${SBATCH_QOS}")
  fi

  if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN sbatch ${sbatch_args[*]} ${BASE_SCRIPT}"
    echo "        TASKS=${task} MODEL_NAMES=${MODEL_NAMES} SEEDS=${seed} METHODS=${methods} OUTPUT_ROOT=${output_root}"
    return 0
  fi

  local job_id
  job_id="$(
    TASKS="${task}" \
    MODEL_NAMES="${MODEL_NAMES}" \
    SEEDS="${seed}" \
    OUTPUT_ROOT="${output_root}" \
    METHODS="${methods}" \
    sbatch "${sbatch_args[@]}" "${BASE_SCRIPT}"
  )"

  echo "submitted ${job_id} -> ${job_name}"
}

echo "Submitting Stage 2 MLP reruns"
echo "  base script    : ${BASE_SCRIPT}"
echo "  model          : ${MODEL_NAMES}"
echo "  output root    : ${OUTPUT_ROOT_BASE}"
echo "  job time       : ${JOB_TIME}"
echo "  wandb dir      : ${WANDB_DIR}"
if [ -n "${SBATCH_QOS}" ]; then
  echo "  sbatch qos     : ${SBATCH_QOS}"
else
  echo "  sbatch qos     : <default>"
fi
echo "  total jobs     : 15"
echo "  missing configs: 55"

for seed in 0 1; do
  submit_job "stsb" "${seed}" "bd_lora,group_local_equal,group_local_param" "stsb-partial"
done

submit_job "stsb" "2" "group_local_param" "stsb-param"

for task in mnli qnli rte wnli; do
  for seed in 0 1 2; do
    submit_job "${task}" "${seed}" "vanilla_lora,bd_lora,group_local_equal,group_local_param" "adapters"
  done
done
