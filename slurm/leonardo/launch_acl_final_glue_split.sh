#!/bin/bash
###############################################################################
# local-lora – Leonardo Booster
# Submit the final GLUE sweep as many smaller SLURM jobs.
#
# Split strategy:
# - one model per job
# - one seed per job
# - one method group per job
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/leonardo_acl_final_glue_ddp_1n4g.sh}"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "Base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

MODEL_NAME_1B="${MODEL_NAME_1B:-/leonardo_work/EUHPC_D31_132/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6}"
MODEL_NAME_3B="${MODEL_NAME_3B:-}"
MODEL_NAMES="${MODEL_NAMES:-}"
TASKS="${TASKS:-cola,sst2,mrpc,qqp,stsb,mnli,qnli,rte,wnli}"
SEEDS="${SEEDS:-0,1,2}"

JOB_TIME="${JOB_TIME:-1-00:00:00}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-aclf2}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-runs/leonardo_acl_final_glue_split_$(date +%Y%m%d_%H%M%S)}"
SBATCH_QOS="${SBATCH_QOS:-}"

SUBMIT_HEAD_ONLY="${SUBMIT_HEAD_ONLY:-1}"
SUBMIT_FULL_FT="${SUBMIT_FULL_FT:-1}"
SUBMIT_ADAPTERS="${SUBMIT_ADAPTERS:-1}"
DRY_RUN="${DRY_RUN:-0}"

sanitize_token() {
  printf '%s' "$1" | tr -c '[:alnum:]._-' '_'
}

model_tag() {
  local model_name="$1"
  local base_name
  base_name="$(basename "${model_name}")"
  sanitize_token "${base_name}"
}

count_enabled() {
  local flag="$1"
  if [ "${flag}" = "1" ]; then
    echo 1
  else
    echo 0
  fi
}

if [ -n "${MODEL_NAMES}" ]; then
  IFS=, read -r -a model_names <<< "${MODEL_NAMES}"
else
  model_names=( "${MODEL_NAME_1B}" )
  if [ -n "${MODEL_NAME_3B}" ]; then
    model_names+=( "${MODEL_NAME_3B}" )
  fi
fi
IFS=, read -r -a seeds <<< "${SEEDS}"

jobs_per_model_seed=$(( $(count_enabled "${SUBMIT_HEAD_ONLY}") + $(count_enabled "${SUBMIT_FULL_FT}") + $(count_enabled "${SUBMIT_ADAPTERS}") ))
total_jobs=$(( ${#model_names[@]} * ${#seeds[@]} * jobs_per_model_seed ))

echo "Submitting split final-GLUE sweep"
echo "  base script    : ${BASE_SCRIPT}"
echo "  tasks          : ${TASKS}"
echo "  output root    : ${OUTPUT_ROOT_BASE}"
echo "  job time       : ${JOB_TIME}"
if [ -n "${SBATCH_QOS}" ]; then
  echo "  sbatch qos     : ${SBATCH_QOS}"
else
  echo "  sbatch qos     : <default>"
fi
echo "  models         : ${#model_names[@]}"
echo "  seeds          : ${#seeds[@]}"
echo "  total jobs     : ${total_jobs}"

submit_job() {
  local method_group="$1"
  local methods="$2"
  local model_name="$3"
  local seed="$4"

  local tag_model
  tag_model="$(model_tag "${model_name}")"

  local tag="final_${tag_model}_seed${seed}_${method_group}"
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
    echo "        MODEL_NAMES=${model_name} SEEDS=${seed} METHODS=${methods} OUTPUT_ROOT=${output_root}"
    return 0
  fi

  local job_id
  job_id="$(
    TASKS="${TASKS}" \
    MODEL_NAMES="${model_name}" \
    SEEDS="${seed}" \
    OUTPUT_ROOT="${output_root}" \
    METHODS="${methods}" \
    sbatch "${sbatch_args[@]}" "${BASE_SCRIPT}"
  )"

  echo "submitted ${job_id} -> ${job_name}"
}

for model_name in "${model_names[@]}"; do
  for seed in "${seeds[@]}"; do
    if [ "${SUBMIT_HEAD_ONLY}" = "1" ]; then
      submit_job "head" "head_only" "${model_name}" "${seed}"
    fi
    if [ "${SUBMIT_FULL_FT}" = "1" ]; then
      submit_job "fullft" "full_ft" "${model_name}" "${seed}"
    fi
    if [ "${SUBMIT_ADAPTERS}" = "1" ]; then
      submit_job "adapters" "vanilla_lora,bd_lora,group_local_equal,group_local_param" "${model_name}" "${seed}"
    fi
  done
done
