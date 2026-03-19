#!/bin/bash
###############################################################################
# local-lora – Leonardo Booster
# Submit the final GLUE sweep as many smaller SLURM jobs.
#
# Split strategy:
# - one model per job
# - one seed per job
# - one method group per job
# - configurable task chunks per job
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/leonardo_acl_final_glue_ddp_1n4g.sh}"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "Base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

MODEL_NAME_1B="${MODEL_NAME_1B:-/leonardo_work/EUHPC_D31_132/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6}"
MODEL_NAME_3B="${MODEL_NAME_3B:-/leonardo_work/EUHPC_D31_132/models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
MODEL_NAMES="${MODEL_NAMES:-}"
TASKS="${TASKS:-cola,sst2,mrpc,qqp,stsb,mnli,qnli,rte,wnli}"
TASKS_PER_JOB="${TASKS_PER_JOB:-1}"
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

chunk_csv_items() {
  local csv="$1"
  local chunk_size="$2"

  if ! [[ "${chunk_size}" =~ ^[0-9]+$ ]] || [ "${chunk_size}" -lt 1 ]; then
    echo "TASKS_PER_JOB must be a positive integer; got '${chunk_size}'." >&2
    return 1
  fi

  local -a items=()
  IFS=, read -r -a items <<< "${csv}"

  local -a chunk=()
  local item=""
  for item in "${items[@]}"; do
    item="${item// /}"
    [ -z "${item}" ] && continue
    chunk+=("${item}")
    if [ "${#chunk[@]}" -eq "${chunk_size}" ]; then
      (
        IFS=,
        echo "${chunk[*]}"
      )
      chunk=()
    fi
  done

  if [ "${#chunk[@]}" -gt 0 ]; then
    (
      IFS=,
      echo "${chunk[*]}"
    )
  fi
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
task_chunks=()
while IFS= read -r task_chunk; do
  [ -n "${task_chunk}" ] && task_chunks+=("${task_chunk}")
done < <(chunk_csv_items "${TASKS}" "${TASKS_PER_JOB}")

if [ "${#task_chunks[@]}" -eq 0 ]; then
  echo "No tasks resolved from TASKS='${TASKS}'." >&2
  exit 1
fi

jobs_per_model_seed=$(( $(count_enabled "${SUBMIT_HEAD_ONLY}") + $(count_enabled "${SUBMIT_FULL_FT}") + $(count_enabled "${SUBMIT_ADAPTERS}") ))
total_jobs=$(( ${#model_names[@]} * ${#seeds[@]} * jobs_per_model_seed * ${#task_chunks[@]} ))

echo "Submitting split final-GLUE sweep"
echo "  base script    : ${BASE_SCRIPT}"
echo "  tasks          : ${TASKS}"
echo "  tasks/job      : ${TASKS_PER_JOB}"
echo "  task chunks    : ${#task_chunks[@]}"
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
  local task_chunk="$1"
  local method_group="$2"
  local methods="$3"
  local model_name="$4"
  local seed="$5"

  local tag_tasks
  tag_tasks="$(sanitize_token "${task_chunk}")"

  local tag_model
  tag_model="$(model_tag "${model_name}")"

  local tag="final_${tag_model}_seed${seed}_${tag_tasks}_${method_group}"
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
    echo "        TASKS=${task_chunk} MODEL_NAMES=${model_name} SEEDS=${seed} METHODS=${methods} OUTPUT_ROOT=${output_root}"
    return 0
  fi

  local job_id
  job_id="$(
    TASKS="${task_chunk}" \
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
    for task_chunk in "${task_chunks[@]}"; do
      if [ "${SUBMIT_HEAD_ONLY}" = "1" ]; then
        submit_job "${task_chunk}" "head" "head_only" "${model_name}" "${seed}"
      fi
      if [ "${SUBMIT_FULL_FT}" = "1" ]; then
        submit_job "${task_chunk}" "fullft" "full_ft" "${model_name}" "${seed}"
      fi
      if [ "${SUBMIT_ADAPTERS}" = "1" ]; then
        submit_job "${task_chunk}" "adapters" "vanilla_lora,bd_lora,group_local_equal,group_local_param" "${model_name}" "${seed}"
      fi
    done
  done
done
