#!/bin/bash
###############################################################################
# local-lora – Leonardo Booster
# Submit the granularity ablation as one job per method family.
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/leonardo_acl_ablation_granularity_ddp_1n4g.sh}"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "Base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

JOB_TIME="${JOB_TIME:-1-00:00:00}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-acla2}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-runs/leonardo_acl_granularity_split_$(date +%Y%m%d_%H%M%S)}"
SBATCH_QOS="${SBATCH_QOS:-}"

SUBMIT_VANILLA="${SUBMIT_VANILLA:-1}"
SUBMIT_BD="${SUBMIT_BD:-1}"
SUBMIT_GROUP_LOCAL_PARAM="${SUBMIT_GROUP_LOCAL_PARAM:-1}"
DRY_RUN="${DRY_RUN:-0}"

count_enabled() {
  local flag="$1"
  if [ "${flag}" = "1" ]; then
    echo 1
  else
    echo 0
  fi
}

total_jobs=$(( $(count_enabled "${SUBMIT_VANILLA}") + $(count_enabled "${SUBMIT_BD}") + $(count_enabled "${SUBMIT_GROUP_LOCAL_PARAM}") ))

echo "Submitting split granularity ablation"
echo "  base script    : ${BASE_SCRIPT}"
echo "  output root    : ${OUTPUT_ROOT_BASE}"
echo "  job time       : ${JOB_TIME}"
if [ -n "${SBATCH_QOS}" ]; then
  echo "  sbatch qos     : ${SBATCH_QOS}"
else
  echo "  sbatch qos     : <default>"
fi
echo "  total jobs     : ${total_jobs}"

submit_job() {
  local tag="$1"
  local methods="$2"
  local output_root="${OUTPUT_ROOT_BASE}/${tag}"
  local job_name="${JOB_NAME_PREFIX}-${tag}"
  local export_spec="ALL,OUTPUT_ROOT,METHODS,R_BASE,BD_N_VALUES,M_VALUES,GROUP_LOCAL_EQUAL_M_VALUES,GROUP_LOCAL_PARAM_M_VALUES,BD_ROW_FACTOR,LORA_LR,MAX_LENGTH,WARMUP_RATIO,SCALING_MODE,TARGET_SUFFIXES"
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
    echo "        METHODS=${methods} OUTPUT_ROOT=${output_root}"
    return 0
  fi

  local job_id
  job_id="$(
    OUTPUT_ROOT="${output_root}" \
    METHODS="${methods}" \
    sbatch "${sbatch_args[@]}" "${BASE_SCRIPT}"
  )"

  echo "submitted ${job_id} -> ${job_name}"
}

if [ "${SUBMIT_VANILLA}" = "1" ]; then
  submit_job "vanilla" "vanilla_lora"
fi
if [ "${SUBMIT_BD}" = "1" ]; then
  submit_job "bd" "bd_lora"
fi
if [ "${SUBMIT_GROUP_LOCAL_PARAM}" = "1" ]; then
  submit_job "glparam" "group_local_param"
fi
