#!/bin/bash
###############################################################################
# local-lora – Leonardo Booster
# Submit the grouping-modes ablation as one job per grouping mode.
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/leonardo_acl_ablation_grouping_modes_ddp_1n4g.sh}"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "Base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

GROUPING_MODES="${GROUPING_MODES:-contiguous,random,head_aligned}"
JOB_TIME="${JOB_TIME:-1-00:00:00}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-acla3}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-runs/leonardo_acl_grouping_split_$(date +%Y%m%d_%H%M%S)}"
SBATCH_QOS="${SBATCH_QOS:-}"
DRY_RUN="${DRY_RUN:-0}"

sanitize_token() {
  printf '%s' "$1" | tr -c '[:alnum:]._-' '_'
}

IFS=, read -r -a grouping_modes <<< "${GROUPING_MODES}"

echo "Submitting split grouping-modes ablation"
echo "  base script    : ${BASE_SCRIPT}"
echo "  output root    : ${OUTPUT_ROOT_BASE}"
echo "  job time       : ${JOB_TIME}"
if [ -n "${SBATCH_QOS}" ]; then
  echo "  sbatch qos     : ${SBATCH_QOS}"
else
  echo "  sbatch qos     : <default>"
fi
echo "  total jobs     : ${#grouping_modes[@]}"

submit_job() {
  local grouping_mode="$1"
  local tag="grouping_$(sanitize_token "${grouping_mode}")"
  local output_root="${OUTPUT_ROOT_BASE}/${tag}"
  local job_name="${JOB_NAME_PREFIX}-${tag}"
  local export_spec="ALL,OUTPUT_ROOT,GROUPING_MODES,PERM_SEED,LORA_LR,MAX_LENGTH,WARMUP_RATIO,SCALING_MODE,TARGET_SUFFIXES,M_VALUES,GROUP_LOCAL_EQUAL_M_VALUES,GROUP_LOCAL_PARAM_M_VALUES,BD_N_VALUES,BD_ROW_FACTOR,METHODS"
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
    echo "        GROUPING_MODES=${grouping_mode} OUTPUT_ROOT=${output_root}"
    return 0
  fi

  local job_id
  job_id="$(
    OUTPUT_ROOT="${output_root}" \
    GROUPING_MODES="${grouping_mode}" \
    sbatch "${sbatch_args[@]}" "${BASE_SCRIPT}"
  )"

  echo "submitted ${job_id} -> ${job_name}"
}

for grouping_mode in "${grouping_modes[@]}"; do
  submit_job "${grouping_mode}"
done
