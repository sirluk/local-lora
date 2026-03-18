#!/bin/bash
###############################################################################
# local-lora – Leonardo Booster
# Submit the ACL protocol-lock sweep as many smaller SLURM jobs.
#
# Each submitted job fixes one hyperparameter setting and reuses the existing
# attention-only or attention+mlp Stage 1 script underneath.
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANT="${VARIANT:-attn_only}" # attn_only | attn_mlp
case "${VARIANT}" in
  attn_only)
    BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/leonardo_acl_protocol_lock_attn_only_ddp_1n4g.sh}"
    TARGET_TAG="attn"
    ;;
  attn_mlp)
    BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/leonardo_acl_protocol_lock_attn_mlp_ddp_1n4g.sh}"
    TARGET_TAG="attnmlp"
    ;;
  *)
    echo "Unsupported VARIANT='${VARIANT}'. Use 'attn_only' or 'attn_mlp'." >&2
    exit 1
    ;;
esac

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "Base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

TASKS="${TASKS:-sst2,mrpc,rte,cola}"
LORA_LRS="${LORA_LRS:-1e-5,2e-5,5e-5,1e-4}"
FT_LRS="${FT_LRS:-5e-6,1e-5,2e-5}"
MAX_LENGTHS="${MAX_LENGTHS:-128,256}"
WARMUP_RATIOS="${WARMUP_RATIOS:-0.0,0.05}"
SCALING_MODES="${SCALING_MODES:-standard,rs}"

M_VALUES="${M_VALUES:-16,8,4,2,1}"
BD_N_VALUES="${BD_N_VALUES:-8,4}"
BD_ROW_FACTOR="${BD_ROW_FACTOR:-block_a}"
GROUPING_MODE="${GROUPING_MODE:-contiguous}"

JOB_TIME="${JOB_TIME:-1-00:00:00}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-aclp1}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-runs/leonardo_acl_protocol_lock_split_${VARIANT}_$(date +%Y%m%d_%H%M%S)}"
SBATCH_QOS="${SBATCH_QOS:-}"

SUBMIT_HEAD_ONLY="${SUBMIT_HEAD_ONLY:-1}"
SUBMIT_FULL_FT="${SUBMIT_FULL_FT:-1}"
SUBMIT_ADAPTERS="${SUBMIT_ADAPTERS:-1}"
DRY_RUN="${DRY_RUN:-0}"

sanitize_token() {
  printf '%s' "$1" | tr -c '[:alnum:]._-' '_'
}

count_enabled() {
  local flag="$1"
  if [ "${flag}" = "1" ]; then
    echo 1
  else
    echo 0
  fi
}

IFS=, read -r -a lora_lrs <<< "${LORA_LRS}"
IFS=, read -r -a ft_lrs <<< "${FT_LRS}"
IFS=, read -r -a max_lengths <<< "${MAX_LENGTHS}"
IFS=, read -r -a warmup_ratios <<< "${WARMUP_RATIOS}"
IFS=, read -r -a scaling_modes <<< "${SCALING_MODES}"

adapter_count=$(( ${#lora_lrs[@]} * ${#max_lengths[@]} * ${#warmup_ratios[@]} * ${#scaling_modes[@]} * $(count_enabled "${SUBMIT_ADAPTERS}") ))
head_count=$(( ${#lora_lrs[@]} * ${#max_lengths[@]} * ${#warmup_ratios[@]} * $(count_enabled "${SUBMIT_HEAD_ONLY}") ))
full_ft_count=$(( ${#ft_lrs[@]} * ${#max_lengths[@]} * ${#warmup_ratios[@]} * $(count_enabled "${SUBMIT_FULL_FT}") ))
total_jobs=$(( adapter_count + head_count + full_ft_count ))

echo "Submitting split protocol-lock sweep"
echo "  variant        : ${VARIANT}"
echo "  base script    : ${BASE_SCRIPT}"
echo "  tasks          : ${TASKS}"
echo "  output root    : ${OUTPUT_ROOT_BASE}"
echo "  job time       : ${JOB_TIME}"
if [ -n "${SBATCH_QOS}" ]; then
  echo "  sbatch qos     : ${SBATCH_QOS}"
else
  echo "  sbatch qos     : <default>"
fi
echo "  head_only jobs : ${head_count}"
echo "  full_ft jobs   : ${full_ft_count}"
echo "  adapter jobs   : ${adapter_count}"
echo "  total jobs     : ${total_jobs}"

submit_job() {
  local method_group="$1"
  local methods="$2"
  local lora_lr="$3"
  local ft_lr="$4"
  local max_length="$5"
  local warmup_ratio="$6"
  local scaling_mode="$7"

  local tag_parts=(
    "${TARGET_TAG}"
    "${method_group}"
    "len$(sanitize_token "${max_length}")"
    "wu$(sanitize_token "${warmup_ratio}")"
  )

  if [ -n "${lora_lr}" ]; then
    tag_parts+=("lora$(sanitize_token "${lora_lr}")")
  fi
  if [ -n "${ft_lr}" ]; then
    tag_parts+=("ft$(sanitize_token "${ft_lr}")")
  fi
  if [ -n "${scaling_mode}" ]; then
    tag_parts+=("sc$(sanitize_token "${scaling_mode}")")
  fi

  local tag
  tag="$(IFS=_; echo "${tag_parts[*]}")"
  local output_root="${OUTPUT_ROOT_BASE}/${tag}"
  local job_name="${JOB_NAME_PREFIX}-${tag}"
  local export_spec="ALL,TASKS,OUTPUT_ROOT,METHODS,LORA_LRS,FT_LRS,MAX_LENGTHS,WARMUP_RATIOS,SCALING_MODES,M_VALUES,GROUP_LOCAL_EQUAL_M_VALUES,GROUP_LOCAL_PARAM_M_VALUES,BD_N_VALUES,BD_ROW_FACTOR,GROUPING_MODE"
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
    echo "        METHODS=${methods} LORA_LRS=${lora_lr:-<unused>} FT_LRS=${ft_lr:-<unused>} MAX_LENGTHS=${max_length} WARMUP_RATIOS=${warmup_ratio} SCALING_MODES=${scaling_mode:-<unused>} OUTPUT_ROOT=${output_root}"
    return 0
  fi

  local job_id
  job_id="$(
    TASKS="${TASKS}" \
    OUTPUT_ROOT="${output_root}" \
    METHODS="${methods}" \
    LORA_LRS="${lora_lr:-1e-4}" \
    FT_LRS="${ft_lr:-2e-5}" \
    MAX_LENGTHS="${max_length}" \
    WARMUP_RATIOS="${warmup_ratio}" \
    SCALING_MODES="${scaling_mode:-standard}" \
    M_VALUES="${M_VALUES}" \
    BD_N_VALUES="${BD_N_VALUES}" \
    BD_ROW_FACTOR="${BD_ROW_FACTOR}" \
    GROUPING_MODE="${GROUPING_MODE}" \
    sbatch "${sbatch_args[@]}" "${BASE_SCRIPT}"
  )"

  echo "submitted ${job_id} -> ${job_name}"
}

if [ "${SUBMIT_HEAD_ONLY}" = "1" ]; then
  for lora_lr in "${lora_lrs[@]}"; do
    for max_length in "${max_lengths[@]}"; do
      for warmup_ratio in "${warmup_ratios[@]}"; do
        submit_job "head" "head_only" "${lora_lr}" "" "${max_length}" "${warmup_ratio}" ""
      done
    done
  done
fi

if [ "${SUBMIT_FULL_FT}" = "1" ]; then
  for ft_lr in "${ft_lrs[@]}"; do
    for max_length in "${max_lengths[@]}"; do
      for warmup_ratio in "${warmup_ratios[@]}"; do
        submit_job "fullft" "full_ft" "" "${ft_lr}" "${max_length}" "${warmup_ratio}" ""
      done
    done
  done
fi

if [ "${SUBMIT_ADAPTERS}" = "1" ]; then
  for lora_lr in "${lora_lrs[@]}"; do
    for max_length in "${max_lengths[@]}"; do
      for warmup_ratio in "${warmup_ratios[@]}"; do
        for scaling_mode in "${scaling_modes[@]}"; do
          submit_job "adapters" "vanilla_lora,bd_lora,group_local_equal,group_local_param" "${lora_lr}" "" "${max_length}" "${warmup_ratio}" "${scaling_mode}"
        done
      done
    done
  done
fi
