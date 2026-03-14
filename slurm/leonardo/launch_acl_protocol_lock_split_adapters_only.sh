#!/bin/bash
###############################################################################
# local-lora – Leonardo Booster
# Stage 1 (Protocol lock) split launcher, ADAPTERS ONLY.
#
# Purpose:
# - Re-run only adapter methods after fixing adapter-only training.
# - Skips head_only and full_ft jobs by default (those are valid and can be reused).
#
# Usage (same as the main launcher):
#   VARIANT=attn_only bash slurm/leonardo/launch_acl_protocol_lock_split_adapters_only.sh
#   VARIANT=attn_mlp  bash slurm/leonardo/launch_acl_protocol_lock_split_adapters_only.sh
#
# Overrides:
# - You can still override SUBMIT_* flags if you want to include baselines:
#     SUBMIT_HEAD_ONLY=1 SUBMIT_FULL_FT=1 bash ...
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default: submit adapters only.
export SUBMIT_HEAD_ONLY="${SUBMIT_HEAD_ONLY:-0}"
export SUBMIT_FULL_FT="${SUBMIT_FULL_FT:-0}"
export SUBMIT_ADAPTERS="${SUBMIT_ADAPTERS:-1}"

# Make job names/output roots easy to spot in the queue and filesystem.
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-aclp1-adapters}"

if [ -z "${OUTPUT_ROOT_BASE:-}" ]; then
  variant="${VARIANT:-attn_only}"
  export OUTPUT_ROOT_BASE="runs/leonardo_acl_protocol_lock_split_${variant}_adapters_only_$(date +%Y%m%d_%H%M%S)"
fi

bash "${SCRIPT_DIR}/launch_acl_protocol_lock_split.sh" "$@"
