#!/bin/bash

set -euo pipefail

# Exports ensure the variables propagate into the launcher (and then into sbatch jobs via --export=ALL,...).
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export JOB_TIME="${JOB_TIME:-1-00:00:00}"
export VARIANT="${VARIANT:-attn_mlp}"
export WANDB_DIR="${WANDB_DIR:-wandb_stage1_attn_mlp}"

bash slurm/leonardo/launch_acl_protocol_lock_split.sh
