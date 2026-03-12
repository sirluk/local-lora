#!/bin/bash
###############################################################################
# local-lora – Leonardo Booster: 1 Node, 4×A100 (64GB)
# Smoke test: run a tiny suite to validate adapters + logging on a single task.
###############################################################################
#SBATCH --job-name=acl-smoke-ddp-1n4g
#SBATCH --account=aifac_5l0_356
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --wait-all-nodes=1
#SBATCH --hint=nomultithread
###############################################################################

set -euo pipefail

echo "Starting job on $(date)"
echo "JobID        : ${SLURM_JOBID:-}"
echo "Node list    : ${SLURM_JOB_NODELIST:-}"
echo "Work dir     : ${SLURM_SUBMIT_DIR:-$PWD}"

# --- 1) Modules --------------------------------------------------------------
module purge
module load profile/base
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module list

# --- 2) Environment ---------------------------------------------------------
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR:-$PWD}"

set +u
source "${BASHRC_PATH:-$HOME/.bashrc}"
set -u
conda activate "${CONDA_ENV:-local-lora}"

export HF_HOME="${HF_HOME:-$PWD/runs/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# Leonardo compute nodes have no outbound internet. Use HF caches only.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

WANDB_MODE_ARG="${WANDB_MODE_ARG:-offline}" # disabled|offline|online

# Optional speed knob: torch.compile (default off)
TORCH_COMPILE="${TORCH_COMPILE:-0}" # set to 1 to enable
TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-}"
COMPILE_ARGS=()
if [ "${TORCH_COMPILE}" = "1" ]; then
  COMPILE_ARGS+=(--torch_compile)
  if [ -n "${TORCH_COMPILE_BACKEND}" ]; then
    COMPILE_ARGS+=(--torch_compile_backend "${TORCH_COMPILE_BACKEND}")
  fi
  if [ -n "${TORCH_COMPILE_MODE}" ]; then
    COMPILE_ARGS+=(--torch_compile_mode "${TORCH_COMPILE_MODE}")
  fi
fi

# Defaults (override via sbatch --export=ALL,VAR=...)
MODEL_NAME_1B="${MODEL_NAME_1B:-/leonardo_work/EUHPC_D31_132/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6}"
TASKS="${TASKS:-sst2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/leonardo_acl_smoke_${SLURM_JOBID:-interactive}}"
METHODS="${METHODS:-all}"

PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-4}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

# Keep it tiny
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"

RUN_TAG_DEFAULT="${SLURM_JOBID:-interactive}_$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-$RUN_TAG_DEFAULT}"

mkdir -p "${OUTPUT_ROOT}"

python -m unittest discover -s tests -p "test_*.py" -q

# --- 3) Launch: 4-GPU DDP ---------------------------------------------------
NPROC="${NPROC:-4}"
CPUS_PER_PROC=$(( ${SLURM_CPUS_PER_TASK:-1} / NPROC ))
if [ "${CPUS_PER_PROC}" -lt 1 ]; then
  CPUS_PER_PROC=1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPUS_PER_PROC}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

JOBID_NUM="${SLURM_JOBID:-0}"
if [[ "${JOBID_NUM}" =~ ^[0-9]+$ ]]; then
  MASTER_PORT_DEFAULT=$((29500 + (JOBID_NUM % 1000)))
else
  MASTER_PORT_DEFAULT=$((29500 + (RANDOM % 1000)))
fi
MASTER_PORT="${MASTER_PORT:-$MASTER_PORT_DEFAULT}"

torchrun --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" run_glue_suite.py \
  --run_tag "${RUN_TAG}" \
  --model_names "${MODEL_NAME_1B}" \
  --tasks "${TASKS}" \
  --methods "${METHODS}" \
  --seeds 0 \
  --learning_rates 1e-4 \
  --full_ft_learning_rates 2e-5 \
  --max_lengths 128 \
  --warmup_ratios 0.0 \
  --scaling_modes standard \
  --bd_n_values 8 \
  --bd_row_factor block_a \
  --grouping_mode contiguous \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --output_root "${OUTPUT_ROOT}" \
  --results_csv "${OUTPUT_ROOT}/results.csv" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BS}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --wandb_mode "${WANDB_MODE_ARG}" \
  --wandb_entity hauzenberger \
  --wandb_project local-lora \
  ${COMPILE_ARGS[@]:+"${COMPILE_ARGS[@]}"} \
  --bf16

python summarize_glue_results.py --results_csv "${OUTPUT_ROOT}/results.csv" || true

echo "Finished at $(date)"
