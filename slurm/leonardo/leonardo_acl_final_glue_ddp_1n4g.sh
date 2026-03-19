#!/bin/bash
###############################################################################
# local-lora – Leonardo Booster: 1 Node, 4×A100 (64GB)
# Stage 2 (Final GLUE): 1B + 3B, seeds {0,1,2}, fixed recipe from protocol lock.
###############################################################################
#SBATCH --job-name=acl-final-glue-ddp-1n4g
#SBATCH --account=aifac_5l0_356
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
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
# Set this to a local snapshot path if running fully offline for 3B.
MODEL_NAME_3B="${MODEL_NAME_3B:-/leonardo_work/EUHPC_D31_132/models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"

if [ -n "${MODEL_NAMES:-}" ]; then
  MODEL_NAMES_ARG="${MODEL_NAMES}"
else
  if [ -n "${MODEL_NAME_3B}" ]; then
    MODEL_NAMES_ARG="${MODEL_NAME_1B},${MODEL_NAME_3B}"
  else
    echo "MODEL_NAME_3B not set; defaulting to 1B only. (Set MODEL_NAME_3B or MODEL_NAMES to include 3B.)"
    MODEL_NAMES_ARG="${MODEL_NAME_1B}"
  fi
fi

TASKS="${TASKS:-cola,sst2,mrpc,qqp,stsb,mnli,qnli,rte,wnli}"
SEEDS="${SEEDS:-0,1,2}"

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/leonardo_acl_final_glue_${SLURM_JOBID:-interactive}}"
METHODS="${METHODS:-all}"

PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-4}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

# Fill these from Stage 1 selection:
LORA_LR="${LORA_LR:-1e-4}"
FT_LR="${FT_LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
SCALING_MODE="${SCALING_MODE:-standard}" # standard|rs

# Chosen configs:
TARGET_SUFFIXES="${TARGET_SUFFIXES:-q_proj,k_proj,v_proj,o_proj}"
M_VALUES="${M_VALUES:-16,4}"        # backward-compatible fallback if per-method m values are unset
GROUP_LOCAL_EQUAL_M_VALUES="${GROUP_LOCAL_EQUAL_M_VALUES:-16}"
GROUP_LOCAL_PARAM_M_VALUES="${GROUP_LOCAL_PARAM_M_VALUES:-4}"
BD_N_VALUES="${BD_N_VALUES:-8}"     # choose 4 or 8 from Stage 1
BD_ROW_FACTOR="${BD_ROW_FACTOR:-block_a}"

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
  --model_names "${MODEL_NAMES_ARG}" \
  --tasks "${TASKS}" \
  --methods "${METHODS}" \
  --m_values "${M_VALUES}" \
  --group_local_equal_m_values "${GROUP_LOCAL_EQUAL_M_VALUES}" \
  --group_local_param_m_values "${GROUP_LOCAL_PARAM_M_VALUES}" \
  --bd_n_values "${BD_N_VALUES}" \
  --bd_row_factor "${BD_ROW_FACTOR}" \
  --seeds "${SEEDS}" \
  --learning_rates "${LORA_LR}" \
  --full_ft_learning_rates "${FT_LR}" \
  --max_lengths "${MAX_LENGTH}" \
  --warmup_ratios "${WARMUP_RATIO}" \
  --scaling_modes "${SCALING_MODE}" \
  --grouping_mode contiguous \
  --target_suffixes "${TARGET_SUFFIXES}" \
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
