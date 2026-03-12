#!/bin/bash
#SBATCH --job-name=glue-group-local-sampled
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

echo "Starting job on $(date)"
echo "JobID        : ${SLURM_JOBID:-}"
echo "Node list    : ${SLURM_JOB_NODELIST:-}"
echo "GPUs visible : ${CUDA_VISIBLE_DEVICES:-}"
echo "Work dir     : ${SLURM_SUBMIT_DIR:-$PWD}"

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR:-$PWD}"

set +u
source ~/.bashrc
set -u
conda activate local-lora

export HF_HOME="${HF_HOME:-$PWD/runs/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# Hugging Face auth (meta-llama/* is gated). If you logged in via `huggingface-cli login`,
# the token is typically stored in ~/.cache/huggingface/token, but setting HF_HOME above can
# hide it. Prefer an explicit HF_TOKEN; otherwise, reuse the cached token if present.
if [ -z "${HF_TOKEN:-}" ]; then
  if [ -n "${HF_HOME:-}" ] && [ -r "${HF_HOME}/token" ]; then
    export HF_TOKEN="$(<"${HF_HOME}/token")"
    echo "Loaded HF_TOKEN from ${HF_HOME}/token"
  elif [ -n "${HOME:-}" ] && [ -r "${HOME}/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(<"${HOME}/.cache/huggingface/token")"
    echo "Loaded HF_TOKEN from ${HOME}/.cache/huggingface/token"
  elif [ -n "${HOME:-}" ] && [ -r "${HOME}/.huggingface/token" ]; then
    export HF_TOKEN="$(<"${HOME}/.huggingface/token")"
    echo "Loaded HF_TOKEN from ${HOME}/.huggingface/token"
  else
    echo "HF_TOKEN not set; gated Hugging Face repos (e.g. meta-llama/*) may fail to download."
  fi
fi

WANDB_MODE_ARG="${WANDB_MODE_ARG:-online}"

TASKS="${TASKS:-cola,sst2,mrpc,qqp,stsb,mnli,qnli,rte,wnli}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-5000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2000}"

# Reduce m-grid by default for quicker feedback; override if you want the full sweep.
M_VALUES="${M_VALUES:-16,4,1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/group_local_sampled_${SLURM_JOBID:-interactive}}"

python -m unittest discover -s tests -p "test_*.py" -q

python run_glue_suite.py \
  --tasks "${TASKS}" \
  --methods head_only,group_local_equal,group_local_param \
  --m_values "${M_VALUES}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --output_root "${OUTPUT_ROOT}" \
  --wandb_mode "${WANDB_MODE_ARG}" \
  --wandb_entity hauzenberger \
  --wandb_project local-lora \
  --bf16

echo "Finished at $(date)"
