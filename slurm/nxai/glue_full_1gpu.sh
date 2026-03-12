#!/bin/bash
#SBATCH --job-name=glue-full-1gpu
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00:00
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

python -m unittest discover -s tests -p "test_*.py" -q

# Optional: enable W&B by setting WANDB_API_KEY and exporting WANDB_MODE_ARG=online (or offline).
# Example:
#   sbatch --export=ALL,WANDB_MODE_ARG=online slurm/glue_full_1gpu.sh
WANDB_MODE_ARG="${WANDB_MODE_ARG:-online}"

python run_glue_suite.py \
  --tasks cola,sst2,mrpc,qqp,stsb,mnli,qnli,rte,wnli \
  --output_root "runs/glue_full_${SLURM_JOBID:-interactive}" \
  --wandb_mode "${WANDB_MODE_ARG}" \
  --wandb_entity hauzenberger \
  --wandb_project local-lora \
  --bf16 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-4

echo "Finished at $(date)"
