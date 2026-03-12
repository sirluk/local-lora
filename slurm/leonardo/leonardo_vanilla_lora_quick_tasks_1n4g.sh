#!/bin/bash
###############################################################################
# local-lora – Leonardo Booster: 1 Node, 4×A100 (64GB)
# Runs 4 independent GLUE tasks in parallel (1 task per GPU by default).
###############################################################################
#SBATCH --job-name=glue-vanilla-quick-1n4g
#SBATCH --account=aifac_5l0_356
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
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
conda activate "${CONDA_ENV:-torch29}"

export HF_HOME="${HF_HOME:-$PWD/runs/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# Hugging Face auth (meta-llama/* is gated). Prefer an explicit HF_TOKEN.
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

# W&B is optional. Suggested defaults for HPC:
WANDB_MODE_ARG="${WANDB_MODE_ARG:-disabled}" # disabled|offline|online

# Default: 4 tasks so each GPU gets 1 GLUE task.
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
TASKS="${TASKS:-cola,sst2,mrpc,rte}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/leonardo_vanilla_quick_${SLURM_JOBID:-interactive}}"

python -m unittest discover -s tests -p "test_*.py" -q

# --- 3) Launch: 4 independent workers (1 per GPU) ---------------------------
srun --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail

    # Re-init env in the srun shell
    module purge
    module load profile/base
    module load gcc/12.2.0
    module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
    module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22

    set +u
    source "'"${BASHRC_PATH:-$HOME/.bashrc}"'"
    set -u
    conda activate "'"${CONDA_ENV:-torch29}"'"
    cd "'"${SLURM_SUBMIT_DIR:-$PWD}"'"

    export HF_HOME="'"${HF_HOME:-$PWD/runs/hf_home}"'"
    export TRANSFORMERS_CACHE="'"${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"'"
    export HF_DATASETS_CACHE="'"${HF_DATASETS_CACHE:-$HF_HOME/datasets}"'"
    export HF_TOKEN="'"${HF_TOKEN:-}"'"
    export TOKENIZERS_PARALLELISM=false

    # Avoid CPU oversubscription: divide CPU cores per task across tokenizer / torch threads
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
    export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$OMP_NUM_THREADS}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

    IFS=, read -r -a ALL_TASKS <<< "'"${TASKS}"'"
    RANK="${SLURM_PROCID}"
    WORLD="${SLURM_NTASKS}"

    # Round-robin assign tasks to ranks (works for any number of tasks)
    ASSIGNED=()
    for i in "${!ALL_TASKS[@]}"; do
      if [ $(( i % WORLD )) -eq "${RANK}" ]; then
        t="${ALL_TASKS[$i]}"
        t="${t// /}"
        [ -n "${t}" ] && ASSIGNED+=("${t}")
      fi
    done

    if [ "${#ASSIGNED[@]}" -eq 0 ]; then
      echo "Rank ${RANK}: no tasks assigned; exiting."
      exit 0
    fi

    TASK_STR="$(IFS=, ; echo "${ASSIGNED[*]}")"
    echo "Rank ${RANK}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} tasks=${TASK_STR}"

    mkdir -p "'"${OUTPUT_ROOT}"'"

    python run_glue_suite.py \
      --model_name "'"${MODEL_NAME}"'" \
      --tasks "${TASK_STR}" \
      --methods head_only,vanilla_lora \
      --output_root "'"${OUTPUT_ROOT}"'" \
      --results_csv "'"${OUTPUT_ROOT}"'/results_rank${RANK}.csv" \
      --wandb_mode "'"${WANDB_MODE_ARG}"'" \
      --wandb_entity hauzenberger \
      --wandb_project local-lora \
      --bf16
  '

# --- 4) Merge per-rank CSVs -------------------------------------------------
if ls "${OUTPUT_ROOT}"/results_rank*.csv >/dev/null 2>&1; then
  first="$(ls -1 "${OUTPUT_ROOT}"/results_rank*.csv | sort | head -n 1)"
  {
    head -n 1 "${first}"
    for f in $(ls -1 "${OUTPUT_ROOT}"/results_rank*.csv | sort); do
      tail -n +2 "${f}" || true
    done
  } > "${OUTPUT_ROOT}/results.csv"
  echo "Wrote merged results to ${OUTPUT_ROOT}/results.csv"
fi

echo "Finished at $(date)"
