#!/bin/bash

TORCH_COMPILE=1 \
JOB_TIME=1-00:00:00 \
VARIANT=attn_mlp \
WANDB_DIR=wandb_stage1_attn_mlp \

bash slurm/leonardo/launch_acl_protocol_lock_split.sh
