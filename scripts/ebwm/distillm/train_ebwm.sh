#!/bin/bash

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

BASE_PATH="/work/hdd/bdta/aqian1/distillEBWM"

# Model config
TEACHER_PATH="gpt2-base"
TOKENIZER_PATH="gpt2"
TEACHER_CKPT_NAME="gpt2-base-sft"
TEACHER_CKPT="${BASE_PATH}/results/gpt2/sft/${TEACHER_CKPT_NAME}"

DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
SAVE_PATH="${BASE_PATH}/results/ebwm/train/distill_base_large"

# Training params
BATCH_SIZE=1
GRAD_ACC=4
EPOCHS=5
LR=5e-5
MCMC_STEPS=3

# Coefficients
RECON_COEF=0.5
DISTILL_COEF=0.5

# Runtime
WANDB_NAME="ebwm-base-large"

torchrun $DISTRIBUTED_ARGS train_ebwm.py \
    --model-path $TOKENIZER_PATH \
  --teacher-model-path $TEACHER_CKPT \
  --teacher-ckpt-name $TEACHER_CKPT_NAME \
  --data-dir $DATA_DIR \
  --batch-size $BATCH_SIZE \
  --gradient-accumulation-steps $GRAD_ACC \
  --epochs $EPOCHS \
  --lr $LR \
  --save $SAVE_PATH \
  --wandb \
  --run_name $WANDB_NAME \
  --deepspeed \
  --deepspeed_config ./configs/deepspeed/ds_config.json