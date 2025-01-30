#!/bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${1-2124}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${2-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# The base path for your distillEBWM repository
BASE_PATH="/work/hdd/bdta/aqian1/distillEBWM"

# Student ckpt (EBWM base)
EBWM_CKPT="${BASE_PATH}/checkpoints/ebwm-base/old_ebwm_ckpt.ckpt"

# Teacher checkpoint
TEACHER_CKPT_NAME="gpt2-base-sft"
TEACHER_CKPT="${BASE_PATH}/results/gpt2/sft/${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"

# Data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"

# Training hyperparams
BATCH_SIZE=2
LR=0.0005
GRAD_ACC=4
EVAL_BATCH_SIZE=4
MAX_LENGTH=512
SAVE_PATH="${BASE_PATH}/results/ebwm/train/distill_base_large"
SEED=10
EPOCHS=3
TOKENIZER="EleutherAI/gpt-neox-20b"

OPTS=""
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"

OPTS+=" --model-path ${TOKENIZER}"  # for tokenizer
OPTS+=" --model-ckpt ${EBWM_CKPT}"
# OPTS+=" --model_type ebwm"

OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio 1.0"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
OPTS+=" --type adaptive-sfkl"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

# We set total_iters or let code compute from epochs
# We run the finetune_ebwm.py entry point:
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_ebwm.py ${OPTS} $@"

echo ${CMD}
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}