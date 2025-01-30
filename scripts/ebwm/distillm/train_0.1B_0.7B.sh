#!/bin/bash

MASTER_ADDR=localhost
MASTER_PORT=29300
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1   # or however many GPUs you have

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

########################
# Path config
########################
BASE_PATH="/work/hdd/bdta/aqian1/distillEBWM"   # change to your path
STUDENT_CKPT="${BASE_PATH}/checkpoints/ebwm-base"    # the EBWM checkpoint or empty
TEACHER_CKPT="${BASE_PATH}/results/gpt2/sft/gpt2-large"
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/10M"

########################
# Hyperparams
########################
BATCH_SIZE=2
LR=0.0005
GRAD_ACC=1
EVAL_BATCH_SIZE=2
MAX_LENGTH=512
EPOCHS=5
SAVE_PATH="${BASE_PATH}/results/ebwm/train/distill_base_large"
SEED=42

########################
# Additional arguments
########################
OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
# The EBWM student model location
OPTS+=" --ebwm-model-path ${STUDENT_CKPT}/ebwm_student.ckpt"  
# or skip if you want to train from scratch
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ebwm-base"
OPTS+=" --teacher-ckpt-name gpt2-large"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num 1000"
# hp
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
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 5"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type of distill
OPTS+=" --type adaptive-sfkl"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"
# generation
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# Distillm Student generation
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_ebwm.py ${OPTS} $@"

echo ${CMD}
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
