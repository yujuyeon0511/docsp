#!/bin/bash
# Multi-node DDP training launcher for InternVL3.5-8B Stage 1
# 5 nodes x 2 GPUs = 10 GPUs total
#
# Usage: Run this script on each node with NODE_RANK argument
#   Node 23 (master): bash launch_multinode.sh 0
#   Node 24:          bash launch_multinode.sh 1
#   Node 25:          bash launch_multinode.sh 2
#   Node 17:          bash launch_multinode.sh 3
#   Node 18:          bash launch_multinode.sh 4

set -e

NODE_RANK=${1:?Usage: bash launch_multinode.sh <NODE_RANK: 0|1|2|3|4>}
NNODES=5
NPROC_PER_NODE=2
MASTER_ADDR=192.168.0.231
MASTER_PORT=29500

CONDA_ENV=/home/juyeon/miniconda3/envs/chartinternvl
WORK_DIR=/NetDisk/juyeon/DocSP
OUTPUT_DIR=${WORK_DIR}/outputs/stage1_multinode

export PATH=${CONDA_ENV}/bin:$PATH
export NCCL_SOCKET_IFNAME=ens3
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

cd ${WORK_DIR}

echo "=== Node ${NODE_RANK} starting ==="
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Nodes: ${NNODES}, GPUs per node: ${NPROC_PER_NODE}"

${CONDA_ENV}/bin/torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    sft_train.py \
    --model_path /NetDisk/juyeon/models/InternVL3_5-8B \
    --output_dir ${OUTPUT_DIR} \
    --datasets_conf datasets_stage1.conf \
    --use_docsp \
    --train_projector_only \
    --use_flash_attn \
    --bf16 \
    --image_size 448 \
    --max_num_tiles 4 \
    --max_length 2048 \
    --drop_long_samples \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --optim adafactor \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --gradient_checkpointing \
    --resume_from_checkpoint auto
