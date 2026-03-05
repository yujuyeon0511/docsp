#!/bin/bash
# Stage 2: Instruction Tuning with LoRA - Multi-node DDP
# 5 nodes x 2 GPUs = 10 GPUs total
#
# Usage:
#   Node 23 (master): bash launch_stage2_multinode.sh 0
#   Node 24:          bash launch_stage2_multinode.sh 1
#   Node 25:          bash launch_stage2_multinode.sh 2
#   Node 17:          bash launch_stage2_multinode.sh 3
#   Node 18:          bash launch_stage2_multinode.sh 4

set -e

NODE_RANK=${1:?Usage: bash launch_stage2_multinode.sh <NODE_RANK: 0|1|2|3|4>}
NNODES=5
NPROC_PER_NODE=2
MASTER_ADDR=192.168.0.231
MASTER_PORT=29500

CONDA_ENV=/home/juyeon/miniconda3/envs/chartinternvl
WORK_DIR=/NetDisk/juyeon/DocSP
OUTPUT_DIR=${WORK_DIR}/outputs/stage2_multinode

export PATH=${CONDA_ENV}/bin:$PATH
export NCCL_SOCKET_IFNAME=ens3
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_BUFFSIZE=8388608
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

cd ${WORK_DIR}

echo "=== Stage 2 - Node ${NODE_RANK} starting ==="
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Nodes: ${NNODES}, GPUs per node: ${NPROC_PER_NODE}"

${CONDA_ENV}/bin/torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    sft_train.py \
    --model_path outputs/stage1_multinode \
    --output_dir ${OUTPUT_DIR} \
    --datasets_conf datasets_stage2.conf \
    --use_docsp \
    --freeze_vision \
    --keep_projector_weights \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --use_flash_attn \
    --bf16 \
    --image_size 448 \
    --max_num_tiles 4 \
    --max_length 4096 \
    --drop_long_samples \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --optim adafactor \
    --logging_steps 10 \
    --save_steps 800 \
    --save_total_limit 3 \
    --gradient_checkpointing \
    --resume_from_checkpoint auto
