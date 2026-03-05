#!/bin/bash
# Stage 2: Instruction Tuning with LoRA - Single-node (2 GPUs)
# Fallback when multi-node is unavailable
# gradient_accumulation adjusted: 8→40 (auto-GPU will make 80) to match 10-GPU effective batch

set -e

NPROC_PER_NODE=2
CONDA_ENV=/home/juyeon/miniconda3/envs/chartinternvl
WORK_DIR=/NetDisk/juyeon/DocSP
OUTPUT_DIR=${WORK_DIR}/outputs/stage2_multinode

export PATH=${CONDA_ENV}/bin:$PATH
export NCCL_SOCKET_IFNAME=ens3
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_ASYNC_ERROR_HANDLING=1

cd ${WORK_DIR}

echo "=== Stage 2 - Single Node (2 GPUs) ==="
echo "Resuming from checkpoint in ${OUTPUT_DIR}"

${CONDA_ENV}/bin/torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
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
    --gradient_accumulation_steps 40 \
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
