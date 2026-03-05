#!/bin/bash
# Stage 1: Projector Alignment (DocSP + MLP1)
# For 2x A100 40GB servers (Kicloud23/24/25)
# Usage: bash run_stage1_40gb.sh <server_id>
# e.g., bash run_stage1_40gb.sh 23

SERVER_ID=${1:-23}
OUTPUT_DIR="outputs/stage1_server${SERVER_ID}"
MASTER_PORT=$((29500 + SERVER_ID))

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export PATH=/home/juyeon/miniconda3/envs/chartinternvl/bin:/home/juyeon/miniconda3/bin:$PATH

cd /NetDisk/juyeon/DocSP

/home/juyeon/miniconda3/envs/chartinternvl/bin/torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} sft_train.py \
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
