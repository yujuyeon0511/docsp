#!/bin/bash
# Stage 1: Projector Alignment (DocSP + MLP1)
# Freeze ViT + LLM, train only mlp1 + DocSP projector
# 2x A100 80GB, DDP

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate chartinternvl

cd /NetDisk/juyeon/DocSP

torchrun --nproc_per_node=2 --master_port=29500 sft_train.py \
    --model_path /NetDisk/juyeon/models/InternVL3_5-8B \
    --output_dir outputs/stage1 \
    --datasets_conf datasets_stage1.conf \
    --use_docsp \
    --train_projector_only \
    --use_flash_attn \
    --bf16 \
    --image_size 448 \
    --max_num_tiles 4 \
    --max_length 2048 \
    --drop_long_samples \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
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
