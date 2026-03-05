#!/bin/bash
# Stage 2: Instruction Tuning with LoRA
# Freeze ViT, keep trained projectors (mlp1 + DocSP), LoRA on LLM
# 2x A100 80GB, DDP

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate chartinternvl

cd /NetDisk/juyeon/DocSP

torchrun --nproc_per_node=2 --master_port=29500 sft_train.py \
    --model_path outputs/stage1 \
    --output_dir outputs/stage2 \
    --datasets_conf datasets_stage2.conf \
    --use_docsp \
    --freeze_vision \
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
