#!/usr/bin/env python
"""Merge LoRA adapters from a DDP (non-DeepSpeed) checkpoint into the base model.

Usage:
    python merge_lora_ddp.py \
        --checkpoint_dir /path/to/stage1_alignment_v2/checkpoint-2400 \
        --model_path /NetDisk/j_son/Model/InternVL3_5-8B_v3 \
        --llm_model_name_or_path /NetDisk/j_son/models/HyperCLOVAX-SEED-Think-14B \
        --output_dir /path/to/merged_output \
        --lora_alpha 128 --lora_rank 64

What this script does:
    1. Loads model weights directly from DDP checkpoint safetensors
    2. Merges LoRA adapters: W_merged = W_base + (alpha/rank) * (B @ A)
    3. Renames keys (removes PEFT wrapper prefixes)
    4. Copies config, tokenizer, and model code files
    5. Saves the merged model as sharded safetensors
"""

import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from glob import glob

import torch
from safetensors.torch import load_file, save_file


def load_checkpoint_tensors(checkpoint_dir: str) -> dict:
    """Load all tensors from a DDP checkpoint directory."""
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    single_path = os.path.join(checkpoint_dir, "model.safetensors")

    all_tensors = {}

    if os.path.exists(index_path):
        # Sharded safetensors
        with open(index_path) as f:
            idx = json.load(f)
        shard_files = sorted(set(idx["weight_map"].values()))
        for sf in shard_files:
            print(f"  Loading shard: {sf}")
            all_tensors.update(load_file(os.path.join(checkpoint_dir, sf)))
    elif os.path.exists(single_path):
        # Single safetensors file
        print(f"  Loading model.safetensors")
        all_tensors = load_file(single_path)
    else:
        # Try .bin format
        bin_files = sorted(glob(os.path.join(checkpoint_dir, "pytorch_model*.bin")))
        if bin_files:
            for bf in bin_files:
                print(f"  Loading {os.path.basename(bf)}")
                all_tensors.update(torch.load(bf, map_location="cpu"))
        else:
            raise FileNotFoundError(
                f"No model weights found in {checkpoint_dir}. "
                f"Expected model.safetensors, model.safetensors.index.json, or pytorch_model*.bin"
            )

    print(f"  Total keys loaded: {len(all_tensors)}")
    return all_tensors


def merge_lora(all_tensors: dict, lora_alpha: int, lora_rank: int) -> dict:
    """Merge LoRA adapters into base weights."""
    # Group LoRA keys
    lora_groups = defaultdict(dict)
    for k in list(all_tensors.keys()):
        if ".lora_A.default.weight" in k:
            prefix = k.rsplit(".lora_A.default.weight", 1)[0]
            lora_groups[prefix]["lora_A"] = k
        elif ".lora_B.default.weight" in k:
            prefix = k.rsplit(".lora_B.default.weight", 1)[0]
            lora_groups[prefix]["lora_B"] = k

    if not lora_groups:
        print("No LoRA adapters found. Model may already be merged.")
        return all_tensors

    scaling = lora_alpha / lora_rank
    print(f"Found {len(lora_groups)} LoRA groups (alpha={lora_alpha}, rank={lora_rank}, scaling={scaling})")

    # Merge LoRA into base weights
    merged_count = 0
    for prefix, keys in lora_groups.items():
        base_key = prefix + ".base_layer.weight"
        if base_key not in all_tensors:
            print(f"  WARNING: base key not found: {base_key}")
            continue

        lora_A = all_tensors[keys["lora_A"]].float()
        lora_B = all_tensors[keys["lora_B"]].float()
        base_w = all_tensors[base_key].float()

        delta = scaling * (lora_B @ lora_A)
        all_tensors[base_key] = (base_w + delta).to(all_tensors[base_key].dtype)
        merged_count += 1

        # Remove LoRA keys
        del all_tensors[keys["lora_A"]]
        del all_tensors[keys["lora_B"]]

    print(f"Merged {merged_count}/{len(lora_groups)} LoRA adapters")

    # Rename: remove PEFT wrapper prefixes
    renamed = {}
    for k, v in all_tensors.items():
        new_k = k.replace(".base_model.model.", ".", 1)
        new_k = new_k.replace(".base_layer.", ".", 1)
        renamed[new_k] = v

    return renamed


def save_sharded(all_tensors: dict, output_dir: str):
    """Save as sharded safetensors (~5GB per shard)."""
    os.makedirs(output_dir, exist_ok=True)

    sorted_keys = sorted(all_tensors.keys())
    total_size = sum(all_tensors[k].numel() * all_tensors[k].element_size() for k in sorted_keys)
    num_shards = max(1, math.ceil(total_size / (5 * 1024**3)))
    target_shard_size = math.ceil(total_size / num_shards)

    shards = []
    current_shard = {}
    current_size = 0
    for k in sorted_keys:
        tensor_size = all_tensors[k].numel() * all_tensors[k].element_size()
        if current_size > 0 and current_size + tensor_size > target_shard_size:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[k] = all_tensors[k]
        current_size += tensor_size
    if current_shard:
        shards.append(current_shard)

    print(f"Saving {len(shards)} shards to {output_dir}...")
    weight_map = {}
    for i, shard in enumerate(shards):
        name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(output_dir, name))
        for k in shard:
            weight_map[k] = name
        print(f"  {name} ({len(shard)} keys)")

    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

    print(f"Total keys: {len(weight_map)}, Total size: {total_size / 1024**3:.2f} GB")


def copy_support_files(model_path: str, checkpoint_dir: str, output_dir: str):
    """Copy config, tokenizer, and model code files.

    Priority: checkpoint (has correct tokenizer with special tokens) > model_path (model code).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Config + tokenizer from checkpoint (includes special tokens added during training)
    checkpoint_files = [
        "config.json", "generation_config.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json",
        "tokenizer.json", "tokenizer.model",
        "chat_template.jinja",
    ]
    ckpt_copied = 0
    for fname in checkpoint_files:
        src = os.path.join(checkpoint_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            ckpt_copied += 1
    print(f"  Copied {ckpt_copied} config/tokenizer files from checkpoint")

    # Model code files from model_path (InternVL base)
    model_code_files = [
        "configuration_intern_vit.py",
        "configuration_internvl_chat.py",
        "modeling_intern_vit.py",
        "modeling_internvl_chat.py",
        "conversation.py",
    ]
    code_copied = 0
    for fname in model_code_files:
        src = os.path.join(model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            code_copied += 1
    print(f"  Copied {code_copied} model code files from model_path")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA from DDP checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to DDP checkpoint dir (e.g. stage1_alignment_v2/checkpoint-2400)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base InternVL model (for model code files)")
    parser.add_argument("--llm_model_name_or_path", type=str, default=None,
                        help="(unused, kept for backward compat)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save merged model")
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_rank", type=int, default=64)
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Output:     {args.output_dir}")
    print(f"LoRA:       alpha={args.lora_alpha}, rank={args.lora_rank}")
    print()

    # Step 1: Load checkpoint weights
    print("Step 1: Loading checkpoint weights...")
    all_tensors = load_checkpoint_tensors(args.checkpoint_dir)

    # Step 2: Merge LoRA
    print("\nStep 2: Merging LoRA adapters...")
    merged_tensors = merge_lora(all_tensors, args.lora_alpha, args.lora_rank)

    # Step 3: Save merged model
    print("\nStep 3: Saving merged model...")
    save_sharded(merged_tensors, args.output_dir)

    # Step 4: Copy support files
    print("\nStep 4: Copying support files...")
    copy_support_files(args.model_path, args.checkpoint_dir, args.output_dir)

    print(f"\nDone! Merged model saved to: {args.output_dir}")
    print(f"\nInference command:")
    print(f"  CUDA_VISIBLE_DEVICES=0 python inference_v3.py \\")
    print(f"    --model_path {args.output_dir} \\")
    print(f"    --image <image_path> \\")
    print(f"    --prompt \"이 이미지를 설명해줘.\" \\")
    print(f"    --max_new_tokens 512 --dtype bfloat16")


if __name__ == "__main__":
    main()
