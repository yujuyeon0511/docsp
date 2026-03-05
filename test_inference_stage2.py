"""Inference test for Stage 2 (LoRA merged) model. Saves results to JSON."""
import sys
import os
import json
import torch
from PIL import Image
from datetime import datetime
from transformers import AutoConfig, AutoModel, AutoTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from model.configuration_internvl_chat import InternVLChatConfig
from model.modeling_internvl_chat import InternVLChatModel

AutoConfig.register("internvl_chat", InternVLChatConfig)
AutoModel.register(InternVLChatConfig, InternVLChatModel)

MODEL_PATH = "/NetDisk/juyeon/DocSP/outputs/stage2_multinode"
TEST_IMAGES_DIR = "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test/png"
OUTPUT_PATH = "/NetDisk/juyeon/DocSP/outputs/stage2_multinode/inference_results.json"

def main():
    print("Loading Stage 2 model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).eval().cuda()

    docsp_enabled = getattr(model, 'use_docsp', False)
    print(f"Model loaded. DocSP enabled: {docsp_enabled}")

    from sft_train import load_image

    images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith('.png')])[:10]
    questions = [
        "Describe this chart in detail.",
        "What is the title of this chart?",
        "What type of chart is this?",
        "What are the main trends shown?",
        "Summarize the key information.",
        "What is the maximum value shown in this chart?",
        "How many categories are displayed?",
        "What colors are used in this chart?",
        "What is the source of this data?",
        "Compare the highest and lowest values.",
    ]

    generation_config = dict(max_new_tokens=256, do_sample=False)
    results = {
        "model_path": MODEL_PATH,
        "stage": "Stage 2 (LoRA Instruction Tuning)",
        "docsp_enabled": docsp_enabled,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": min(len(images), len(questions)),
        "samples": [],
    }

    for i, (img_name, question) in enumerate(zip(images, questions)):
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        print(f"\n{'='*60}")
        print(f"[Sample {i+1}] Image: {img_name}")
        print(f"Question: {question}")

        pixel_values = load_image(img_path, input_size=448, max_num=4).to(torch.bfloat16).cuda()
        answer = model.chat(tokenizer, pixel_values, question, generation_config)
        print(f"Answer: {answer}")

        results["samples"].append({
            "index": i + 1,
            "image": img_path,
            "question": question,
            "answer": answer,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"Total samples: {len(results['samples'])}")

if __name__ == "__main__":
    main()
