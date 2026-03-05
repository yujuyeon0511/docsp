"""Quick inference test for Stage 1 trained model."""
import sys
import os
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from model.configuration_internvl_chat import InternVLChatConfig
from model.modeling_internvl_chat import InternVLChatModel

AutoConfig.register("internvl_chat", InternVLChatConfig)
AutoModel.register(InternVLChatConfig, InternVLChatModel)

MODEL_PATH = "/NetDisk/juyeon/DocSP/outputs/stage1_multinode"
TEST_IMAGES_DIR = "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test/png"

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).eval().cuda()
    print(f"Model loaded. DocSP enabled: {getattr(model, 'use_docsp', False)}")

    # Find test images
    images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith('.png')])[:5]
    questions = [
        "Describe this chart in detail.",
        "What is the title of this chart?",
        "What type of chart is this?",
        "What are the main trends shown?",
        "Summarize the key information.",
    ]

    generation_config = dict(max_new_tokens=256, do_sample=False)

    for i, (img_name, question) in enumerate(zip(images, questions)):
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        print(f"\n{'='*60}")
        print(f"[Sample {i+1}] Image: {img_name}")
        print(f"Question: {question}")

        image = Image.open(img_path).convert("RGB")
        from sft_train import load_image
        pixel_values = load_image(img_path, input_size=448, max_num=4).to(torch.bfloat16).cuda()

        answer = model.chat(tokenizer, pixel_values, question, generation_config)
        print(f"Answer: {answer}")

    print(f"\n{'='*60}")
    print("Inference test completed successfully!")

if __name__ == "__main__":
    main()
