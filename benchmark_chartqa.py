"""ChartQA Benchmark Evaluation for InternVL + DocSP models."""
import sys
import os
import json
import argparse
import torch
from datetime import datetime
from transformers import AutoConfig, AutoModel, AutoTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from model.configuration_internvl_chat import InternVLChatConfig
from model.modeling_internvl_chat import InternVLChatModel

AutoConfig.register("internvl_chat", InternVLChatConfig)
AutoModel.register(InternVLChatConfig, InternVLChatModel)

CHARTQA_DIR = "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test"
CHARTQA_IMAGES = os.path.join(CHARTQA_DIR, "png")


def relaxed_accuracy(pred: str, label: str) -> bool:
    """ChartQA relaxed accuracy: exact match or within 5% for numeric answers."""
    pred = pred.strip().lower()
    label = label.strip().lower()
    if pred == label:
        return True
    try:
        pred_val = float(pred.replace(",", "").replace("%", ""))
        label_val = float(label.replace(",", "").replace("%", ""))
        if label_val == 0:
            return pred_val == 0
        return abs(pred_val - label_val) / abs(label_val) <= 0.05
    except (ValueError, ZeroDivisionError):
        return False


def evaluate(model, tokenizer, load_image_fn, test_file, split_name, max_samples=None):
    """Evaluate on one split of ChartQA."""
    with open(test_file) as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]

    correct = 0
    total = len(data)
    results = []
    generation_config = dict(max_new_tokens=128, do_sample=False)

    for i, sample in enumerate(data):
        img_path = os.path.join(CHARTQA_IMAGES, sample["imgname"])
        question = sample["query"]
        label = str(sample["label"])

        if not os.path.exists(img_path):
            print(f"  [{i+1}/{total}] Image not found: {img_path}, skipping")
            total -= 1
            continue

        try:
            pixel_values = load_image_fn(img_path, input_size=448, max_num=4).to(torch.bfloat16).cuda()
            prompt = f"{question}\nAnswer the question with a single word or number."
            pred = model.chat(tokenizer, pixel_values, prompt, generation_config)
        except Exception as e:
            print(f"  [{i+1}/{total}] Error: {e}, skipping")
            total -= 1
            continue

        is_correct = relaxed_accuracy(pred, label)
        if is_correct:
            correct += 1

        results.append({
            "image": sample["imgname"],
            "question": question,
            "label": label,
            "prediction": pred,
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0 or i == total - 1:
            acc = correct / max(len(results), 1) * 100
            print(f"  [{split_name}] {i+1}/{len(data)} - Running acc: {acc:.1f}%")

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n  [{split_name}] Final: {correct}/{total} = {accuracy:.2f}%")
    return accuracy, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples per split")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.model_path, "chartqa_benchmark.json")

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).eval().cuda()

    docsp_enabled = getattr(model, 'use_docsp', False)
    print(f"DocSP enabled: {docsp_enabled}")

    from sft_train import load_image

    print("\n=== ChartQA Human Split ===")
    human_acc, human_results = evaluate(
        model, tokenizer, load_image,
        os.path.join(CHARTQA_DIR, "test_human.json"),
        "Human", args.max_samples
    )

    print("\n=== ChartQA Augmented Split ===")
    aug_acc, aug_results = evaluate(
        model, tokenizer, load_image,
        os.path.join(CHARTQA_DIR, "test_augmented.json"),
        "Augmented", args.max_samples
    )

    avg_acc = (human_acc + aug_acc) / 2

    summary = {
        "model_path": args.model_path,
        "docsp_enabled": docsp_enabled,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "max_samples": args.max_samples,
        "results": {
            "human_accuracy": round(human_acc, 2),
            "augmented_accuracy": round(aug_acc, 2),
            "average_accuracy": round(avg_acc, 2),
        },
        "human_details": human_results,
        "augmented_details": aug_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"ChartQA Benchmark Results:")
    print(f"  Human:     {human_acc:.2f}%")
    print(f"  Augmented: {aug_acc:.2f}%")
    print(f"  Average:   {avg_acc:.2f}%")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
