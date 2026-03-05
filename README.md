# DocSP

Document Structure-aware Spatial Perception module for InternVL3.5-8B.

Extends [LLaVA-SP](https://github.com/lhaoran2932/LLaVA-SP)'s spatial token approach with frequency decomposition to separate structural (text, axes, borders) and semantic (color regions, layout) information in document/chart images.

## Method

DocSP adds a lightweight projector that runs in parallel with InternVL's standard pixel-shuffle + MLP path. It takes raw ViT features (before pixel shuffle) and produces 8 spatial tokens (4 structural + 4 semantic) per image tile.

```
ViT features [B, 1024, 32, 32]
    |
    +-- Pixel Shuffle + MLP1 --> 256 patch tokens  (original InternVL path)
    |
    +-- DocSP Projector ------> 8 spatial tokens   (ours)
    |
    v
Concat: 264 tokens per tile --> LLM (Qwen3-8B)
```

Key components in `model/docsp_projector.py`:
- **FrequencyDecomposer**: Gaussian-initialized low-pass filter + residual high-pass, with refinement blocks
- **MultiScaleSFE**: Pool + depthwise conv at scales [4, 8, 16, 32] to extract one token per scale
- **DAGFeatureIntegrator**: 4-head cross-attention, structural tokens attend to high-freq detail features

Architecture details: [`docs/architecture.md`](docs/architecture.md)

## Setup

```bash
conda create -n docsp python=3.10
conda activate docsp

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.3 flash-attn --no-build-isolation
pip install peft safetensors pillow packaging
```

Base model: [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B)

> transformers must be 4.51.3. >= 4.57 breaks checkpoint resume (weights_only issue), < 4.49 lacks Qwen3 support.

## Training

2-stage SFT, following LLaVA convention:

**Stage 1 - Projector Alignment** (ViT & LLM frozen, train MLP1 + DocSP only):
```bash
# single GPU
python sft_train.py \
    --model_path /path/to/InternVL3_5-8B \
    --output_dir outputs/stage1 \
    --datasets_conf datasets_stage1.conf \
    --use_docsp --train_projector_only \
    --use_flash_attn --bf16 \
    --image_size 448 --max_num_tiles 4 --max_length 2048 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --learning_rate 1e-3 --warmup_ratio 0.03 \
    --num_train_epochs 1 --optim adafactor \
    --gradient_checkpointing

# multi-node (5 nodes x 2 GPUs)
bash launch_multinode.sh <NODE_RANK>
```

**Stage 2 - LoRA Instruction Tuning** (ViT frozen, LoRA on LLM):
```bash
python sft_train.py \
    --model_path outputs/stage1 \
    --output_dir outputs/stage2 \
    --datasets_conf datasets_stage2.conf \
    --use_docsp --freeze_vision --keep_projector_weights \
    --use_lora --lora_rank 64 --lora_alpha 64 --lora_dropout 0.05 \
    --use_flash_attn --bf16 \
    --image_size 448 --max_num_tiles 4 --max_length 4096 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 --warmup_ratio 0.03 \
    --num_train_epochs 1 --optim adafactor \
    --gradient_checkpointing

# multi-node
bash launch_stage2_multinode.sh <NODE_RANK>
```

### Dataset format

`datasets_stage1.conf` / `datasets_stage2.conf` are tab-separated:
```
/path/to/data.jsonl	/path/to/images	50000
/path/to/data2.jsonl	__NONE__
```

JSONL follows LLaVA format (see `sft_train.py` for details).

## Inference

```python
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from model.configuration_internvl_chat import InternVLChatConfig
from model.modeling_internvl_chat import InternVLChatModel

AutoConfig.register("internvl_chat", InternVLChatConfig)
AutoModel.register(InternVLChatConfig, InternVLChatModel)

model = InternVLChatModel.from_pretrained(
    "outputs/stage2", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, use_flash_attn=True,
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained("outputs/stage2", trust_remote_code=True, use_fast=False)

from sft_train import load_image
pixel_values = load_image("chart.png", input_size=448, max_num=4).to(torch.bfloat16).cuda()
response = model.chat(tokenizer, pixel_values, "Describe this chart.", dict(max_new_tokens=512, do_sample=False))
```

## Evaluation

### ChartQA

```bash
python benchmark_chartqa.py --model_path outputs/stage2
```

### VLMEvalKit (comprehensive)

Uses [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for standard VLM benchmarks.
Running DocSP-InternVL3.5-8B and InternVL2.5-8B baseline on the same 10 benchmarks:

```bash
# DocSP
python run.py --model DocSP-InternVL3_5-8B \
    --data MMVet GQA_TestDev_Balanced VizWiz ScienceQA_TEST TextVQA_VAL \
          POPE MME MMBench_DEV_EN SEEDBench_IMG LLaVABench \
    --work-dir ./outputs/docsp_benchmarks --verbose

# Baseline (InternVL2.5-8B)
python run.py --model InternVL2_5-8B \
    --data MMVet GQA_TestDev_Balanced VizWiz ScienceQA_TEST TextVQA_VAL \
          POPE MME MMBench_DEV_EN SEEDBench_IMG LLaVABench \
    --work-dir ./outputs/baseline_benchmarks --verbose
```

## Results

Trained on 10x A100-40GB (5 nodes), bilingual Korean/English data.

### Training

| | Stage 1 | Stage 2 |
|--|---------|---------|
| Data | ~1.5M (caption/OCR) | ~614K (reasoning/QA) |
| Steps | 10,399 | 3,840 |
| Final loss | 1.234 | 0.4025 |
| Trainable | 72.3M (projectors) | 174.6M (projectors + LoRA) |

### ChartQA (relaxed accuracy)

| Split | Acc |
|-------|-----|
| Human | 76.32% |
| Augmented | 93.60% |
| **Avg** | **84.96%** |

(InternVL2.5-8B baseline: ~83%)

### General VLM Benchmarks

> In progress — running on Kicloud23 (DocSP) and Kicloud24 (baseline).

| Benchmark | DocSP-InternVL3.5-8B | InternVL2.5-8B |
|-----------|---------------------|----------------|
| MM-Vet | - | - |
| GQA | - | - |
| VizWiz | - | - |
| SQA-IMG | - | - |
| TextVQA | - | - |
| POPE | - | - |
| MME | - | - |
| MMBench | - | - |
| SEED-IMG | - | - |
| LLaVA-Bench | - | - |

## Structure

```
model/
  docsp_projector.py          # DocSP module
  modeling_internvl_chat.py   # InternVL + DocSP integration
  modeling_intern_vit.py      # vision encoder
sft_train.py                  # training script
merge_lora.py                 # LoRA merge utility
benchmark_chartqa.py          # ChartQA evaluation
eval/
  vlmeval_wrapper.py          # VLMEvalKit model wrapper
  radar_chart.py              # benchmark radar chart visualization
launch_multinode.sh           # Stage 1 multi-node launcher
launch_stage2_multinode.sh    # Stage 2 multi-node launcher
datasets_stage1.conf          # Stage 1 data config
datasets_stage2.conf          # Stage 2 data config
docs/
  architecture.md             # full architecture with tensor shapes
  references.md               # bibliography (BibTeX)
```

## References

- [InternVL](https://github.com/OpenGVLab/InternVL) (base model)
- [LLaVA-SP](https://github.com/lhaoran2932/LLaVA-SP) (baseline, spatial token concept)
- See [`docs/references.md`](docs/references.md) for full bibliography
- See [`related_papers.md`](related_papers.md) for related work survey (2024-2026)
