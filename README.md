# DocSP: Document Structure-aware Spatial Perception for Vision-Language Models

A plug-and-play spatial projector module that enhances VLMs with document/chart-specific spatial understanding through frequency decomposition and multi-scale feature extraction. Built on **InternVL3.5-8B**.

---

## Overview

Standard VLM projectors (e.g., MLP) flatten 2D ViT features into 1D token sequences, losing critical spatial relationships. This is especially problematic for document and chart understanding, where **structural elements** (text boundaries, grid lines, axes) and **semantic regions** (color areas, backgrounds) convey distinct spatial information.

DocSP addresses this by:
1. **Frequency Decomposition**: Separating ViT features into high-frequency (structural) and low-frequency (semantic) components using a learnable Gaussian-initialized low-pass filter
2. **Multi-Scale Spatial Feature Extraction (SFE)**: Extracting spatial tokens at geometric scales [4, 8, 16, 32] from each frequency band
3. **Detail-Aware Guided Feature Integration (DAG-FI)**: Cross-attention where structural tokens attend to high-frequency detail features
4. **Dual-branch Projection**: Separate projections for structural and semantic tokens to preserve distinct information characteristics

The result: **8 additional spatial tokens** (4 structural + 4 semantic) appended to the standard 256 patch tokens per image tile, adding only ~3% token overhead.

### Architecture

```
Input Image (448x448)
    |
    v
InternViT-300M (Frozen, 304M params)
    |
    v
[B, 1024, 32, 32]  (ViT features)
    |
    +--- Path A: Pixel Shuffle (0.5x) + MLP1 --> [B, 256, 4096]  (patch tokens)
    |
    +--- Path B: DocSP Projector (38.7M params)
    |       |
    |       +- Channel Reduce: 1024 -> 256
    |       +- Freq Decompose: High (structural) + Low (semantic)
    |       +- Multi-Scale SFE: 4 tokens each at scales [4,8,16,32]
    |       +- DAG-FI: Cross-attention enrichment
    |       +- Dual Projection: -> [B, 8, 4096]  (spatial tokens)
    |
    v
Concatenate: [B, 264, 4096]  (256 patch + 8 spatial)
    |
    v
Qwen3-8B LLM (LoRA r=64 in Stage 2)
    |
    v
Output tokens
```

See [`docs/architecture.md`](docs/architecture.md) for the complete data flow with tensor shapes.

---

## Key Components

| Module | File | Description |
|--------|------|-------------|
| **DocSP Projector** | `model/docsp_projector.py` | Core module: FrequencyDecomposer, MultiScaleSFE, DAGFeatureIntegrator, DocSPProjector |
| **InternVL Chat Model** | `model/modeling_internvl_chat.py` | Modified InternVL3.5-8B with DocSP integration (dual-path forward) |
| **Training Script** | `sft_train.py` | 2-stage SFT trainer with multi-node DDP support |
| **LoRA Merge** | `merge_lora.py` | Merge LoRA adapters into base model weights |
| **Benchmark** | `benchmark_chartqa.py` | ChartQA evaluation with relaxed accuracy metric |

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1
- NVIDIA GPU (A100 40GB recommended, minimum 24GB)

### Setup

```bash
# Create conda environment
conda create -n chartinternvl python=3.10
conda activate chartinternvl

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install transformers==4.51.3
pip install flash-attn --no-build-isolation
pip install peft safetensors pillow packaging
pip install matplotlib seaborn  # for visualization

# Clone and setup
git clone https://github.com/yujuyeon0511/docsp.git
cd docsp
```

> **Note on transformers version**: Use 4.51.3 specifically. Versions >= 4.57 enforce `weights_only=True` in `torch.load()` which conflicts with checkpoint resume on PyTorch < 2.6. Versions < 4.49 do not support `Qwen3ForCausalLM`.

### Base Model

Download InternVL3.5-8B:
```bash
# From HuggingFace
huggingface-cli download OpenGVLab/InternVL3_5-8B --local-dir /path/to/InternVL3_5-8B
```

---

## Training

DocSP uses a 2-stage supervised fine-tuning (SFT) strategy following the LLaVA paradigm:

### Stage 1: Projector Alignment

Trains only the MLP projector + DocSP module while keeping ViT and LLM frozen.

```bash
# Single GPU
python sft_train.py \
    --model_path /path/to/InternVL3_5-8B \
    --output_dir outputs/stage1 \
    --datasets_conf datasets_stage1.conf \
    --use_docsp \
    --train_projector_only \
    --use_flash_attn --bf16 \
    --image_size 448 --max_num_tiles 4 --max_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-3 --warmup_ratio 0.03 \
    --num_train_epochs 1 --optim adafactor \
    --logging_steps 10 --save_steps 1000 \
    --gradient_checkpointing

# Multi-node DDP (5 nodes x 2 GPUs)
# On each node:
bash launch_multinode.sh <NODE_RANK>  # 0, 1, 2, 3, 4
```

**Key flags:**
- `--use_docsp`: Enable DocSP projector (adds 8 spatial tokens per tile)
- `--train_projector_only`: Freeze ViT + LLM, train only MLP1 + DocSP
- `--datasets_conf`: Tab-separated config file listing JSONL dataset paths

### Stage 2: LoRA Instruction Tuning

Adds LoRA adapters to the LLM while keeping projector weights from Stage 1.

```bash
# Single GPU
python sft_train.py \
    --model_path outputs/stage1 \
    --output_dir outputs/stage2 \
    --datasets_conf datasets_stage2.conf \
    --use_docsp \
    --freeze_vision \
    --keep_projector_weights \
    --use_lora --lora_rank 64 --lora_alpha 64 --lora_dropout 0.05 \
    --use_flash_attn --bf16 \
    --image_size 448 --max_num_tiles 4 --max_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 --warmup_ratio 0.03 \
    --num_train_epochs 1 --optim adafactor \
    --logging_steps 10 --save_steps 800 \
    --gradient_checkpointing \
    --resume_from_checkpoint auto

# Multi-node DDP
bash launch_stage2_multinode.sh <NODE_RANK>
```

**Key flags:**
- `--keep_projector_weights`: Load trained MLP1 + DocSP from Stage 1
- `--use_lora`: Enable LoRA on q/k/v/o/gate/up/down_proj (174.6M / 8.37B = 2.09% trainable)
- `--freeze_vision`: Keep ViT frozen

### Dataset Configuration

Dataset config files (`datasets_stage1.conf`, `datasets_stage2.conf`) use tab-separated format:

```
# JSONL_PATH<TAB>IMAGE_ROOT<TAB>MAX_SAMPLES
/path/to/dataset.jsonl	/path/to/images	50000
/path/to/dataset2.jsonl	__NONE__
```

- `IMAGE_ROOT`: Base path for relative image paths in JSONL. Use `__NONE__` for datasets with absolute paths.
- `MAX_SAMPLES`: Optional cap on samples from this dataset.

Each JSONL file contains entries in LLaVA format:
```json
{
  "image": "path/to/image.png",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe this chart."},
    {"from": "gpt", "value": "This bar chart shows..."}
  ]
}
```

### Multi-Node DDP Setup

For multi-node training, configure NCCL environment variables in the launch script:

```bash
export NCCL_SOCKET_IFNAME=ens3        # Network interface
export NCCL_IB_DISABLE=1              # Disable InfiniBand (if not available)
export NCCL_P2P_DISABLE=1             # Disable P2P (if cross-node)
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_BUFFSIZE=8388608
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
```

### Auto-GPU Memory Adjustment

The training script automatically adjusts batch size for GPU memory constraints:
- **80GB GPU**: Uses configured `per_device_train_batch_size`
- **40GB GPU**: Forces `batch_size=1`, doubles `gradient_accumulation_steps`
- Effective batch size is preserved regardless of GPU memory

---

## Inference

### Quick Inference Test

```python
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from model.configuration_internvl_chat import InternVLChatConfig
from model.modeling_internvl_chat import InternVLChatModel

AutoConfig.register("internvl_chat", InternVLChatConfig)
AutoModel.register(InternVLChatConfig, InternVLChatModel)

MODEL_PATH = "outputs/stage2"  # or Stage 1 path

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
model = InternVLChatModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
).eval().cuda()

print(f"DocSP enabled: {model.use_docsp}")

# Load and preprocess image
from sft_train import load_image
pixel_values = load_image("path/to/chart.png", input_size=448, max_num=4)
pixel_values = pixel_values.to(torch.bfloat16).cuda()

# Generate
generation_config = dict(max_new_tokens=512, do_sample=False)
question = "Describe this chart in detail."
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(response)
```

Or use the provided test scripts:
```bash
python test_inference.py           # Stage 1 model
python test_inference_stage2.py    # Stage 2 model
```

---

## Evaluation

### ChartQA Benchmark

Evaluates on ChartQA test set (1250 Human + 1250 Augmented = 2500 samples) using relaxed accuracy (exact match or within 5% for numeric answers):

```bash
python benchmark_chartqa.py \
    --model_path outputs/stage2 \
    --max_samples 100  # optional: limit samples for quick test
```

Results are saved to `outputs/stage2/chartqa_benchmark.json`.

### Run All Evaluations

```bash
bash run_stage2_eval.sh
```

This sequentially runs:
1. Inference test (10 samples)
2. Visualization (ViT attention, information flow, token redundancy)
3. ChartQA benchmark (full 2500 samples)

---

## Training Results

### Stage 1: Projector Alignment

| Metric | Value |
|--------|-------|
| Total Steps | 10,399 |
| Training Time | 36h 8m (5 nodes x 2 A100-40GB) |
| Final Loss | 1.234 |
| Dataset | ~1.5M samples (Korean Caption/OCR + English OCR/Caption) |
| Trainable Params | 72.3M (MLP1 33.6M + DocSP 38.7M) |

### Stage 2: LoRA Instruction Tuning

| Metric | Value |
|--------|-------|
| Total Steps | 3,840 |
| Training Time | ~21h total (5 nodes x 2 A100-40GB) |
| Final Loss | 0.4025 (from 1.4005) |
| Dataset | ~614K samples (Korean Reasoning/QA + English Reasoning/QA) |
| Trainable Params | 174.6M / 8.37B (2.09%) |
| LoRA Config | rank=64, alpha=64, targets: q/k/v/o/gate/up/down_proj |

#### Loss Trajectory (Stage 2)

| Step | Loss | LR | Note |
|------|------|-----|------|
| 10 | 1.4005 | 1.55e-5 | Start |
| 240 | 0.4912 | 1.93e-4 | Warmup complete |
| 3210 | 0.4038 | 3.39e-5 | Resume checkpoint |
| 3670 | 0.3827 | 9.18e-6 | Lowest loss |
| 3840 | 0.4025 | 5.37e-8 | Final |

### ChartQA Benchmark (Stage 2)

| Split | Accuracy |
|-------|----------|
| Human | 34.96% |
| Augmented | 37.36% |
| **Average** | **36.16%** |

> Note: Performance optimization is ongoing. The current results reflect the initial training with bilingual (Korean+English) data mix focused on document/chart understanding.

---

## Project Structure

```
docsp/
├── README.md                         # This file
├── sft_train.py                      # Main training script (2-stage SFT)
├── merge_lora.py                     # LoRA adapter merging utility
├── benchmark_chartqa.py              # ChartQA evaluation script
├── test_inference.py                 # Stage 1 inference test
├── test_inference_stage2.py          # Stage 2 inference test
├── run_stage2_eval.sh                # Full evaluation pipeline
│
├── model/
│   ├── __init__.py
│   ├── docsp_projector.py            # DocSP module (core contribution)
│   ├── modeling_internvl_chat.py     # Modified InternVL with DocSP integration
│   ├── modeling_intern_vit.py        # InternViT-300M vision encoder
│   ├── configuration_internvl_chat.py
│   ├── configuration_intern_vit.py
│   └── conversation.py              # Chat template utilities
│
├── launch_multinode.sh               # Stage 1 multi-node launcher
├── launch_stage2_multinode.sh        # Stage 2 multi-node launcher
├── launch_stage2_singlenode.sh       # Stage 2 single-node fallback
├── run_stage1.sh                     # Stage 1 single-node launcher
├── run_stage2.sh                     # Stage 2 single-node launcher
│
├── datasets_stage1.conf              # Stage 1 dataset config (~1.5M samples)
├── datasets_stage2.conf              # Stage 2 dataset config (~614K samples)
│
├── docs/
│   ├── architecture.md               # Detailed architecture & data flow
│   ├── references.md                 # Bibliography with BibTeX entries
│   └── llavasp_architecture.md       # LLaVA-SP baseline analysis
│
├── related_papers.md                 # Related work (2024-2026 top venues)
│
└── outputs/                          # Training outputs (not tracked in git)
    ├── stage1_multinode/             # Stage 1 checkpoints & model
    └── stage2_multinode/             # Stage 2 checkpoints & model
```

---

## DocSP Module Details

### FrequencyDecomposer (`model/docsp_projector.py`)

Learnable frequency decomposition via depthwise convolution:
- **Low-pass**: 5x5 depthwise conv initialized as Gaussian kernel (sigma=1.0)
- **High-pass**: Residual (input - low_pass output)
- Each branch refined by depthwise-separable residual block

```python
# Forward: [B, C, H, W] -> (high, low), each [B, C, H, W]
high, low = freq_decomposer(x)
# high: structural features (text, borders, axes, grid lines)
# low:  semantic features (color regions, backgrounds, layout)
```

### MultiScaleSFE

For each scale s in [4, 8, 16, 32]:
`AdaptiveAvgPool2d(s) -> DWConv(k=s) -> GroupNorm -> PWConv(1) -> GELU`

Produces one spatial token per scale: `[B, num_scales, C]`

### DAGFeatureIntegrator

Multi-head cross-attention (4 heads) where structural tokens (Q) attend to pooled high-frequency features (K, V):
- Pre-norm on Q, K inputs for training stability
- Output concatenated with original tokens: `[Z_struct || Z_attn]`

### vs. LLaVA-SP (Baseline)

| Aspect | LLaVA-SP | DocSP |
|--------|----------|-------|
| ViT | CLIP-ViT-L/14 (24x24) | InternViT-300M (32x32) |
| Spatial Tokens | 6 | 8 (4 struct + 4 sem) |
| SFE Scales | [4,8,12,16,20,24] | [4,8,16,32] |
| Freq Decomposition | None | Gaussian LP + residual HP |
| DFI Attention | Single-head | 4-head + LayerNorm |
| Normalization | None | GroupNorm + LayerNorm |
| Total Tokens | 576 + 6 = 582 | 256 + 8 = 264 |

---

## Related Work

See [`related_papers.md`](related_papers.md) for a curated list of 15 related papers from top-tier venues (CVPR, NeurIPS, ICLR, ECCV) spanning 2024-2026, organized by topic:

- Document/Chart Understanding VLMs
- Spatial Perception in VLMs
- Token Redundancy Reduction
- Multi-stage Training & VLM Architecture

See [`docs/references.md`](docs/references.md) for the complete bibliography with BibTeX entries.

---

## Troubleshooting

### Common Issues

**1. `torch.load` weights_only error during checkpoint resume**
```
_pickle.UnpicklingError: Weights only load failed
```
The training script includes a `torch.load` patch (lines 16-22 of `sft_train.py`) that forces `weights_only=False`. This is required because checkpoints contain numpy objects that are not in PyTorch's safe globals list.

**2. Corrupted image crash**
```
PIL.UnidentifiedImageError: cannot identify image file
```
The training script handles corrupted images in `__getitem__` with try-except, skipping to the next sample. If you encounter persistent crashes, check your dataset for corrupted image files.

**3. NCCL connection issues in multi-node training**
- Verify all nodes can SSH to each other
- Ensure `NCCL_SOCKET_IFNAME` matches your network interface (`ip addr` to check)
- Set `NCCL_IB_DISABLE=1` and `NCCL_P2P_DISABLE=1` if no InfiniBand/NVLink

**4. Out of memory on smaller GPUs**
The script auto-adjusts for 40GB GPUs (batch_size=1, doubled gradient accumulation). For GPUs < 40GB, additionally reduce `--max_num_tiles` to 2 and `--max_length` to 2048.

---

## License

This project builds upon [InternVL](https://github.com/OpenGVLab/InternVL) (MIT License) and [LLaVA-SP](https://github.com/lhaoran2932/LLaVA-SP).
