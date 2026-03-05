# LLaVA-SP: Full Architecture Pipeline (Code-Level Analysis)

> LLaVA-SP 원본 코드 기반의 완전한 데이터 흐름 분석.
> **주요 2가지 변형**: Pooling SFE (논문 최종) / Cropping SFE + DFI.
> 모든 텐서 shape은 **single image (1 tile), batch_size=1** 기준.
> 코드 참조: `/NetDisk/juyeon/LLaVA-SP/`

---

## 0. 전체 파이프라인 Overview

```
Input Image (336×336×3)
    │
    ▼
┌──────────────────────────────────────┐
│      CLIP-ViT-L/14 (Vision Tower)   │
│   (Frozen, ~304M)                   │
│                                      │
│   Patch Embed → 24×24 patches       │
│   + [CLS] token                      │
│   → 24 Transformer Blocks           │
│   → output: [1, 577, 1024]          │
│   → remove CLS: [1, 576, 1024]      │
│   → reshape: [1, 24, 24, 1024]      │
│                                      │
│   Returns:                           │
│     local_features = [1, 24, 24, 1024]  (2D grid, SFE 입력)
│     patch_features = [1, 576, 1024]     (1D flat, MLP 입력)
└──────────┬───────────────────────────┘
           │
     ┌─────┴──────┐
     │             │
     ▼             ▼
┌─────────┐  ┌───────────────────────────────────────┐
│  MLP    │  │        Spatial Token Branch            │
│Projector│  │   (SFE: Spatial Feature Extractor)     │
│(mm_proj)│  │                                        │
│         │  │   Pooling SFE: 6 spatial tokens        │
│576 patch│  │        또는                              │
│ ×1024   │  │   Cropping SFE + DFI: 6 spatial tokens │
│   ↓     │  │                                        │
│ Linear  │  │   → conv_linear: GELU + Linear         │
│1024→5120│  │   → [1, 6, 4096]                       │
│ GELU    │  │                                        │
│ Linear  │  └────────────┬────────────────────────────┘
│5120→5120│               │
│         │               │
│[1, 576, │               │
│  5120]  │               │
└────┬────┘               │
     │                    │
     └───────cat──────────┘
             │
             ▼
    [1, 582, 5120]               ← 6 spatial + 576 patch tokens
    (Visual Tokens)              ← 주의: spatial이 앞에 위치!
             │
             ▼
┌──────────────────────────────────────┐
│  Token Embedding Replacement         │
│  <image> 위치에 visual tokens 대입   │
│  → input_embeds: [1, seq_len, 5120]  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│       Vicuna-7B / 13B LLM           │
│   (LoRA r=128 in Fine-tuning)       │
│   hidden_size = 5120 (13B)          │
│                                      │
│   Causal Self-Attention              │
│   → logits: [1, seq_len, vocab]     │
└──────────────┬───────────────────────┘
               │
               ▼
        Output Tokens
     (Autoregressive Decoding)
```

---

## 1. Input Preprocessing

```
Raw Image (any size)
    │
    ▼
LLaVA-1.5 AnyRes Processing:
    │
    ├─ 336×336 base image (항상 생성)
    └─ 고해상도 tiles (optional, image_grid_pinpoints 기준)
    │
    ▼
ImageNet Normalization (CLIP 표준)
    mean = [0.48145466, 0.4578275, 0.40821073]
    std  = [0.26862954, 0.26130258, 0.27577711]
    │
    ▼
[num_tiles, 3, 336, 336]
```

**코드 위치**: `CLIPImageProcessor` (HuggingFace CLIP standard)

---

## 2. CLIP-ViT-L/14 (Vision Encoder)

### 2.1 전체 흐름

```
Input: [B, 3, 336, 336]
    │
    ▼
┌─ Patch Embedding ─────────────────────────────────┐
│  Conv2d(3, 1024, kernel=14, stride=14)            │
│  336 / 14 = 24 patches per axis                   │
│  → [B, 24×24, 1024] = [B, 576, 1024]             │
│  + [CLS] token prepend                             │
│  + Learnable Position Embedding (577, 1024)        │
│  → [B, 577, 1024]                                 │
└───────────────────────────────────────────────────┘
    │
    ▼
┌─ 24× Transformer Block ──────────────────────────┐
│  Pre-LN Transformer (CLIP standard)               │
│                                                    │
│  Attention:                                        │
│    heads = 16, head_dim = 64                       │
│    dim = 1024                                      │
│                                                    │
│  MLP:                                              │
│    Linear(1024, 4096) → QuickGELU → Linear(4096, 1024) │
│                                                    │
│  ×24 blocks                                        │
└───────────────────────────────────────────────────┘
    │
    ▼
Output: hidden_states[-2]   (penultimate layer, select_layer=-2)
→ [B, 577, 1024]
```

**코드 위치**: `clip_encoder.py:121-132`
```python
# clip_encoder.py:129-130
image_forward_outs = self.vision_tower(
    images.to(device=self.device, dtype=self.dtype),
    output_hidden_states=True
)
image_features = self.feature_select(image_forward_outs)
```

### 2.2 Feature Select (핵심 분기점)

ViT 출력에서 **patch features**와 **local features (2D grid)** 를 동시에 추출.

```python
# clip_encoder.py:35-118
def feature_select(self, image_forward_outs):
    image_features = image_forward_outs.hidden_states[self.select_layer]  # [-2]
    cls = image_features[:, 0].unsqueeze(1)              # [CLS] token (미사용)

    # select_feature == 'patch' → CLS 제거
    image_features = image_features[:, 1:]               # [B, 576, 1024]
    patch_features = image_features                       # 1D flat → MLP projector용

    bsz = image_features.shape[0]
    l = int(image_features.shape[1] ** 0.5)               # l = 24
    dim = image_features.shape[2]                          # dim = 1024
    image_features = image_features.reshape(bsz, l, l, dim)  # [B, 24, 24, 1024]

    # a == '336_pooling' → 2D grid 그대로 반환
    out = image_features                                   # [B, 24, 24, 1024]

    return out, patch_features
    #      ↑ local_features (SFE용)    ↑ patch_features (MLP용)
```

**핵심**: ViT는 1D sequence로 처리하지만, 출력을 다시 2D grid `[B, 24, 24, 1024]`로 reshape하여 spatial 정보를 복원. 이것이 SFE의 입력이 됨.

---

## 3. MLP Projector (Patch Token 경로)

LLaVA-1.5의 표준 2-layer MLP projector.

```
patch_features: [B, 576, 1024]
    │
    ▼
┌─ mm_projector (mlp2x_gelu) ─────────────────────┐
│                                                    │
│  Linear(1024, 5120)  ← ViT dim → LLM dim         │
│      ↓                                             │
│  GELU                                              │
│      ↓                                             │
│  Linear(5120, 5120)  ← LLM dim 유지               │
│                                                    │
└───────────────────────────────────────────────────┘
    │
    ▼
Patch Tokens: [B, 576, 5120]
```

**코드 위치**: `builder.py:33-46`
```python
# builder.py:39-46  (mm_projector_type = 'mlp2x_gelu')
mlp_depth = 2  # from regex match
modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]  # Linear(1024, 5120)
for _ in range(1, mlp_depth):
    modules.append(nn.GELU())
    modules.append(nn.Linear(config.hidden_size, config.hidden_size))  # Linear(5120, 5120)
return nn.Sequential(*modules)
```

**호출**: `llava_arch.py:857` (Pooling SFE 경로)
```python
p = self.get_model().mm_projector(patch_features)  # [B, 576, 5120]
```

---

## 4. Spatial Token Branch — 변형 A: Pooling SFE (논문 최종 모델)

> `version = 'pooling'` → `a = 'conv336_pooling_sfe'`

### 4.1 모듈 정의

**6개 스케일의 AdaptiveAvgPool2d + Conv2d** 조합. 각 스케일에서 1개의 spatial token 생성.

```python
# llava_arch.py:95-114
# a == 'conv336_pooling_sfe'
self.conv_linear = nn.Sequential(
    nn.GELU(),
    nn.Linear(512, 4096, bias=False)
)

self.conv_4 = nn.Sequential(
    nn.AdaptiveAvgPool2d(4),           # [B, 1024, 24, 24] → [B, 1024, 4, 4]
    nn.Conv2d(1024, 512, 4, bias=False) # kernel=4, [B, 1024, 4, 4] → [B, 512, 1, 1]
)

self.conv_8 = nn.Sequential(
    nn.AdaptiveAvgPool2d(8),           # → [B, 1024, 8, 8]
    nn.Conv2d(1024, 512, 8, bias=False) # kernel=8, → [B, 512, 1, 1]
)

self.conv_12 = nn.Sequential(
    nn.AdaptiveAvgPool2d(12),          # → [B, 1024, 12, 12]
    nn.Conv2d(1024, 512, 12, bias=False) # kernel=12, → [B, 512, 1, 1]
)

self.conv_16 = nn.Sequential(
    nn.AdaptiveAvgPool2d(16),          # → [B, 1024, 16, 16]
    nn.Conv2d(1024, 512, 16, bias=False) # kernel=16, → [B, 512, 1, 1]
)

self.conv_20 = nn.Sequential(
    nn.AdaptiveAvgPool2d(20),          # → [B, 1024, 20, 20]
    nn.Conv2d(1024, 512, 20, bias=False) # kernel=20, → [B, 512, 1, 1]
)

self.conv_24 = nn.Conv2d(1024, 512, 24, bias=False)
# 스케일 24: Pool 없음 (원본 24×24 그대로 사용)
# kernel=24, [B, 1024, 24, 24] → [B, 512, 1, 1]
```

### 4.2 Forward 흐름 (SFE)

```
local_features: [B, 24, 24, 1024]   (= ViT 2D grid 출력)
    │
    ▼ permute → [B, 1024, 24, 24]   (채널 first로 변환)
    │
    ├─ conv_4:  Pool(4)  + Conv(k=4)  → [B, 512, 1, 1] → squeeze → [B, 1, 512]
    ├─ conv_8:  Pool(8)  + Conv(k=8)  → [B, 512, 1, 1] → squeeze → [B, 1, 512]
    ├─ conv_12: Pool(12) + Conv(k=12) → [B, 512, 1, 1] → squeeze → [B, 1, 512]
    ├─ conv_16: Pool(16) + Conv(k=16) → [B, 512, 1, 1] → squeeze → [B, 1, 512]
    ├─ conv_20: Pool(20) + Conv(k=20) → [B, 512, 1, 1] → squeeze → [B, 1, 512]
    └─ conv_24: Conv(k=24)           → [B, 512, 1, 1] → squeeze → [B, 1, 512]
    │
    ▼ cat(dim=1)
    │
    [B, 6, 512]   (6 spatial tokens, 각각 512-dim)
    │
    ▼ conv_linear: GELU → Linear(512, 4096)
    │
    [B, 6, 4096]  (6 spatial tokens, LLM dim으로 투영)
```

**코드 위치**: `llava_arch.py:856-867`
```python
# a == 'conv336_pooling_sfe'
p = self.get_model().mm_projector(patch_features)  # [B, 576, 5120]

res = []
# 동일한 local_features를 6개 다른 스케일로 처리
res.append(self.get_model().conv_4(local_features.permute(0, 3, 1, 2))   # [B, 1024, 24, 24]
           .squeeze(-1).permute(0, 2, 1))  # → [B, 512, 1, 1] → [B, 512, 1] → [B, 1, 512]
res.append(self.get_model().conv_8(local_features.permute(0, 3, 1, 2))
           .squeeze(-1).permute(0, 2, 1))
res.append(self.get_model().conv_12(local_features.permute(0, 3, 1, 2))
           .squeeze(-1).permute(0, 2, 1))
res.append(self.get_model().conv_16(local_features.permute(0, 3, 1, 2))
           .squeeze(-1).permute(0, 2, 1))
res.append(self.get_model().conv_20(local_features.permute(0, 3, 1, 2))
           .squeeze(-1).permute(0, 2, 1))
res.append(self.get_model().conv_24(local_features.permute(0, 3, 1, 2))
           .squeeze(-1).permute(0, 2, 1))

c = self.get_model().conv_linear(torch.cat(res, dim=1))  # [B, 6, 512] → [B, 6, 4096]
image_features = torch.cat([c, p], dim=1)                 # [B, 6+576, 5120]
# 주의: spatial tokens이 앞, patch tokens이 뒤
```

### 4.3 SFE 각 스케일이 포착하는 정보

```
┌─────────┬────────────────┬──────────────────────────────────────┐
│ 스케일   │ Pool → Conv    │ 포착하는 spatial 정보                │
├─────────┼────────────────┼──────────────────────────────────────┤
│ conv_4  │ 24→4 → k=4     │ 가장 넓은 영역 (6×6 패치 = ~84px)   │
│         │                │ 전체 이미지의 1/4 영역씩 요약         │
│         │                │ → 이미지 전체 레이아웃               │
├─────────┼────────────────┼──────────────────────────────────────┤
│ conv_8  │ 24→8 → k=8     │ 3×3 패치 영역 = ~42px               │
│         │                │ → 중간 규모 구조                     │
├─────────┼────────────────┼──────────────────────────────────────┤
│ conv_12 │ 24→12 → k=12   │ 2×2 패치 영역 = ~28px               │
│         │                │ → 텍스트 블록, 작은 객체             │
├─────────┼────────────────┼──────────────────────────────────────┤
│ conv_16 │ 24→16 → k=16   │ 1.5×1.5 패치 영역 = ~21px           │
│         │                │ → 세밀한 구조                        │
├─────────┼────────────────┼──────────────────────────────────────┤
│ conv_20 │ 24→20 → k=20   │ 1.2×1.2 패치 영역 = ~17px           │
│         │                │ → 미세 텍스처                        │
├─────────┼────────────────┼──────────────────────────────────────┤
│ conv_24 │ 없음 → k=24    │ 1×1 패치 (원본 해상도)              │
│         │                │ → 전체 그리드 단일 요약              │
│         │                │ 24×24 전체를 하나의 벡터로 압축       │
└─────────┴────────────────┴──────────────────────────────────────┘

핵심 설계 원리:
  Pool(s)로 해상도를 s×s로 조정한 뒤, Conv(k=s)로 s×s 전체를 1×1로 압축.
  → 각 스케일에서 정확히 1개의 spatial token 생성.
  → 6개 스케일 = 6개 spatial tokens.
```

---

## 5. Spatial Token Branch — 변형 B: Cropping SFE + DFI

> `version = 'cropping'` → `a = 'conv336_cropping_sfe_dfi'`

### 5.1 모듈 정의

**Cropping SFE**: Pool 없이 원본 24×24에서 직접 crop하여 conv 적용.
**DFI (Detail Feature Integrator)**: Cross-attention으로 spatial tokens를 detail features로 보강.

```python
# llava_arch.py:281-299
# a == 'conv336_cropping_sfe_dfi'

# Projection head (DFI 출력 포함하여 2배 dim)
self.conv_linear = nn.Sequential(
    nn.GELU(),
    nn.Linear(1024, 4096)       # 512(SFE) + 512(DFI) = 1024 → 4096
)

# DFI: Cross-Attention modules
self.query_projector = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512))
self.key_projector   = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512))
self.value_projector = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512))

# Cropping SFE: 원본 24×24에서 직접 convolution
self.conv_4  = nn.Conv2d(1024, 512, 4, bias=False)   # [B,1024,24,24] → [B,512,21,21]
self.conv_8  = nn.Conv2d(1024, 512, 8, bias=False)   # → [B,512,17,17]
self.conv_12 = nn.Conv2d(1024, 512, 12, bias=False)  # → [B,512,13,13]
self.conv_16 = nn.Conv2d(1024, 512, 16, bias=False)  # → [B,512,9,9]
self.conv_20 = nn.Conv2d(1024, 512, 20, bias=False)  # → [B,512,5,5]
self.conv_24 = nn.Conv2d(1024, 512, 24, bias=False)  # → [B,512,1,1]

# DFI용 detail feature extractor
self.conv_CM = nn.Conv2d(1024, 512, 16, stride=2, bias=False)
# [B,1024,24,24] → [B,512,5,5] → flatten → [B,25,512]  (K,V용 detail tokens)
```

### 5.2 Forward 흐름 (Cropping SFE + DFI)

```
local_features: [B, 24, 24, 1024]
    │
    ▼ (index [0]~[5] = center crop of different sizes)
    │
    ├─ local_features[0]: center 4×4 crop  → permute → conv_4  → [B,512,1,1] → [B,1,512]
    ├─ local_features[1]: center 8×8 crop  → permute → conv_8  → [B,512,1,1] → [B,1,512]
    ├─ local_features[2]: center 12×12 crop → permute → conv_12 → [B,512,1,1] → [B,1,512]
    ├─ local_features[3]: center 16×16 crop → permute → conv_16 → [B,512,1,1] → [B,1,512]
    ├─ local_features[4]: center 20×20 crop → permute → conv_20 → [B,512,1,1] → [B,1,512]
    └─ local_features[5]: full 24×24       → permute → conv_24 → [B,512,1,1] → [B,1,512]
    │
    ▼ cat(dim=1)
    │
    c = [B, 6, 512]   (6 SFE spatial tokens)
    │
    ▼
┌─ DFI (Detail Feature Integrator) ───────────────┐
│                                                   │
│  Detail Features (K, V):                         │
│   local_features[5] (full 24×24)                 │
│    → permute → conv_CM(k=16, stride=2)           │
│    → [B, 512, 5, 5]                              │
│    → reshape → [B, 25, 512]                      │
│                                                   │
│  Cross-Attention:                                 │
│   Q = query_projector(c)        [B, 6, 512]      │
│       = LN(c) → Linear(512,512)                  │
│                                                   │
│   K = key_projector(conv_fuse)  [B, 25, 512]     │
│       = LN(conv_fuse) → Linear(512,512)          │
│                                                   │
│   V = value_projector(conv_fuse) [B, 25, 512]    │
│       = LN(conv_fuse) → Linear(512,512)          │
│                                                   │
│   Single-head Attention:                          │
│     attn = Q @ K^T / √512       [B, 6, 25]      │
│     attn = softmax(attn)                          │
│     embed_feat = attn @ V        [B, 6, 512]     │
│                                                   │
│  Fusion (Eq. 4 of paper):                        │
│   embed_fuse = cat(c, embed_feat, dim=-1)        │
│   → [B, 6, 1024]  (SFE tokens + DFI attention)  │
│                                                   │
└──────────────────────────────────────────────────┘
    │
    ▼ conv_linear: GELU → Linear(1024, 4096)
    │
    [B, 6, 4096]  (6 spatial tokens, LLM dim)
    │
    ▼ cat with patch tokens
    │
    [B, 582, 5120]
```

**코드 위치**: `llava_arch.py:696-720`
```python
# a == 'conv336_cropping_sfe_dfi'

# SFE: 6개 스케일별 crop → conv → 1 token
res.append(self.get_model().conv_4(local_features[0].permute(0,3,1,2)).squeeze(-1).permute(0,2,1))
res.append(self.get_model().conv_8(local_features[1].permute(0,3,1,2)).squeeze(-1).permute(0,2,1))
res.append(self.get_model().conv_12(local_features[2].permute(0,3,1,2)).squeeze(-1).permute(0,2,1))
res.append(self.get_model().conv_16(local_features[3].permute(0,3,1,2)).squeeze(-1).permute(0,2,1))
res.append(self.get_model().conv_20(local_features[4].permute(0,3,1,2)).squeeze(-1).permute(0,2,1))
res.append(self.get_model().conv_24(local_features[5].permute(0,3,1,2)).squeeze(-1).permute(0,2,1))

c = torch.cat(res, dim=1).nan_to_num()  # [B, 6, 512]

# DFI: Detail feature extraction
conv_fuse = self.get_model().conv_CM(
    local_features[5].permute(0, 3, 1, 2)  # [B, 1024, 24, 24]
).reshape(c.shape[0], c.shape[2], -1)       # [B, 512, 25]
 .permute(0, 2, 1).nan_to_num()             # [B, 25, 512]

# DFI: Cross-Attention
embed_query = self.get_model().query_projector(c)          # [B, 6, 512]
embed_key   = self.get_model().key_projector(conv_fuse)    # [B, 25, 512]
embed_value = self.get_model().value_projector(conv_fuse)  # [B, 25, 512]

embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
embed_att = embed_att.nan_to_num()                         # [B, 6, 25]

embed_feat = (embed_att.softmax(-1) @ embed_value)         # [B, 6, 512]

# Concatenate SFE + DFI
embed_fuse = torch.cat([c, embed_feat], dim=-1)            # [B, 6, 1024]

# Final projection + combine with patch tokens
image_features = torch.cat([
    self.get_model().conv_linear(embed_fuse),               # [B, 6, 4096]
    self.get_model().mm_projector(patch_features)            # [B, 576, 5120]
], dim=1)                                                    # [B, 582, ?]
```

**주의**: Cropping SFE는 `conv_linear`가 `Linear(1024, 4096)`인데 `mm_projector`는 `Linear → 5120`. 논문에서 5120 (Vicuna-13B)을 사용하므로 최종 dim은 맞춤. Pooling SFE는 `Linear(512, 4096)`.

---

## 6. Cropping SFE의 Center Crop 생성 과정

Cropping 변형에서 `local_features[0]~[5]`가 어떻게 생성되는지:

```python
# clip_encoder.py:56-73
# a == '336_6_tokens'  (Cropping mode)
l = 24  # grid size
center_x = int(l / 2)  # = 12
center_y = int(l / 2)  # = 12
min_x = center_x - 1   # = 11
max_x = center_x + 1   # = 13
min_y = center_y - 1   # = 11
max_y = center_y + 1   # = 13

out = []
for i in range(0, int(l / 2), 2):  # i = 0, 2, 4, 6, 8, 10
    i_x = min_x - i - 1     # 10, 8, 6, 4, 2, 0
    j_x = max_x + i + 1     # 14, 16, 18, 20, 22, 24
    i_y = min_y - i - 1
    j_y = max_y + i + 1
    out.append(image_features[:, i_x:j_x, i_y:j_y, :])
```

```
crop 결과:
  out[0]: features[10:14, 10:14]  → 4×4   crop (center)
  out[1]: features[8:16,  8:16]   → 8×8   crop
  out[2]: features[6:18,  6:18]   → 12×12 crop
  out[3]: features[4:20,  4:20]   → 16×16 crop
  out[4]: features[2:22,  2:22]   → 20×20 crop
  out[5]: features[0:24,  0:24]   → 24×24 full grid

시각화:
┌──────────────────────────┐
│                          │  24×24 (out[5])
│   ┌──────────────────┐   │
│   │                  │   │  20×20 (out[4])
│   │   ┌──────────┐   │   │
│   │   │          │   │   │  16×16 (out[3])
│   │   │  ┌────┐  │   │   │
│   │   │  │ ■■ │  │   │   │  12×12 (out[2])
│   │   │  │ ■■ │  │   │   │   8×8 (out[1])
│   │   │  └────┘  │   │   │   4×4 (out[0])
│   │   │          │   │   │
│   │   └──────────┘   │   │
│   │                  │   │
│   └──────────────────┘   │
│                          │
└──────────────────────────┘

→ center-out nested crop: 중심에서 바깥으로 점점 넓어지는 영역
→ 각 crop에 대해 kernel=crop_size Conv를 적용하여 1 token으로 압축
```

---

## 7. Pooling SFE vs Cropping SFE 비교

```
                    Pooling SFE                    Cropping SFE
                    ───────────                    ────────────
입력            │  전체 24×24 grid (동일 입력)  │  center crop 6개 (다른 크기)
                │  local_features 한 번 사용    │  local_features[0]~[5] 사용
Pool 사용       │  ✅ AdaptiveAvgPool2d(s)      │  ❌ 없음 (직접 crop)
Conv kernel     │  Pool 후 s×s → k=s로 압축    │  crop 후 crop_size → k=crop로 압축
Detail 집중     │  uniform downsampling         │  center-biased (중심 강조)
DFI 사용        │  ❌ 없음                      │  ✅ Cross-Attention
파라미터 수     │  ~3.2M (conv) + 2.1M (linear) │  ~3.2M + 1.8M (DFI) + 2.1M
Spatial dim     │  512                          │  512 (SFE) + 512 (DFI) = 1024
Projection      │  GELU + Linear(512→4096)      │  GELU + Linear(1024→4096)
논문 기본 모델  │  ✅ (Table 1 기본)             │  ablation / variant
```

---

## 8. LLaVA-1.6 (AnyRes) + SP 통합

LLaVA-1.6의 spatial merge와 LLaVA-SP의 결합 방식:

```python
# llava_arch.py:967-1008
# mm_patch_merge_type.startswith('spatial')

for image_idx, image_feature in enumerate(image_features):
    if image_feature.shape[0] > 1:  # multi-tile
        # Spatial tokens 분리 (앞 6개)
        sp_image_feature = image_feature[:, 0:6, :]    # [num_tiles, 6, 5120]
        image_feature = image_feature[:, 6:, :]         # [num_tiles, 576, 5120]

        base_image_feature = image_feature[0]            # base tile: [576, 5120]
        image_feature = image_feature[1:]                # high-res tiles

        # LLaVA-1.6 standard: unpad + merge
        height = width = 24  # num_patches_per_side
        image_feature = image_feature.view(
            num_patch_height, num_patch_width, height, width, -1
        )
        # ... unpad processing ...

        image_feature = torch.cat(
            (base_image_feature, image_feature), dim=0
        )
    else:
        image_feature = image_feature[0]

    # SP tokens을 최종 image_feature 앞에 붙임
    image_feature = torch.cat(
        (sp_image_feature.flatten(0, 1),  # [num_tiles*6, 5120] all tiles의 SP tokens
         image_feature),                    # [576+..., 5120] merged patch tokens
        dim=0
    )
```

```
Multi-tile (예: 3 tiles) 시 토큰 구성:

┌──────────────────────────────────────────────────────┐
│ SP_tile1 (6) │ SP_tile2 (6) │ SP_tile3 (6)          │  ← 18 spatial tokens
│ base_patches (576) │ highres_patches (unpadded)       │  ← ~1152+ patch tokens
└──────────────────────────────────────────────────────┘

→ SP tokens이 항상 sequence 맨 앞에 위치
→ 모든 tile의 spatial tokens이 flatten되어 모임
```

---

## 9. Token Embedding Replacement

```python
# llava_arch.py:948-1098  prepare_inputs_labels_for_multimodal()

# 1. <image> 토큰 위치 찾기
image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)

# 2. 텍스트 토큰 임베딩
cur_input_embeds = self.get_model().embed_tokens(cur_input_ids_noim)

# 3. <image> 위치에 visual tokens 삽입
for i in range(num_images):
    cur_image_features = image_features[cur_image_idx]  # [582, 5120]
    cur_new_input_embeds.append(cur_image_features)
    cur_new_labels.append(
        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, ...)
    )
```

```
최종 input_embeds 구성 (single tile):
┌────┬───────┬───────────────────────────────────────────────────┬──────────────┬──────────┐
│BOS │ User: │ s1 s2 s3 s4 s5 s6 │ p1 p2 ... p576              │ 질문 텍스트   │Assistant:│
│    │       │ ← 6 spatial    →  │ ← 576 patch tokens →        │              │          │
│    │       │        582 visual tokens                          │              │          │
└────┴───────┴───────────────────────────────────────────────────┴──────────────┴──────────┘
```

---

## 10. LLM (Vicuna-7B / 13B)

```
Input: input_embeds [1, seq_len, 5120]  (Vicuna-13B 기준)
    │
    ▼
┌─ 40× Transformer Decoder Block (LLaMA architecture) ─┐
│                                                        │
│  ┌─ Multi-Head Attention ─────────────────────────┐  │
│  │  Q = W_Q(RMSNorm(x))   [1, seq, 5120]         │  │
│  │  K = W_K(RMSNorm(x))   [1, seq, 5120]         │  │
│  │  V = W_V(RMSNorm(x))   [1, seq, 5120]         │  │
│  │                                                 │  │
│  │  40 heads, head_dim = 128                       │  │
│  │  + RoPE position encoding                       │  │
│  │  + Causal mask                                  │  │
│  └─────────────────────────────────────────────────┘  │
│  x = x + Attn(x)                                     │
│                                                        │
│  ┌─ Feed-Forward Network ─────────────────────────┐  │
│  │  gate = W_gate(RMSNorm(x))  [1, seq, 13824]   │  │
│  │  up   = W_up(RMSNorm(x))    [1, seq, 13824]   │  │
│  │  ffn  = SiLU(gate) ⊙ up                        │  │
│  │  out  = W_down(ffn)          [1, seq, 5120]    │  │
│  │  SwiGLU activation                              │  │
│  └─────────────────────────────────────────────────┘  │
│  x = x + out                                         │
│                                                        │
│  ×40 blocks                                           │
└────────────────────────────────────────────────────────┘
    │
    ▼
  logits → Autoregressive Decoding

Training: LoRA (r=128) on LLM
  - target_modules: q_proj, v_proj (기본)
  - 학습 가능 파라미터: ~574M (LoRA만)
```

---

## 11. Training Pipeline

### Stage 1: Projector Alignment (Pretrain)

```
Frozen:  ViT (CLIP-ViT-L/14), LLM (Vicuna)
Trained: mm_projector (MLP), SFE convolutions, conv_linear
         (+ DFI if cropping variant)

Data:    LLaVA-558K (image-caption pairs)
Epochs:  1
LR:      1e-3
```

### Stage 2: Instruction Tuning (Fine-tune)

```
Frozen:  ViT (CLIP-ViT-L/14)
Trained: LLM (LoRA r=128), mm_projector, SFE, conv_linear, DFI

Data:    LLaVA-665K (instruction-following data)
Epochs:  1
LR:      2e-5
```

---

## 12. Parameter Summary

```
┌─────────────────────────────────────────────────────────────┐
│ Component              │ Parameters │ Training Status        │
├────────────────────────┼────────────┼────────────────────────┤
│ CLIP-ViT-L/14          │    304M    │ Frozen (both stages)   │
│ MLP Projector (mlp2x)  │   ~10.5M   │ Stage 1+2 train       │
│ SFE (6 convolutions)   │    ~3.2M   │ Stage 1+2 train       │
│ conv_linear             │    ~2.1M   │ Stage 1+2 train       │
│ DFI (cropping only)    │    ~1.8M   │ Stage 1+2 train       │
│ Vicuna-13B LLM         │  13,000M   │ Frozen → LoRA(S2)     │
│ (LoRA r=128, S2 only)  │   (~574M)  │                        │
├────────────────────────┼────────────┼────────────────────────┤
│ Pooling SFE variant    │            │                        │
│  Trainable (Stage 1)   │   ~15.8M   │ projector + SFE        │
│  Trainable (Stage 2)   │   ~590M    │ + LoRA                 │
├────────────────────────┼────────────┼────────────────────────┤
│ Cropping SFE+DFI       │            │                        │
│  Trainable (Stage 1)   │   ~17.6M   │ projector + SFE + DFI  │
│  Trainable (Stage 2)   │   ~592M    │ + LoRA                 │
└────────────────────────┴────────────┴────────────────────────┘
```

---

## 13. Token Flow Summary (1 tile 기준)

```
Image 336×336
    → CLIP-ViT-L/14: 576 patches × 1024-dim
    → remove CLS: 576 patches
    → reshape: [24, 24, 1024] (2D grid)

    Path A (Patch):
        → mm_projector: MLP(1024→5120→5120)
        → 576 × 5120-dim

    Path B (Spatial, SFE):
      [Pooling variant]
        → Pool(4)+Conv(4): 1 token × 512-dim
        → Pool(8)+Conv(8): 1 token × 512-dim
        → Pool(12)+Conv(12): 1 token × 512-dim
        → Pool(16)+Conv(16): 1 token × 512-dim
        → Pool(20)+Conv(20): 1 token × 512-dim
        → Conv(24): 1 token × 512-dim
        → cat: 6 tokens × 512-dim
        → conv_linear: GELU + Linear(512→4096)
        → 6 × 4096-dim

      [Cropping+DFI variant]
        → crop(4)+Conv(4): 1 token × 512-dim
        → crop(8)+Conv(8): 1 token × 512-dim
        → ... (6 scales)
        → cat: 6 tokens × 512-dim
        → DFI cross-attention with detail features
        → cat(SFE, DFI): 6 × 1024-dim
        → conv_linear: GELU + Linear(1024→4096)
        → 6 × 4096-dim

    Combine: 6 (spatial, front) + 576 (patch, back) = 582 tokens
    → Replace <image> in text sequence
    → LLM: autoregressive generation
    → Output: text tokens
```

---

## 14. LLaVA-SP 코드의 주요 특징 / 한계점

### 코드 레벨에서 관찰된 특징

1. **하드코딩된 version 선택**
```python
# llava_arch.py:47-51, llava_arch.py:607-612
version = 'pooling'              # 하드코딩!
if version == 'pooling':
    a = 'conv336_pooling_sfe'
elif version == 'cropping':
    a = 'conv336_cropping_sfe_dfi'
```
→ config가 아닌 코드 내 문자열로 variant 선택. 실험 전환 시 코드 수정 필요.

2. **Normalization 부재**
   - SFE의 Conv2d에 bias=False이고 BatchNorm/GroupNorm 없음
   - `nan_to_num()` 호출이 여러 곳에 있음 → 학습 불안정성 시사
   - DFI의 Q, K에만 LayerNorm 적용

3. **Channel dim 불일치**
   - Pooling SFE: `conv_linear = Linear(512→4096)` → spatial tokens = 4096-dim
   - MLP projector: `Linear(1024→5120→5120)` → patch tokens = 5120-dim
   - 두 경로의 출력 dim이 다름 (4096 vs 5120)
   - 실제 `torch.cat` 시 에러 없이 동작하려면 LLM hidden_size에 맞춰야 함

4. **DFI의 Single-head Attention**
   - `embed_query @ embed_key.T / √d` — single head, 512-dim
   - DocSP는 이를 4-head multi-head attention으로 개선

5. **Pooling SFE에서 동일 입력**
   - `local_features.permute(0,3,1,2)` — 동일한 2D grid가 6개 conv에 반복 입력
   - Cropping SFE는 `local_features[0]~[5]` — 다른 크기 crop이 입력
   - Pooling이 더 단순하지만 논문에서 더 좋은 결과를 보임

6. **다양한 실험 variant가 코드에 남아있음**
   - conv336, conv336_3_tokens, conv336_12_tokens, conv336_CM, conv_block, tfm336, ...
   - 총 10+ variant가 if-elif 체인으로 공존
   - 대규모 ablation의 흔적

---

## 15. LLaVA-SP vs DocSP 아키텍처 대조표

```
                     LLaVA-SP (Pooling SFE)              DocSP (Ours)
                     ──────────────────────              ────────────────
Base Model       │  CLIP-ViT-L/14 + Vicuna-13B       │  InternViT-300M + Qwen3-8B
ViT Grid Size    │  24×24 (336/14)                    │  32×32 (448/14)
ViT Hidden Dim   │  1024                              │  1024
LLM Hidden Dim   │  5120 (13B) / 4096 (7B)            │  4096

── SFE ──
Spatial Tokens   │  6 tokens                          │  8 tokens (4 struct + 4 sem)
SFE Scales       │  [4, 8, 12, 16, 20, 24]            │  [4, 8, 16, 32]
SFE Input        │  24×24 grid (단일)                  │  32×32 grid → high/low 분리
SFE Method       │  Pool(s) + Conv2d(1024→512, k=s)   │  Pool(s) + DWConv(k=s) + PWConv(1)
SFE Mid Dim      │  512                               │  256
SFE Activation   │  없음 (bias=False, no norm)         │  GroupNorm + GELU

── Frequency Decomposition ──
Freq Decompose   │  ❌ 없음                            │  ✅ Gaussian LP(5×5) + residual HP
Dual Branch      │  ❌ 단일 (content-agnostic)         │  ✅ structural SFE + semantic SFE

── DFI ──
Detail Integrator│  Single-head, 512-dim              │  4-head MHA, 256-dim
DFI Input (K,V)  │  Conv(1024→512, k=16, stride=2)   │  Pool(8) + Conv(1×1)
                 │  → [B, 25, 512]                    │  → [B, 64, 256]
DFI Pre-norm     │  LN on Q, K only                   │  LN on Q, K
DFI Output       │  cat(tokens, attn) → 1024          │  cat(tokens, attn) → 512
Attn Dropout     │  ❌ 없음                            │  ✅ configurable

── Projection ──
Projection       │  GELU + Linear(512→4096, no bias)  │  LN + MLP(512→4096→4096)
                 │  or GELU + Linear(1024→4096)       │  LN + MLP(256→4096→4096) (sem)
Normalization    │  ❌ 없음 (nan_to_num 사용)           │  ✅ GroupNorm, LayerNorm 전반

── Patch Path ──
Patch Projector  │  MLP2x(1024→5120→5120)             │  PixelShuffle(0.5) + MLP1(4096→4096)
Patch Tokens     │  576                               │  256

── Final ──
Total Tokens     │  576 + 6 = 582                     │  256 + 8 = 264
Token Order      │  [spatial, patch]                   │  [patch, spatial]
LLM Tuning       │  LoRA r=128                        │  LoRA r=64
Training         │  LLaVA 2-stage                     │  InternVL 2-stage
```

---

## 16. 코드 파일-구조 매핑 (LLaVA-SP)

```
/NetDisk/juyeon/LLaVA-SP/
├── llava/
│   ├── model/
│   │   ├── llava_arch.py              ← 핵심 파일
│   │   │   ├── LlavaMetaModel.__init__()       [L34-493]  SFE/DFI 모듈 정의
│   │   │   ├── LlavaMetaForCausalLM
│   │   │   │   ├── encode_images()              [L602-946] SFE/DFI forward
│   │   │   │   └── prepare_inputs_labels_for_multimodal() [L948-1098]
│   │   │   │       └── SP tokens 분리 & 앞에 붙이기 [L971-1006]
│   │   │
│   │   ├── multimodal_encoder/
│   │   │   └── clip_encoder.py        ← ViT + feature_select (2D grid 반환)
│   │   │       ├── CLIPVisionTower.__init__()    [L7-22]
│   │   │       ├── feature_select()              [L35-118] patch + local 분리
│   │   │       └── forward()                     [L120-132]
│   │   │
│   │   └── multimodal_projector/
│   │       └── builder.py             ← MLP projector (mlp2x_gelu)
│   │           └── build_vision_projector()      [L33-46]
│   │
│   ├── constants.py                   ← IMAGE_TOKEN_INDEX = -200
│   └── mm_utils.py                    ← get_anyres_image_grid_shape()
│
└── 2507.00505v3.pdf                   ← LLaVA-SP 논문
```
