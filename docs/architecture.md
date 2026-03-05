# DocSP + InternVL3.5-8B: Full Architecture Pipeline

> 입력 이미지 한 장이 LLM 출력 텍스트가 되기까지의 전체 데이터 흐름.
> 모든 텐서 shape은 **single tile, batch_size=1** 기준.

---

## 0. 전체 파이프라인 Overview

```
Input Image (448×448×3)
    │
    ▼
┌──────────────────────────────────────┐
│         InternViT-300M               │
│   (Frozen Vision Encoder, 304M)      │
│                                      │
│   Patch Embed → 32×32 patches        │
│   + [CLS] token                      │
│   → 24 Transformer Blocks            │
│   → output: [1, 1025, 1024]          │
│   → remove CLS: [1, 1024, 1024]      │
│   → reshape: [1, 32, 32, 1024]       │
└──────────┬───────────────────────────┘
           │
     ┌─────┴──────┐
     │             │
     ▼             ▼
┌─────────┐  ┌───────────────────────────────────────┐
│  Pixel  │  │         DocSP Projector (38.7M)        │
│ Shuffle │  │                                        │
│ (0.5x)  │  │  [1, 1024, 32, 32]                    │
│         │  │       │                                │
│ 32×32   │  │   Channel Reduce → [1, 256, 32, 32]   │
│  ×1024  │  │       │                                │
│   ↓     │  │   Freq Decompose                       │
│ 16×16   │  │    ├─ High [1,256,32,32]               │
│  ×4096  │  │    └─ Low  [1,256,32,32]               │
│         │  │       │            │                    │
│         │  │   Struct SFE    Sem SFE                 │
│         │  │   [1,4,256]    [1,4,256]                │
│         │  │       │                                 │
│         │  │   DAG-FI (cross-attn)                   │
│         │  │   [1,4,256]                             │
│         │  │       │                                 │
│         │  │   cat(struct, attn) → [1,4,512]         │
│         │  │       │            │                    │
│         │  │   struct_proj   sem_proj                │
│         │  │   [1,4,4096]   [1,4,4096]               │
│         │  │       │            │                    │
│         │  │   cat → [1, 8, 4096]  (8 spatial tokens)│
│         │  └────────────┬────────────────────────────┘
│         │               │
│ flatten │               │
│ [1,256, │               │
│  4096]  │               │
│    │    │               │
│  mlp1   │               │
│ [1,256, │               │
│  4096]  │               │
└────┬────┘               │
     │                    │
     └───────cat──────────┘
             │
             ▼
    [1, 264, 4096]              ← 256 patch + 8 spatial tokens
    (Visual Tokens)
             │
             ▼
┌──────────────────────────────────────┐
│  Token Embedding Replacement         │
│                                      │
│  input_ids: [<s> User: <img>         │
│   <IMG_CONTEXT>×264 </img>           │
│   질문 텍스트 <|im_end|>             │
│   Assistant: ]                       │
│                                      │
│  text_embeds = Embed(input_ids)      │
│  text_embeds[IMG_CONTEXT 위치]       │
│      ← visual_tokens 대입            │
│                                      │
│  → input_embeds: [1, seq_len, 4096]  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│         Qwen3-8B LLM (7.6B)         │
│   (LoRA r=64 in Stage 2)            │
│                                      │
│   36 Transformer Decoder Blocks      │
│   hidden_size = 4096                 │
│   num_heads = 32                     │
│   vocab_size = 151,936               │
│                                      │
│   Causal Self-Attention              │
│   + RoPE Position Encoding           │
│   → logits: [1, seq_len, 151936]     │
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
Dynamic Resolution & Tiling (max_num_tiles=4)
    │
    ├─ 작은 이미지: 1 tile (448×448)
    ├─ 중간 이미지: 2 tiles (448×896 또는 896×448)
    └─ 큰 이미지: 최대 4 tiles (예: 896×896 → 2×2)
    │
    ▼
각 tile을 개별 처리: [num_tiles, 3, 448, 448]
    │
    ▼
ImageNet Normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
```

**Multi-tile인 경우**: 각 tile이 독립적으로 ViT + DocSP를 통과하여 tile당 264 tokens 생성.
4 tiles → 264 × 4 = 1,056 visual tokens.

---

## 2. InternViT-300M (Vision Encoder)

```
Input: [B, 3, 448, 448]   (B = num_tiles)
    │
    ▼
┌─ Patch Embedding ─────────────────────────────────┐
│  Conv2d(3, 1024, kernel=14, stride=14)            │
│  448 / 14 = 32 patches per axis                   │
│  → [B, 32×32, 1024] = [B, 1024, 1024]            │
│  + [CLS] token prepend                             │
│  + Learnable Position Embedding (1025, 1024)       │
│  → [B, 1025, 1024]                                │
└───────────────────────────────────────────────────┘
    │
    ▼
┌─ 24× Transformer Block ──────────────────────────┐
│                                                    │
│   x = x + Attention(LN(x))                        │
│   x = x + MLP(LN(x))                              │
│                                                    │
│   Attention:                                       │
│     heads = 16, head_dim = 64                      │
│     QKV proj: Linear(1024, 3072)                   │
│     → Flash Attention 2                            │
│     Out proj: Linear(1024, 1024)                   │
│                                                    │
│   MLP:                                             │
│     Linear(1024, 4096) → GELU → Linear(4096, 1024)│
│                                                    │
│   + DropPath (stochastic depth)                    │
│   + LayerScale                                     │
│                                                    │
│   ×24 blocks                                       │
└───────────────────────────────────────────────────┘
    │
    ▼
Output: [B, 1025, 1024]
    │
    ▼
Remove [CLS] token (index 0)
    │
    ▼
[B, 1024, 1024] → reshape → [B, 32, 32, 1024]

Parameters: 304M (frozen during training)
```

---

## 3. Dual-Path Projection

ViT 출력 `[B, 32, 32, 1024]`이 두 경로로 동시에 처리됨:

### Path A: Pixel Shuffle + MLP1 (기존 InternVL 경로)

```
[B, 32, 32, 1024]
    │
    ▼
┌─ Pixel Shuffle (downsample_ratio=0.5) ───────────┐
│                                                    │
│  32×32×1024 공간을 2×2 블록으로 묶어 채널로 전환:  │
│                                                    │
│  Step 1: [B, 32, 32, 1024]                        │
│       → [B, 32, 16, 2048]    (H 축 0.5배, C 2배)  │
│       → permute                                    │
│       → [B, 16, 32, 2048]                          │
│       → [B, 16, 16, 4096]   (W 축 0.5배, C 2배)   │
│       → permute                                    │
│       → [B, 16, 16, 4096]                          │
│                                                    │
│  결과: 공간 해상도 ½, 채널 4배                     │
│  인접 4개 패치의 정보가 하나로 합쳐짐               │
└───────────────────────────────────────────────────┘
    │
    ▼
[B, 16, 16, 4096] → flatten → [B, 256, 4096]
    │
    ▼
┌─ MLP1 Projector ─────────────────────────────────┐
│                                                    │
│  LayerNorm(4096)                                   │
│      ↓                                             │
│  Linear(4096, 4096) ─── ViT dim → LLM dim         │
│      ↓                                             │
│  GELU                                              │
│      ↓                                             │
│  Linear(4096, 4096) ─── LLM dim 유지               │
│                                                    │
│  Parameters: ~33.6M                                │
└───────────────────────────────────────────────────┘
    │
    ▼
Patch Tokens: [B, 256, 4096]
```

### Path B: DocSP Projector (신규 경로)

```
[B, 32, 32, 1024] → permute → [B, 1024, 32, 32]
    │
    ▼
┌─ 3.1 Channel Reduction ─────────────────────────┐
│                                                   │
│  Conv2d(1024, 256, 1×1)   1024 → 256 채널 축소    │
│  GroupNorm(32, 256)                               │
│  GELU                                             │
│                                                   │
│  → [B, 256, 32, 32]                              │
└───────────────────────────────────────────────────┘
    │
    ▼
┌─ 3.2 Frequency Decomposition ───────────────────┐
│                                                   │
│  Low-pass filter:                                 │
│    DWConv 5×5 (Gaussian init, σ=1.0)             │
│    Z_low = LP(x)              smooth features     │
│                                                   │
│  High-pass (residual):                            │
│    Z_high = x − Z_low         edge features       │
│                                                   │
│  Refinement (각 branch):                          │
│    DWConv 3×3 → GN → GELU → Conv 1×1             │
│    + residual connection                          │
│                                                   │
│  → Z_high: [B, 256, 32, 32]  (구조: 텍스트, 표,   │
│                                축, 경계선)         │
│  → Z_low:  [B, 256, 32, 32]  (의미: 색상 영역,    │
│                                배경, 레이아웃)     │
└──────────┬────────────────────┬───────────────────┘
           │                    │
           ▼                    ▼
┌─ 3.3a Struct SFE ────┐ ┌─ 3.3b Sem SFE ─────────┐
│  입력: Z_high         │ │  입력: Z_low            │
│  [B, 256, 32, 32]     │ │  [B, 256, 32, 32]      │
│                        │ │                         │
│  Scale 4:              │ │  Scale 4:               │
│   Pool(4)→DWConv(4)   │ │   Pool(4)→DWConv(4)    │
│   →GN→Conv(1)→GELU   │ │   →GN→Conv(1)→GELU    │
│   → 1 token [B,256]   │ │   → 1 token [B,256]    │
│                        │ │                         │
│  Scale 8:              │ │  Scale 8:               │
│   Pool(8)→DWConv(8)   │ │   Pool(8)→DWConv(8)    │
│   →GN→Conv(1)→GELU   │ │   →GN→Conv(1)→GELU    │
│   → 1 token [B,256]   │ │   → 1 token [B,256]    │
│                        │ │                         │
│  Scale 16:             │ │  Scale 16:              │
│   Pool(16)→DWConv(16) │ │   Pool(16)→DWConv(16)  │
│   →GN→Conv(1)→GELU   │ │   →GN→Conv(1)→GELU    │
│   → 1 token [B,256]   │ │   → 1 token [B,256]    │
│                        │ │                         │
│  Scale 32 (full grid): │ │  Scale 32 (full grid):  │
│   Pool(32)→DWConv(32) │ │   Pool(32)→DWConv(32)  │
│   →GN→Conv(1)→GELU   │ │   →GN→Conv(1)→GELU    │
│   → 1 token [B,256]   │ │   → 1 token [B,256]    │
│                        │ │                         │
│  stack → [B, 4, 256]  │ │  stack → [B, 4, 256]   │
│  (4 struct tokens)     │ │  (4 semantic tokens)    │
└────────┬───────────────┘ └────────┬────────────────┘
         │                          │
         ▼                          │
┌─ 3.4 DAG-FI (Cross-Attention) ─┐ │
│                                  │ │
│  Detail Features:                │ │
│   Z_high → Pool(8) → Conv 1×1   │ │
│   → [B, 64, 256] (K, V source)  │ │
│                                  │ │
│  Q = W_Q(LN(struct_tokens))     │ │
│      [B, 4, 256]                 │ │
│  K = W_K(LN(detail))            │ │
│      [B, 64, 256]               │ │
│  V = W_V(detail)                │ │
│      [B, 64, 256]               │ │
│                                  │ │
│  Multi-Head Attention (4 heads): │ │
│   head_dim = 64                  │ │
│   attn = softmax(QK^T/8)V       │ │
│   → merge heads → out_proj      │ │
│                                  │ │
│  → Z_attn: [B, 4, 256]          │ │
└────────┬─────────────────────────┘ │
         │                           │
         ▼                           │
┌─ 3.5 Concat & Project ────────────┤
│                                    │
│  struct_fused = cat(struct, attn)  │
│  [B, 4, 512]                      │
│       │                            │
│       ▼                            ▼
│  ┌─ struct_proj ──┐   ┌─ sem_proj ─────────┐
│  │ LN(512)        │   │ LN(256)            │
│  │ Linear(512,    │   │ Linear(256,        │
│  │        4096)   │   │        4096)       │
│  │ GELU           │   │ GELU               │
│  │ Linear(4096,   │   │ Linear(4096,       │
│  │        4096)   │   │        4096)       │
│  │                │   │                     │
│  │ → [B,4,4096]   │   │ → [B,4,4096]       │
│  └───────┬────────┘   └───────┬─────────────┘
│          │                    │
│          └───── cat ──────────┘
│                  │
│                  ▼
│         [B, 8, 4096]
│        (8 spatial tokens)
└─────────────────────────────────────────────────┘
```

### Path A + B 결합

```
Patch Tokens:   [B, 256, 4096]  ← MLP1 (기존 InternVL)
Spatial Tokens: [B,   8, 4096]  ← DocSP (신규)
                     │
                     ▼
              torch.cat(dim=1)
                     │
                     ▼
            [B, 264, 4096]
         (Visual Representation)
```

---

## 4. Token Embedding Replacement

```
Tokenizer 출력 (예시):
┌────────────────────────────────────────────────────┐
│ <s>  User:  <img>  <IMG_CTX>×264  </img>           │
│ "이 차트의 2023년 매출액은?"  <|im_end|>            │
│ Assistant:                                          │
│                                                     │
│ token_ids: [1, 2048, 45, 100000, 100000, ...(×264), │
│             100001, 8834, 923, ...]                  │
└────────────────────────────────────────────────────┘
                     │
                     ▼
┌─ Embedding Layer ────────────────────────────────┐
│                                                   │
│  text_embeds = Embed(token_ids)                   │
│  → [1, seq_len, 4096]                            │
│                                                   │
│  위치 찾기:                                       │
│  selected = (token_ids == IMG_CONTEXT_TOKEN_ID)   │
│                                                   │
│  교체:                                            │
│  text_embeds[selected] = visual_tokens            │
│                                                   │
│  즉, <IMG_CONTEXT> 위치 264개에                   │
│  visual_tokens [264, 4096]이 순서대로 대입됨       │
│                                                   │
│  결과: [1, seq_len, 4096]                         │
│  (텍스트 + 비전이 하나의 시퀀스로 결합)            │
└──────────────────────────────────────────────────┘

최종 input_embeds 구성:
┌────┬───────┬─────┬───────────────────────────┬──────┬──────────────┬──────────┐
│<s> │ User: │<img>│ v1 v2 ... v256 s1..s4 e1..e4 │</img>│ 질문 텍스트   │Assistant:│
│    │       │     │ ←patch tokens→ ←spatial→       │      │              │          │
│    │       │     │     264 visual tokens           │      │              │          │
└────┴───────┴─────┴───────────────────────────┴──────┴──────────────┴──────────┘
                        ↑ 이 부분이 DocSP의 출력
```

---

## 5. Qwen3-8B LLM (Language Model)

```
Input: input_embeds [1, seq_len, 4096]
    │
    ▼
┌─ 36× Transformer Decoder Block ────────────────┐
│                                                  │
│  ┌─ Causal Self-Attention ───────────────────┐  │
│  │                                            │  │
│  │  Q = W_Q(LN(x))  [1, seq, 4096]          │  │
│  │  K = W_K(LN(x))  [1, seq, 1024] (GQA)    │  │
│  │  V = W_V(LN(x))  [1, seq, 1024] (GQA)    │  │
│  │                                            │  │
│  │  32 query heads, 8 KV heads (GQA 4:1)     │  │
│  │  head_dim = 128                            │  │
│  │  + RoPE position encoding                  │  │
│  │  + Causal mask (미래 토큰 masking)         │  │
│  │  + Flash Attention 2                       │  │
│  │                                            │  │
│  │  O = W_O(Attn(Q, K, V))                   │  │
│  └────────────────────────────────────────────┘  │
│  x = x + O                                      │
│                                                  │
│  ┌─ Feed-Forward Network ────────────────────┐  │
│  │                                            │  │
│  │  gate = W_gate(LN(x))  [1, seq, 12288]   │  │
│  │  up   = W_up(LN(x))    [1, seq, 12288]   │  │
│  │  ffn  = SiLU(gate) ⊙ up                   │  │
│  │  out  = W_down(ffn)     [1, seq, 4096]    │  │
│  │                                            │  │
│  │  SwiGLU activation                         │  │
│  │  intermediate_size = 12,288                │  │
│  └────────────────────────────────────────────┘  │
│  x = x + out                                    │
│                                                  │
│  ×36 blocks                                      │
└──────────────────────────────────────────────────┘
    │
    ▼
┌─ Output Head ────────────────────────────────────┐
│                                                   │
│  hidden = LN(x)            [1, seq_len, 4096]    │
│  logits = W_lm(hidden)     [1, seq_len, 151936]  │
│                                                   │
│  vocab_size = 151,936                             │
└───────────────────────────────────────────────────┘
    │
    ▼
Autoregressive Decoding:
  token_i = argmax(logits[:, -1, :])  또는  sampling
  → 다음 토큰 생성 → 반복 (EOS까지)
```

---

## 6. Training: Loss Computation

```
                    input_embeds
                         │
                         ▼
                    Qwen3-8B LLM
                         │
                         ▼
               logits [1, seq_len, 151936]
                         │
                    shift by 1
                    (teacher forcing)
                         │
              ┌──────────┴──────────┐
              │                     │
    shift_logits[:, :-1]   shift_labels[:, 1:]
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
              CrossEntropyLoss(logits, labels)
                         │
                         ▼
                    scalar loss
                         │
                         ▼
                    loss.backward()

Note: 이미지/질문 부분의 label은 -100 (ignore_index)
      Assistant 응답 부분만 loss 계산 대상
```

---

## 7. Parameter Summary

```
┌─────────────────────────────────────────────────────┐
│ Component           │ Parameters │ Training Status   │
├─────────────────────┼────────────┼───────────────────┤
│ InternViT-300M      │    304M    │ Frozen (both)     │
│ MLP1 Projector      │   33.6M    │ Stage 1+2 train   │
│ DocSP Projector     │   38.7M    │ Stage 1+2 train   │
│ Qwen3-8B LLM       │  7,616M    │ Frozen → LoRA(S2) │
│ (LoRA r=64, S2 only)│   (574M)   │                   │
├─────────────────────┼────────────┼───────────────────┤
│ Total Model         │  7,993M    │                   │
│ Trainable (Stage 1) │   72.3M    │ mlp1 + docsp      │
│ Trainable (Stage 2) │  646.3M    │ mlp1+docsp+LoRA   │
└─────────────────────┴────────────┴───────────────────┘
```

---

## 8. Token Flow Summary (1 tile 기준)

```
Image 448×448
    → ViT: 1024 patches × 1024-dim
    → remove CLS: 1024 patches

    Path A (Patch):
        → Pixel Shuffle: 1024 → 256 patches × 4096-dim
        → MLP1: 256 × 4096-dim

    Path B (Spatial, DocSP):
        → Channel Reduce: 1024-dim → 256-dim
        → Freq Decompose: high + low
        → Struct SFE (4 scales): 4 tokens × 256-dim
        → Sem SFE (4 scales): 4 tokens × 256-dim
        → DAG-FI: struct 4 × 256 + attn 4 × 256 → 4 × 512
        → struct_proj: 4 × 4096
        → sem_proj: 4 × 4096
        → concat: 8 × 4096

    Combine: 256 + 8 = 264 tokens × 4096-dim

    → Replace <IMG_CONTEXT> in text sequence
    → LLM: autoregressive generation
    → Output: text tokens
```

---

## 9. DocSP vs LLaVA-SP 비교

```
                    LLaVA-SP                    DocSP (Ours)
                    ────────                    ────────────
ViT             │  CLIP-ViT-L/14 (24×24)   │  InternViT-300M (32×32)
Spatial Tokens  │  6 tokens                 │  8 tokens (4 struct + 4 sem)
SFE Scales      │  [4,8,12,16,20,24]        │  [4,8,16,32]
SFE Method      │  Pool + Conv2d(C→C/2)     │  Pool + DWConv + PWConv
Freq Decompose  │  없음                      │  Gaussian LP + residual HP
DFI Attention   │  Single-head, 512-dim     │  4-head, 256-dim + LN
DFI Output      │  cat(tokens, attn) → 1024 │  cat(tokens, attn) → 512
Normalization   │  없음                      │  GroupNorm + LayerNorm
Projection      │  GELU + Linear(512→4096)  │  LN + MLP(512→4096→4096)
Patch Projector │  MLP(1024→4096→4096)       │  PixelShuffle + MLP1
Total Tokens    │  576 + 6 = 582            │  256 + 8 = 264
LLM             │  Vicuna-7B (LoRA r=128)   │  Qwen3-8B (LoRA r=64)
```
