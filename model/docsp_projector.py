"""
DocSP: Document Structure-aware Spatial Perception Projector

Plug-and-play projector module that extracts spatial tokens from ViT features
using frequency decomposition and multi-scale aggregation.

Design principles:
- LLM-agnostic: Works with any Vision Encoder (dim C) + LLM (dim D) combination
- Loss-unchanged: Only adds spatial tokens; standard CE loss is used
- Extends LLaVA-SP's SFE/DFI with learnable Frequency Decomposition (FD-SFE)

Architecture overview:
    Z_vit ∈ R^{B×C×H×W}
        → Channel Reduction (1×1 conv, GroupNorm, GELU) → x ∈ R^{B×C'×H×W}
        → Frequency Decomposition:
            Z_low  = Refine(LP(x))     (semantic: regions, layout)
            Z_high = Refine(x − LP(x)) (structural: edges, text, grids)
        → Multi-Scale SFE:
            Z_struct = SFE(Z_high)  [B, N_s, C']
            Z_sem    = SFE(Z_low)   [B, N_e, C']
        → DAG-FI (Detail-Aware Guided Feature Integration):
            Z_attn = MHAttn(Q=Z_struct, K=V=Pool(Z_high))  [B, N_s, C']
            Z_fused = [Z_struct ‖ Z_attn]  [B, N_s, 2C']
        → Projection:
            H_struct = MLP(Z_fused)  [B, N_s, D]
            H_sem    = MLP(Z_sem)    [B, N_e, D]
        → Output: [H_struct ; H_sem] ∈ R^{B×(N_s+N_e)×D}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecomposer(nn.Module):
    """
    Learnable frequency decomposition via depthwise convolution.

    Decomposes input features into low-frequency (semantic layout) and
    high-frequency (structural detail) components.

    Low-pass: 5×5 depthwise conv initialized as Gaussian kernel (σ=1.0).
    High-pass: identity − low_pass (residual high-frequency).
    Each branch is refined by a depthwise-separable residual block.
    """

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        num_groups = min(32, channels)

        # Low-pass filter: depthwise conv initialized as Gaussian
        self.low_pass = nn.Conv2d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, groups=channels, bias=False,
        )

        # Refinement blocks with residual connections
        self.high_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.low_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

        self._init_low_pass_gaussian(kernel_size)

    def _init_low_pass_gaussian(self, kernel_size, sigma=1.0):
        """Initialize depthwise conv weights as Gaussian kernel for proper low-pass behavior."""
        ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / kernel.sum()
        with torch.no_grad():
            self.low_pass.weight.copy_(
                kernel.unsqueeze(0).unsqueeze(0).expand_as(self.low_pass.weight)
            )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            high: [B, C, H, W] high-frequency features (edges, text, grid lines)
            low:  [B, C, H, W] low-frequency features (regions, backgrounds)
        """
        low = self.low_pass(x)
        high = x - low

        high = self.high_refine(high) + high
        low = self.low_refine(low) + low
        return high, low


class MultiScaleSFE(nn.Module):
    """
    Multi-Scale Spatial Feature Extractor (FD-SFE).

    For each scale s_i, applies AdaptiveAvgPool2d(s_i) → depthwise Conv2d(k=s_i)
    → pointwise Conv2d(1) → GELU to produce a single spatial token [1×1×C].

    Following LLaVA-SP's proven pattern: Pool(s) + Conv(k=s) collapses the s×s
    spatial region into a single token, capturing scale-specific structure.
    """

    def __init__(self, channels, scales):
        super().__init__()
        num_groups = min(32, channels)
        self.extractors = nn.ModuleList()
        for s in scales:
            self.extractors.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(channels, channels, s, groups=channels, bias=False),
                    nn.GroupNorm(num_groups, channels),
                    nn.Conv2d(channels, channels, 1),
                    nn.GELU(),
                )
            )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            tokens: [B, num_scales, C]
        """
        tokens = []
        for ext in self.extractors:
            tok = ext(x)                              # [B, C, 1, 1]
            tokens.append(tok.flatten(2).squeeze(-1))  # [B, C]
        return torch.stack(tokens, dim=1)              # [B, num_scales, C]


class DAGFeatureIntegrator(nn.Module):
    """
    Detail-Aware Guided Feature Integration (DAG-FI).

    Multi-head cross-attention where structural spatial tokens (Q) attend to
    pooled high-frequency detail features (K, V).

    Follows LLaVA-SP Eq. 4 pattern:
        Z_attn = softmax(Q K^T / √d_k) V
    Returns the raw attention output (without residual), to be concatenated
    with the original tokens externally: Z_fused = [Z_struct ‖ Z_attn].

    Differences from LLaVA-SP:
    - Multi-head attention (h heads) instead of single-head
    - LayerNorm on Q/K inputs for training stability
    - Attention dropout for regularization
    """

    def __init__(self, dim, num_heads=4, detail_pool_size=8, attn_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Pre-norm on Q, K inputs (following LLaVA-SP's LN+Linear pattern)
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)

        # Detail feature extraction: pool + 1×1 conv
        self.detail_pool = nn.AdaptiveAvgPool2d(detail_pool_size)
        self.detail_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, tokens, high_freq_map):
        """
        Args:
            tokens: [B, N, D] structural spatial tokens (queries)
            high_freq_map: [B, D, H, W] high-frequency feature map
        Returns:
            attn_out: [B, N, D] detail-enhanced features (raw, no residual)
        """
        B, N, D = tokens.shape

        # Prepare detail features as K, V
        detail = self.detail_pool(high_freq_map)       # [B, D, p, p]
        detail = self.detail_conv(detail)
        detail = detail.flatten(2).transpose(1, 2)     # [B, p*p, D]

        # Q, K, V with pre-norm on Q, K
        q = self.q_proj(self.q_norm(tokens))            # [B, N, D]
        k = self.k_proj(self.k_norm(detail))            # [B, p*p, D]
        v = self.v_proj(detail)                         # [B, p*p, D]

        # Multi-head reshape: [B, h, seq, d_k]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, N, p*p]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v                                  # [B, h, N, d_k]

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return out  # Raw attention output; concat with tokens externally


class DocSPProjector(nn.Module):
    """
    DocSP: Document Structure-aware Spatial Perception Projector.

    Takes raw ViT features (before pixel shuffle) and produces spatial tokens
    that capture both structural (high-freq) and semantic (low-freq) information
    at multiple spatial scales.

    Interface: DocSPProjector(vision_dim, llm_dim) -> plug-and-play with any MLLM

    Key design choices (with justification):
    1. Frequency Decomposition: Documents have bimodal frequency distribution --
       high-freq (text, borders, axes) vs low-freq (color regions, backgrounds).
       A learnable Gaussian-initialized low-pass filter separates these components,
       enabling specialized processing for each.
    2. Multi-Scale SFE: Following LLaVA-SP, geometric scale progression
       [s, 2s, 4s, ..., G] captures both local detail and global layout.
    3. DAG-FI: Cross-attention enriches structural tokens with fine-grained
       detail, following LLaVA-SP's concat pattern: [Z_struct || Z_attn].
    4. Dual-branch projection: Structural tokens (512-dim from concat) and
       semantic tokens (256-dim) are projected to LLM dimension separately,
       preserving their distinct information characteristics.

    Args:
        vision_dim: Vision encoder hidden dimension (e.g., 1024 for InternViT)
        llm_dim: LLM hidden dimension (e.g., 4096 for Qwen3-8B)
        grid_size: Spatial grid size of ViT features before pixel shuffle
                   (e.g., 32 for 448px image / 14px patch)
        mid_dim: Internal bottleneck dimension for spatial processing
        num_struct_tokens: Number of structural spatial tokens (from high-freq)
        num_sem_tokens: Number of semantic spatial tokens (from low-freq)
        num_dfi_heads: Number of attention heads in DAG-FI
        attn_drop: Dropout rate for DAG-FI attention weights
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        llm_dim: int = 4096,
        grid_size: int = 32,
        mid_dim: int = 256,
        num_struct_tokens: int = 4,
        num_sem_tokens: int = 4,
        num_dfi_heads: int = 4,
        attn_drop: float = 0.0,
    ):
        super().__init__()

        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.grid_size = grid_size
        self.mid_dim = mid_dim
        self.num_struct_tokens = num_struct_tokens
        self.num_sem_tokens = num_sem_tokens
        self.num_spatial_tokens = num_struct_tokens + num_sem_tokens
        num_groups = min(32, mid_dim)

        # 1. Channel bottleneck: vision_dim -> mid_dim
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(vision_dim, mid_dim, 1, bias=False),
            nn.GroupNorm(num_groups, mid_dim),
            nn.GELU(),
        )

        # 2. Frequency decomposition
        self.freq_decompose = FrequencyDecomposer(mid_dim)

        # 3. Multi-scale spatial feature extraction
        scales = self._compute_scales(grid_size, num_struct_tokens)
        self.struct_sfe = MultiScaleSFE(mid_dim, scales)
        self.sem_sfe = MultiScaleSFE(mid_dim, scales)

        # 4. DAG-FI: enhance structural tokens with detail information
        self.dfi = DAGFeatureIntegrator(
            dim=mid_dim,
            num_heads=num_dfi_heads,
            detail_pool_size=max(8, grid_size // 4),
            attn_drop=attn_drop,
        )

        # 5. Projection to LLM dimension
        # Structural: concat(original, dfi_attn) -> 2*mid_dim -> llm_dim
        self.struct_proj = nn.Sequential(
            nn.LayerNorm(mid_dim * 2),
            nn.Linear(mid_dim * 2, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        # Semantic: mid_dim -> llm_dim
        self.sem_proj = nn.Sequential(
            nn.LayerNorm(mid_dim),
            nn.Linear(mid_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        self._init_weights()

    @staticmethod
    def _compute_scales(grid_size, n):
        """
        Compute n geometrically spaced scales ending at grid_size.

        Examples:
            grid=32, n=4 -> [4, 8, 16, 32]
            grid=24, n=4 -> [3, 6, 12, 24]
            grid=16, n=4 -> [2, 4, 8, 16]

        The largest scale always equals grid_size, ensuring the SFE
        captures full-image global features (matching LLaVA-SP's conv_24
        which covers the entire 24x24 CLIP grid).
        """
        scales = []
        for i in range(n):
            s = max(2, grid_size // (2 ** (n - 1 - i)))
            scales.append(min(s, grid_size))
        return scales

    def _init_weights(self):
        """
        Initialize weights following modern best practices.

        - Conv2d: Kaiming normal (fan_out) appropriate for networks with GELU
        - Linear: Truncated normal (std=0.02), following ViT/BERT convention
        - GroupNorm/LayerNorm: weight=1, bias=0
        - Low-pass filter: Already initialized as Gaussian in FrequencyDecomposer
        """
        for name, m in self.named_modules():
            # Skip low-pass filter (already Gaussian-initialized)
            if 'low_pass' in name:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, vit_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vit_features: [B, C, H, W] raw ViT features (before pixel shuffle).
                          C = vision_dim, H = W = grid_size.
        Returns:
            spatial_tokens: [B, N, D] where N = num_struct + num_sem, D = llm_dim
        """
        # 1. Channel reduction: [B, C, H, W] -> [B, C', H, W]
        x = self.channel_reduce(vit_features)

        # 2. Frequency decomposition: -> (high, low), each [B, C', H, W]
        high, low = self.freq_decompose(x)

        # 3. Multi-scale spatial extraction
        struct_tokens = self.struct_sfe(high)   # [B, N_s, C']
        sem_tokens = self.sem_sfe(low)          # [B, N_e, C']

        # 4. DAG-FI: cross-attend structural tokens to high-freq detail
        dfi_attn = self.dfi(struct_tokens, high)  # [B, N_s, C']
        # Concat original + attention (LLaVA-SP Eq.4 pattern, no redundant residual)
        struct_fused = torch.cat([struct_tokens, dfi_attn], dim=-1)  # [B, N_s, 2C']

        # 5. Project to LLM dimension
        h_struct = self.struct_proj(struct_fused)  # [B, N_s, D]
        h_sem = self.sem_proj(sem_tokens)          # [B, N_e, D]

        # Concatenate: structural first, then semantic
        spatial_tokens = torch.cat([h_struct, h_sem], dim=1)  # [B, N, D]

        return spatial_tokens
