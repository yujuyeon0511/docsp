# Related Papers: InternVL + DocSP (Document Structure-aware Spatial Perception)

A curated collection of top-tier conference/journal papers related to Vision-Language Models for document/chart understanding, spatial perception, token efficiency, and multi-stage training.

---

## 2024 Papers

### 1. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks
- **Authors:** Zhe Chen, Jiannan Wu, Wenhai Wang, et al.
- **Venue:** CVPR 2024 (Oral)
- **Relevance:** The foundational VLM architecture that DocSP builds upon. Scales a vision foundation model to 6B parameters and progressively aligns it with an LLM using web-scale image-text data, achieving strong performance on document understanding benchmarks including DocVQA and ChartQA.
- **Citations:** ~900+ (Semantic Scholar, as of early 2026)

### 2. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities
- **Authors:** Boyuan Chen, Zhuo Xu, Sean Kirmani, et al.
- **Venue:** CVPR 2024
- **Relevance:** Directly relevant to spatial perception in VLMs. Proposes a data synthesis and pre-training mechanism that enables VLMs to perform spatial reasoning and metric distance estimation from 2D input images, addressing the spatial understanding gap in existing models.
- **Citations:** ~200+

### 3. SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models
- **Authors:** An-Chieh Cheng, Hongxu Yin, Yang Fu, et al.
- **Venue:** NeurIPS 2024
- **Relevance:** Enhances VLMs' spatial perception through a data curation pipeline using 3D scene graphs and a plug-in module for integrating depth information into the visual encoder, complementary to DocSP's structure-aware spatial encoding approach.
- **Citations:** ~100+

### 4. FastV: An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models
- **Authors:** Liang Chen, Haozhe Zhao, Tianyu Liu, et al.
- **Venue:** ECCV 2024 (Oral)
- **Relevance:** Pioneering work on visual token redundancy reduction in VLMs. Discovers that attention to visual tokens drops sharply after the second LLM layer and prunes redundant tokens, achieving 45% FLOPs reduction without performance loss -- directly related to DocSP's token efficiency goals.
- **Citations:** ~200+

### 5. CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs
- **Authors:** Zirui Wang, Mengzhou Xia, Luxi He, et al.
- **Venue:** NeurIPS 2024 (Datasets & Benchmarks Track)
- **Relevance:** A comprehensive chart understanding evaluation benchmark with 2,323 natural charts from scientific papers, including descriptive and reasoning questions. Reveals substantial gaps between proprietary and open-source models on chart comprehension, motivating DocSP's chart understanding improvements.
- **Citations:** ~80+

---

## 2025 Papers

### 6. DocLayLLM: An Efficient Multi-modal Extension of Large Language Models for Text-rich Document Understanding
- **Authors:** Wenhui Liao, Jiapeng Wang, Hongliang Li, et al.
- **Venue:** CVPR 2025
- **Relevance:** Directly related to document structure understanding. Integrates visual patch tokens and 2D positional tokens into LLMs for text-rich document understanding, and innovatively proposes CoT Pre-training and CoT Annealing techniques for layout-aware document reasoning.
- **Citations:** ~20+

### 7. Docopilot: Improving Multimodal Models for Document-Level Understanding
- **Authors:** Yuchen Duan, Zhe Chen, Yusong Hu, et al.
- **Venue:** CVPR 2025
- **Relevance:** Addresses document-level multimodal understanding with Doc-750K, a large-scale dataset with 758K QA pairs covering diverse document structures and cross-page dependencies. Achieves efficient document-level training through multimodal data packing and Ring Attention, relevant to DocSP's multi-page document handling.
- **Citations:** ~15+

### 8. Marten: Visual Question Answering with Mask Generation for Multi-modal Document Understanding
- **Authors:** Zining Wang, Tongkun Guan, Pei Fu, et al.
- **Venue:** CVPR 2025
- **Relevance:** Introduces a novel visual-language alignment method that casts document understanding as a joint VQA and mask generation task (VQAMask), with a large-scale dataset (MTMask6M). Directly relevant to structure-aware document understanding and spatial grounding in documents.
- **Citations:** ~10+

### 9. ATP-LLaVA: Adaptive Token Pruning for Large Vision Language Models
- **Authors:** Xubing Ye, Yukang Gan, Yixiao Ge, et al.
- **Venue:** CVPR 2025
- **Relevance:** Proposes instance-adaptive and layer-wise token pruning for VLMs, learning pruning thresholds with lightweight prediction heads. Reduces visual token count by 75% with only 1.9% performance degradation -- directly relevant to DocSP's token redundancy reduction strategy.
- **Citations:** ~15+

### 10. Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification
- **Authors:** Wenxuan Huang, Zijie Zhai, Yunhang Shen, et al.
- **Venue:** ICLR 2025
- **Relevance:** Proposes a dynamic sparsification framework that reduces vision context redundancy in both prefill and decoding stages using learnable binary mask predictors. Achieves ~80% image token reduction and ~50% computation savings, closely related to DocSP's efficiency goals for document-heavy inputs.
- **Citations:** ~30+

---

## 2026 Papers

### 11. FlashVID: Efficient Video Large Language Models via Training-free Tree-based Spatiotemporal Token Merging
- **Authors:** Ziyang Fan, Keyu Chen, Ruilong Xing, et al.
- **Venue:** ICLR 2026 (Oral)
- **Relevance:** Proposes attention-and-diversity-based token selection (ADTS) with tree-based spatiotemporal token merging, retaining only 10% of visual tokens while preserving 99.1% performance. The spatiotemporal token merging strategy is applicable to DocSP's multi-page document token reduction.
- **Citations:** N/A (newly published)

### 12. VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration
- **Authors:** Hanxun Yu, Wentong Li, Xuan Qu, et al.
- **Venue:** ICLR 2026
- **Relevance:** Proposes a unified training-free framework integrating Dominant Vision Token Selection (DVTS) for essential token preservation and Text-Guided Vision Complement (TGVC) for context-aware token merging. The text-guided approach is particularly relevant to DocSP's structure-aware token selection.
- **Citations:** N/A (newly published)

### 13. HiDivDrop: Vision Token Reduction in MLLMs via Late Injection and Differentiable Top-K
- **Authors:** (Under review -- authors to be confirmed upon publication)
- **Venue:** ICLR 2026
- **Relevance:** Introduces Late Injection strategy that bypasses passive shallow layers and Concave Pyramid Pruning with Early Exit, compressing ~90% of visual tokens while matching original performance and accelerating training by 1.72x. Directly relevant to DocSP's hierarchical token reduction approach.
- **Citations:** N/A (newly published)

### 14. Nuwa: Mending the Spatial Integrity Torn by VLM Token Pruning
- **Authors:** Yihong Huang et al.
- **Venue:** ICLR 2026
- **Relevance:** Critically important for DocSP: reveals that existing token pruning methods disrupt the global spatial reference frame, causing degradation on spatial localization tasks. Proposes a two-stage framework (swarm-intelligence-inspired spatial anchor retention + text-guided pruning) that preserves spatial integrity during token compression -- directly addresses DocSP's core challenge of maintaining spatial perception with reduced tokens.
- **Citations:** N/A (newly published)

### 15. PruneSID: Prune Redundancy, Preserve Essence -- Vision Token Compression in VLMs via Synergistic Importance-Diversity
- **Authors:** (Under review -- authors to be confirmed upon publication)
- **Venue:** ICLR 2026
- **Relevance:** Proposes a training-free approach with Principle Semantic Components Analysis (PSCA) for clustering tokens into semantically coherent groups and information-aware dynamic compression ratios based on image complexity. The importance-diversity synergy is relevant to DocSP's need to preserve diverse document structure elements while pruning redundant tokens.
- **Citations:** N/A (newly published)

---

## Summary by Topic

| Topic | Papers |
|-------|--------|
| **Document/Chart Understanding VLM** | #1 InternVL, #5 CharXiv, #6 DocLayLLM, #7 Docopilot, #8 Marten |
| **Spatial Perception in VLMs** | #2 SpatialVLM, #3 SpatialRGPT, #14 Nuwa |
| **Token Redundancy Reduction** | #4 FastV, #9 ATP-LLaVA, #10 Dynamic-LLaVA, #11 FlashVID, #12 VisionTrim, #13 HiDivDrop, #15 PruneSID |
| **Multi-stage Training / VLM Architecture** | #1 InternVL, #6 DocLayLLM, #7 Docopilot |

---

*Note: Citation counts are approximate and based on available data as of March 2026. Papers from 2026 (ICLR 2026) are newly published and citation counts are not yet available. All venues listed are top-tier (CVPR, ECCV, NeurIPS, ICLR, ICML).*
