# DocSP: Related Works & Bibliography

> DocSP 논문 작성을 위한 관련 연구 정리.
> 각 카테고리별로 **관련성 설명** + **BibTeX** 포함.

## 연구 분야 정의

**LLaVA-SP / DocSP가 속하는 연구 분야:**
- **cs.CV → Multimodal Large Language Models → Vision-Language Projector Design**
- **핵심 문제**: ViT가 2D 이미지를 1D patch sequence로 flatten → 인접 패치 간 spatial relationship 파괴
- **접근법**: Projector 단계에서 convolution 기반 spatial tokens를 추출하여 visual representation 강화
- **DocSP 차별점**: frequency decomposition으로 structural(고주파) / semantic(저주파) 정보를 분리하여 문서 특화

---

## 1. Direct Baseline: LLaVA-SP

DocSP의 직접적 기반. Spatial tokens 개념, SFE/DFI 구조를 제안.
DocSP는 LLaVA-SP의 content-agnostic 한계를 frequency decomposition으로 극복.

```bibtex
@inproceedings{lou2025llavasp,
  title={LLaVA-SP: Enhancing Visual Representation with Visual Spatial Tokens for MLLMs},
  author={Lou, Haoran and Fan, Chunxiao and Liu, Ziyan and Wu, Yuexin and Wang, Xinliang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

**DocSP와의 관계**: LLaVA-SP는 uniform pooling+conv로 spatial tokens를 추출하나, 문서의 구조적 특성(텍스트 경계 vs 색상 영역)을 구분하지 못함. DocSP는 주파수 분해를 도입하여 structural/semantic 정보를 명시적으로 분리.

---

## 2. LLaVA Series (Visual Instruction Tuning)

MLLM의 표준 아키텍처(ViT + Projector + LLM)와 2-stage 학습 전략을 확립.

```bibtex
@inproceedings{liu2023llava,
  title={Visual Instruction Tuning},
  author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}

@inproceedings{liu2024llava15,
  title={Improved Baselines with Visual Instruction Tuning},
  author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@misc{liu2024llavanext,
  title={LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge},
  author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Li, Bo and Zhang, Yuanhan and Shen, Sheng and Lee, Yong Jae},
  howpublished={\url{https://llava-vl.github.io/blog/2024-01-30-llava-next/}},
  year={2024}
}
```

**DocSP와의 관계**: DocSP는 LLaVA의 [ViT → Projector → LLM] 패러다임을 따르되, MLP projector를 DocSP projector로 교체. 2-stage 학습 전략(projector alignment → instruction tuning)도 동일하게 적용.

---

## 3. InternVL Series (Base Model)

DocSP의 base model. InternViT-300M + Qwen3-8B 구조 제공.

```bibtex
@inproceedings{chen2024internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@article{chen2024internvl15,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}

@inproceedings{chen2024internvl25,
  title={Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling},
  author={Chen, Zhe and Wang, Weiyun and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Cui, Erfei and Zhu, Jinguo and Ye, Shenglong and Tian, Hao and Liu, Zhaoyang and Gu, Lixin and others},
  booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2025}
}

@article{zhu2025internvl3,
  title={InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models},
  author={Zhu, Jinguo and Wang, Weiyun and Chen, Zhe and Liu, Zhaoyang and Ye, Shenglong and Gu, Lixin and Tian, Hao and Duan, Yuchen and others},
  journal={arXiv preprint arXiv:2504.10479},
  year={2025}
}

@article{wang2025internvl35,
  title={InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency},
  author={Wang, Weiyun and Gao, Zhangwei and Gu, Lixin and Pu, Hengjun and Cui, Long and Wei, Xingguang and Liu, Zhaoyang and others},
  journal={arXiv preprint arXiv:2508.18265},
  year={2025}
}
```

**DocSP와의 관계**: InternVL의 pixel shuffle + MLP1 projector를 유지하면서 DocSP spatial tokens를 추가하는 dual-path 구조. InternViT의 32×32 grid 출력을 DocSP의 입력으로 활용.

---

## 4. Qwen-VL Series (LLM Backbone)

Qwen3-8B가 DocSP의 LLM backbone.

```bibtex
@article{bai2023qwenvl,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}

@article{wang2024qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```

---

## 5. Document Understanding MLLMs

DocSP가 타겟하는 문서 이해 분야의 주요 모델들.

```bibtex
@article{feng2024docpedia,
  title={DocPedia: Unleashing the Power of Large Multimodal Model in the Frequency Domain for Versatile Document Understanding},
  author={Feng, Hao and Liu, Qi and Liu, Hao and Zhou, Wengang and Li, Houqiang and Huang, Can},
  journal={Science China Information Sciences},
  volume={67},
  year={2024}
}

@article{ye2023mplugdocowl,
  title={mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding},
  author={Ye, Jiabo and Hu, Anwen and Xu, Haiyang and Ye, Qinghao and Yan, Ming and Dan, Yuhao and Zhao, Chenlin and Xu, Guohai and Li, Chenliang and Tian, Junfeng and Qi, Qian and Zhang, Ji and Huang, Fei},
  journal={arXiv preprint arXiv:2307.02499},
  year={2023}
}

@article{liu2024textmonkey,
  title={TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document},
  author={Liu, Yuliang and Yang, Biao and Liu, Qiang and Li, Zhang and Ma, Zhiyin and Zhang, Shuo and Bai, Xiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}

@inproceedings{ye2023ureader,
  title={UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model},
  author={Ye, Jiabo and Hu, Anwen and Xu, Haiyang and Ye, Qinghao and Yan, Ming and Xu, Guohai and Li, Chenliang and Tian, Junfeng and Qian, Qi and Zhang, Ji and Jin, Qin and He, Liang and Lin, Xin Alex and Huang, Fei},
  booktitle={Findings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2023}
}
```

**DocSP와의 관계**: DocPedia는 주파수 도메인을 문서 이해에 처음 적용한 연구로, DocSP의 frequency decomposition 동기에 직접적 영향. 단, DocPedia는 ViT 입력을 DCT 변환하는 반면, DocSP는 ViT 출력의 feature-level 분해.

---

## 6. Frequency Decomposition in Vision

DocSP의 핵심 기여인 frequency decomposition의 이론적 배경.

```bibtex
@inproceedings{qin2021fcanet,
  title={FcaNet: Frequency Channel Attention Networks},
  author={Qin, Zequn and Zhang, Pengyi and Wu, Fei and Li, Xi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}

@inproceedings{chen2019octave,
  title={Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution},
  author={Chen, Yunpeng and Fan, Haoqi and Xu, Bing and Yan, Zhicheng and Kalantidis, Yannis and Rohrbach, Marcus and Yan, Shuicheng and Feng, Jiashi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}

@article{liu2019mwcnn,
  title={Multi-level Wavelet Convolutional Neural Networks},
  author={Liu, Pengju and Zhang, Hongzhi and Lian, Wei and Zuo, Wangmeng},
  journal={IEEE Access},
  volume={7},
  pages={74973--74985},
  year={2019}
}
```

**DocSP와의 관계**:
- **FcaNet**: 주파수 채널 어텐션 — 채널별 주파수 특성 활용의 이론적 근거
- **Octave Conv**: high/low frequency 분리 처리의 선행 연구. DocSP의 dual-branch 구조와 유사한 철학
- **MWCNN**: Wavelet 기반 multi-level 분해. DocSP의 Gaussian LP filter는 wavelet의 경량화 대안

---

## 7. Multi-scale Feature Extraction

DocSP의 Multi-Scale SFE (Spatial Feature Extractor) 설계 배경.

```bibtex
@inproceedings{lin2017fpn,
  title={Feature Pyramid Networks for Object Detection},
  author={Lin, Tsung-Yi and Dollar, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}

@inproceedings{he2014spp,
  title={Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2014}
}
```

**DocSP와의 관계**: SPP의 multi-scale pooling 개념을 spatial token 생성에 적용. FPN의 feature pyramid 철학을 ViT feature의 2D 구조에서 구현. DocSP의 [4, 8, 16, 32] 스케일은 SPP의 multi-level bin과 유사.

---

## 8. Chart/Table Understanding Benchmarks

DocSP의 평가 벤치마크 및 학습 데이터 관련.

```bibtex
@inproceedings{masry2022chartqa,
  title={ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning},
  author={Masry, Ahmed and Long, Do Xuan and Tan, Jia Qing and Joty, Shafiq and Hoque, Enamul},
  booktitle={Findings of the Association for Computational Linguistics (ACL)},
  year={2022}
}

@inproceedings{mathew2021docvqa,
  title={DocVQA: A Dataset for VQA on Document Images},
  author={Mathew, Minesh and Karatzas, Dimosthenis and Jawahar, C.V.},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2021}
}

@article{han2023chartllama,
  title={ChartLlama: A Multimodal LLM for Chart Understanding and Generation},
  author={Han, Yucheng and Zhang, Chi and Chen, Xin and Yang, Xu and Wang, Zhibin and Yu, Gang and Fu, Bin and Zhang, Hanwang},
  journal={arXiv preprint arXiv:2311.16483},
  year={2023}
}

@inproceedings{zhang2024tinychart,
  title={TinyChart: Efficient Chart Understanding with Visual Token Merging and Program-of-Thoughts Learning},
  author={Zhang, Liang and Hu, Anwen and Xu, Haiyang and Yan, Ming and Xu, Yichen and Jin, Qin and Zhang, Ji and Huang, Fei},
  booktitle={Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024}
}

@inproceedings{masry2024chartinstruct,
  title={ChartInstruct: Instruction Tuning for Chart Comprehension and Reasoning},
  author={Masry, Ahmed and Shahmohammadi, Mehrad and Parvez, Md Rizwan and Hoque, Enamul and Joty, Shafiq},
  booktitle={Findings of the Association for Computational Linguistics (ACL)},
  year={2024}
}
```

---

## 9. Vision-Language Projector Design (핵심 관련 연구)

LLaVA-SP/DocSP와 가장 직접적으로 관련된 분야. MLLM projector 아키텍처 개선 연구들.

```bibtex
@inproceedings{cha2024honeybee,
  title={Honeybee: Locality-enhanced Projector for Multimodal LLM},
  author={Cha, Junbum and Kang, Wooyoung and Mun, Jonghwan and Roh, Byungseok},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@article{qian2024saep,
  title={SAEP: Spatial-Aware Efficient Projector for MLLMs via Multi-Layer Feature Aggregation},
  author={Qian, Shun and Liu, Bingquan and Sun, Chengjie and Xu, Zhen and Wang, Baoxun},
  journal={arXiv preprint arXiv:2410.10319},
  year={2024}
}

@article{li2024tokenpacker,
  title={TokenPacker: Efficient Visual Projector for Multimodal LLM},
  author={Li, Wentong and Yuan, Yuqian and Liu, Jian and Tang, Dongqi and Wang, Song and Zhu, Jianke and Zhang, Lei},
  journal={International Journal of Computer Vision},
  year={2025}
}

@inproceedings{yao2024denseconnector,
  title={Dense Connector for MLLMs},
  author={Yao, Huanjin and Wu, Wenhao and Yang, Taojiannan and Song, YuXin and Zhang, Mengxi and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}

@article{li2024deco,
  title={DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models},
  author={Li, Yaomin and others},
  journal={arXiv preprint arXiv:2405.20985},
  year={2024}
}

@inproceedings{masry2025alignvlm,
  title={AlignVLM: Bridging Vision and Language Latent Spaces for Multimodal Understanding},
  author={Masry, Ahmed and Rodriguez, Juan A. and Zhang, Tianyu and Wang, Suyuchen and Wang, Chao and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}

@inproceedings{liu2024visualanchors,
  title={Visual Anchors Are Strong Information Aggregators for Multimodal Large Language Model},
  author={Liu, Haogeng and You, Quanzeng and Han, Xiaotian and Liu, Yongfei and Huang, Huaibo and He, Ran and Yang, Hongxia},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}

@article{shang2024llavaprumerge,
  title={LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models},
  author={Shang, Yuzhang and Cai, Mu and Xu, Bingxin and Lee, Yong Jae and Yan, Yan},
  journal={arXiv preprint arXiv:2403.15388},
  year={2024}
}
```

**DocSP와의 관계**:
- **SAEP** ⭐ **가장 직접적 관련**: depthwise conv로 multi-layer ViT features에서 spatial information 강화. LLaVA-SP/DocSP와 거의 동일한 동기와 접근
- **Honeybee**: C-Abstractor(conv 기반)로 locality 보존. DocSP도 conv로 spatial locality 보존하되, 주파수 기반 분리 추가
- **TokenPacker**: coarse-to-fine visual projector로 토큰 압축. region-to-point injection이 DocSP의 multi-scale SFE와 유사한 개념
- **Dense Connector**: multi-layer ViT feature 융합. DocSP는 single-layer에서 주파수 분해로 multi-scale 추출
- **DeCo**: adaptive pooling으로 spatial-locality 보존하는 projector. DocSP의 pooling-based SFE와 유사
- **AlignVLM**: visual features를 LLM embedding space에 직접 매핑. DocSP와 대비되는 대안적 접근
- **Visual Anchors**: ViT 내 정보 집약 토큰을 발견하여 connector 설계에 활용
- **LLaVA-PruMerge**: 토큰 수 절감 관점. DocSP는 8개만 추가하여 효율성 유지

---

## 10. Cross-Attention in VLMs

DocSP의 DAG-FI (Detail-Aggregated Feature Integrator) 설계 배경.

```bibtex
@inproceedings{alayrac2022flamingo,
  title={Flamingo: a Visual Language Model for Few-Shot Learning},
  author={Alayrac, Jean-Baptiste and Donahue, Jeff and Luc, Pauline and Miech, Antoine and Barr, Iain and Hasson, Yana and Lenc, Karel and Mensch, Arthur and Millican, Katie and Reynolds, Malcolm and Ring, Roman and Rutherford, Eliza and Cabi, Serkan and Han, Tengda and Gong, Zhitao and Samangooei, Sina and Monteiro, Marianne and Menick, Jacob and Borgeaud, Sebastian and Brock, Andrew and Nematzadeh, Aida and Sharifzadeh, Sahand and Binkowski, Mikolaj and Barreira, Ricardo and Vinyals, Oriol and Zisserman, Andrew and Simonyan, Karen},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@inproceedings{li2023blip2,
  title={BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
  author={Li, Junnan and Li, Dongxu and Savarese, Silvio and Hoi, Steven},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2023}
}

@inproceedings{jaegle2021perceiver,
  title={Perceiver: General Perception with Iterative Attention},
  author={Jaegle, Andrew and Gimeno, Felix and Brock, Andrew and Zisserman, Andrew and Vinyals, Oriol and Carreira, Joao},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2021}
}
```

**DocSP와의 관계**:
- **Flamingo**: Perceiver Resampler로 visual features를 고정 수 토큰으로 압축 — DocSP의 SFE가 유사한 역할
- **BLIP-2**: Q-Former의 cross-attention으로 learnable queries가 visual features를 집약 — DAG-FI의 설계 참고
- **Perceiver**: Iterative cross-attention으로 high-dim → low-dim 매핑 — DAG-FI의 이론적 근거

---

## 11. Foundation Models & Training Techniques

학습 인프라 관련 핵심 참고문헌.

```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2022}
}

@inproceedings{radford2021clip,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2021}
}

@inproceedings{zhai2023siglip,
  title={Sigmoid Loss for Language Image Pre-Training},
  author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}

@inproceedings{shi2016pixelshuffle,
  title={Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network},
  author={Shi, Wenzhe and Caballero, Jose and Huszar, Ferenc and Totz, Johannes and Aitken, Andrew P. and Bishop, Rob and Rueckert, Daniel and Wang, Zehan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}

@inproceedings{wu2018groupnorm,
  title={Group Normalization},
  author={Wu, Yuxin and He, Kaiming},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}

@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and Re, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@inproceedings{shazeer2018adafactor,
  title={Adafactor: Adaptive Learning Rates with Sublinear Memory Cost},
  author={Shazeer, Noam and Stern, Mitchell},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2018}
}
```

**DocSP와의 관계**:
- **LoRA**: Stage 2에서 LLM fine-tuning에 사용 (r=64)
- **Pixel Shuffle**: InternVL의 32×32→16×16 downsampling에 사용. DocSP는 pixel shuffle 전 원본 32×32 features를 입력으로 받음
- **GroupNorm**: DocSP 전체에서 BatchNorm 대신 사용 (small batch 안정성)
- **Adafactor**: 메모리 효율적 optimizer로 2×A100에서 대규모 학습 가능

---

## 12. Korean Document Understanding

DocSP의 bilingual (한국어/영어) 문서 이해 타겟 관련.

```bibtex
@article{ju2024varcovision,
  title={VARCO-VISION: Expanding Frontiers in Korean Vision-Language Models},
  author={Ju, Jeongho and Kim, Daeyoung and Park, SunYoung and Kim, Youngjune},
  journal={arXiv preprint arXiv:2411.19103},
  year={2024}
}
```

**DocSP와의 관계**: 한국어 VLM 연구의 선행 사례. DocSP는 한국어 문서/차트/테이블에 특화된 bilingual MLLM을 목표로 하며, VARCO-VISION 등과 비교 평가 가능.

---

## 13. Visual Token Compression / Pruning / Merging

DocSP의 "토큰 추가" 접근과 반대 방향의 연구. 논문에서 efficiency 비교 시 인용.

```bibtex
@inproceedings{chen2024fastv,
  title={An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models},
  author={Chen, Liang and Zhao, Haozhe and Liu, Tianyu and Bai, Shuai and Lin, Junyang and Zhou, Chang and Chang, Baobao},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}

@inproceedings{xing2024pyramiddrop,
  title={PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction},
  author={Xing, Long and Huang, Qidong and Dong, Xiaoyi and Lu, Jiajie and Zhang, Pan and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@inproceedings{yang2025visionzip,
  title={VisionZip: Longer is Better but Not Necessary in Vision Language Models},
  author={Yang, Senqiao and Chen, Yukang and Tian, Zhuotao and Wang, Chengyao and Li, Jingyao and Yu, Bei and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@inproceedings{ye2025atpllava,
  title={ATP-LLaVA: Adaptive Token Pruning for Large Vision Language Models},
  author={Ye, Xubing and Gan, Yukang and Ge, Yixiao and Zhang, Xiao-Ping and Tang, Yansong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@inproceedings{zhang2025llavamini,
  title={LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token},
  author={Zhang, Shaolei and Fang, Qingkai and Yang, Zhe and Feng, Yang},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025}
}

@inproceedings{huang2025dynamicllava,
  title={Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification},
  author={Huang, Wenxuan and Zhai, Zijie and Shen, Yunhang and Cao, Shaosheng and Zhao, Fei and others},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025}
}

@article{zhang2024sparsevlm,
  title={SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference},
  author={Zhang, Yuan and Fan, Chun-Kai and Ma, Junpeng and Zheng, Wenzhao and Huang, Tao and others},
  journal={arXiv preprint arXiv:2410.04417},
  year={2024}
}

@inproceedings{ranjbar2025divprune,
  title={DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models},
  author={Ranjbar Alvar, Saeed and Singh, Gursimran and Akbari, Mohammad and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

**DocSP와의 관계**: 이들은 visual token을 줄여서 효율성을 높이는 접근. DocSP는 반대로 8개 spatial tokens를 추가하여 representation quality를 높이되, 추가 토큰 수가 매우 적어(+3%) inference latency 거의 동일. 논문에서 "token compression과 orthogonal한 접근"으로 positioning 가능.
- **FastV**: attention 기반 토큰 pruning (ECCV'24 Oral). 가장 대표적 비교 대상
- **LLaVA-Mini**: 극단적 1-token 압축. DocSP의 반대 극단으로 비교 의미 있음
- **ATP-LLaVA**: Spatial Augmented Pruning — spatial 관계를 pruning에 활용. DocSP와 같은 spatial concern
- **PyramidDrop**: shallow layer에서 모든 visual token이 필요하다는 발견 → DocSP의 spatial tokens이 early layer에서 가장 효과적

---

## 14. Multi-Scale / Multi-Resolution Visual Feature Fusion

DocSP의 multi-scale SFE와 직접 관련된 연구들.

```bibtex
@article{zhang2024llavauhdv2,
  title={LLaVA-UHD v2: an MLLM Integrating High-Resolution Semantic Pyramid via Hierarchical Window Transformer},
  author={Zhang, Yipeng and Liu, Yifan and Guo, Zonghao and Zhang, Yidan and Yang, Xuesong and others},
  journal={arXiv preprint arXiv:2412.13871},
  year={2024}
}

@article{gen2024llavahr,
  title={Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models},
  author={Gen, Luoyu and others},
  journal={arXiv preprint arXiv:2403.03003},
  year={2024}
}

@article{li2024minigemini,
  title={Mini-Gemini: Mining the Potential of Multi-modality Vision Language Models},
  author={Li, Yanwei and Zhang, Yuechen and Wang, Chengyao and others},
  journal={arXiv preprint arXiv:2403.18814},
  year={2024}
}

@article{shi2024eagle,
  title={EAGLE: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders},
  author={Shi, Min and Liu, Fuxiao and Wang, Shihao and Liao, Shijia and Radhakrishnan, Subhashree and others},
  journal={arXiv preprint arXiv:2408.15998},
  year={2024}
}

@article{huang2024minimonkey,
  title={Mini-Monkey: Multi-Scale Adaptive Cropping for Multimodal Large Language Models},
  author={Huang, Mingxin and Liu, Yuliang and Liang, Dingkang and Jin, Lianwen and Bai, Xiang},
  journal={arXiv preprint arXiv:2408.02034},
  year={2024}
}

@article{cao2024mmfuser,
  title={MMFuser: Multimodal Multi-Layer Feature Fuser for Fine-Grained Vision-Language Understanding},
  author={Cao, Yue and Liu, Yangzhou and Chen, Zhe and Shi, Guangchen and Wang, Wenhai and Zhao, Danhuai and Lu, Tong},
  journal={arXiv preprint arXiv:2410.11829},
  year={2024}
}

@inproceedings{lin2025multilayerfusion,
  title={Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices},
  author={Lin, Junyan and Chen, Haoran and Fan, Yue and Fan, Yingqi and Jin, Xin and Su, Hui and Fu, Jinlan and Shen, Xiaoyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@inproceedings{li2025instructguidedfusion,
  title={Instruction-Guided Fusion of Multi-Layer Visual Features in Large Vision-Language Models},
  author={Li, Xu and Zheng, Yi and Chen, Haotian and Chen, Xiaolei and Liang, Yuxuan and Lai, Chenghang and Li, Bin and Xue, Xiangyang},
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025}
}
```

**DocSP와의 관계**:
- **LLaVA-UHD v2** ⭐: Hierarchical window attention + inverse semantic pyramid. DocSP와 매우 유사한 multi-scale spatial 접근
- **LLaVA-HR**: dual-resolution pathway. DocSP의 dual-path (pixel shuffle + DocSP) 구조와 유사한 철학
- **Mini-Monkey**: multi-scale adaptive cropping (input-level). DocSP는 feature-level multi-scale
- **MMFuser**: cross-attention으로 shallow layer의 fine-grained 정보를 deep layer에 주입. DAG-FI와 유사
- **Multi-Layer Fusion (CVPR'25)**: multi-layer fusion의 체계적 연구. DocSP의 single-layer frequency 분해가 multi-layer와 비교 가능

---

## 15. Visual Perception Token / Special Token Approaches

DocSP의 "spatial tokens 추가" 접근과 직접 관련. 표준 patch tokens 외 특수 토큰을 추가하는 연구.

```bibtex
@inproceedings{yu2025visualperceptiontoken,
  title={Introducing Visual Perception Token into Multimodal Large Language Model},
  author={Yu, Runpeng and Ma, Xinyin and Wang, Xinchao and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}

@article{bigverdi2024aurora,
  title={Perception Tokens Enhance Visual Reasoning in Multimodal Language Models},
  author={Bigverdi, Mahtab and Luo, Zelun and Hsieh, Cheng-Yu and Shen, Ethan and Chen, Dongping and Shapiro, Linda G and Krishna, Ranjay},
  journal={arXiv preprint arXiv:2412.03548},
  year={2024}
}

@article{jain2024visperlm,
  title={VisPer-LM: Elevating Visual Perception in Multimodal LLMs with Visual Embedding Distillation},
  author={Jain, Jitesh and Yang, Zhengyuan and Shi, Humphrey and Gao, Jianfeng and Yang, Jianwei},
  journal={arXiv preprint arXiv:2412.09585},
  year={2024}
}
```

**DocSP와의 관계**:
- **Visual Perception Token** ⭐ (ICCV'25): LLM이 자율적으로 visual perception action을 trigger하는 특수 토큰. DocSP의 spatial tokens와 같은 "추가 토큰" 패러다임이지만, DocSP는 projector에서, VPT는 LLM에서 생성
- **Aurora**: depth, detection 등 expert perception을 토큰으로 인코딩. DocSP의 structural/semantic 분리와 유사한 "specialized tokens" 개념
- **VisPer-LM** (NeurIPS'25): expert vision encoder의 지식을 LLM hidden representation에 distillation. DocSP와 같이 richer visual information 주입이 목표

---

## 16. Visual-Enhanced MLLMs (LLaVA-SP 논문이 직접 인용한 연구)

LLaVA-SP Related Works 섹션 2.2에서 직접 비교한 논문들.

```bibtex
@inproceedings{lin2023sphinx,
  title={SPHINX: The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal Large Language Models},
  author={Lin, Ziyi and Liu, Chris and Zhang, Renrui and Gao, Peng and others},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}

@article{wang2024densefusion,
  title={DenseFusion-1M: Merging Vision Experts for Comprehensive Multimodal Perception},
  author={Li, Xiaotong and others},
  journal={arXiv preprint arXiv:2407.08303},
  year={2024}
}

@article{li2024monkey,
  title={Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models},
  author={Li, Zhang and Yang, Biao and Liu, Qiang and Ma, Zhiyin and Zhang, Shuo and Yang, Jingxu and Sun, Yabo and Liu, Yuliang and Bai, Xiang},
  journal={arXiv preprint arXiv:2311.06607},
  year={2023}
}
```

**DocSP와의 관계**:
- **SPHINX**: 4개 vision encoder의 feature를 channel/sequence-wise mixing. DocSP는 단일 encoder에서 frequency 분해로 다양한 feature 추출
- **Monkey**: 이미지를 블록으로 나누어 병렬 ViT 처리. DocSP는 tile 단위 처리 + spatial token 추가

---

## Citation Map: DocSP의 기술적 계보

```
    Projector Design             Token Efficiency           Frequency Domain
    ───────────────              ──────────────             ────────────────
    Q-Former [ICML'23]           FastV [ECCV'24]            FcaNet [ICCV'21]
    Honeybee [CVPR'24]           LLaVA-PruMerge [ICCV'25]  Octave Conv [ICCV'19]
    SAEP [2024]                  PyramidDrop [CVPR'25]      DocPedia [2024]
    TokenPacker [IJCV'25]        LLaVA-Mini [ICLR'25]
    Dense Connector [NeurIPS'24] ATP-LLaVA [CVPR'25]
    Visual Anchors [NeurIPS'24]  VisionZip [CVPR'25]
            │                          │                         │
            ▼                          ▼                         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Multi-scale Features    Cross-Attention VLMs    Special Tokens │
    │  SPP [ECCV'14]           Perceiver [ICML'21]     VPT [ICCV'25] │
    │  FPN [CVPR'17]           Flamingo [NeurIPS'22]   Aurora [2024]  │
    │  LLaVA-UHD v2 [2024]    BLIP-2 [ICML'23]                      │
    │  Mini-Monkey [2024]      MMFuser [2024]                        │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
                          LLaVA [NeurIPS'23]
                          LLaVA-1.5 [CVPR'24]
                          LLaVA-SP [ICCV'25]
                                     │
                               ┌─────┴─────┐
                               │   DocSP   │
                               │  (Ours)   │
                               └─────┬─────┘
                                     │
                     ┌───────────────┼───────────────┐
               InternVL3.5      LoRA [ICLR'22]   GroupNorm [ECCV'18]
               [base model]     Adafactor [ICML'18]  FlashAttn [NeurIPS'22]
                                Pixel Shuffle [CVPR'16]
```

---

## Related Works 섹션 구성 제안 (논문용)

### 2.1 Multimodal Large Language Models
- LLaVA, LLaVA-1.5, LLaVA-NeXT, InternVL series, Qwen-VL
- 표준 아키텍처: ViT + Projector + LLM, 2-stage training

### 2.2 Visual-Enhanced MLLMs & Projector Design ⭐ (핵심 섹션)
- **MLP projector**: LLaVA — 단순하지만 spatial 정보 손실
- **Q-Former / Perceiver**: BLIP-2, Flamingo — cross-attention으로 압축, but 과도한 추상화
- **Conv-based projector**: Honeybee (C-Abstractor), SAEP — spatial prior 주입
- **Multi-scale projector**: TokenPacker, DeCo — coarse-to-fine 또는 pooling 기반
- **Spatial tokens**: LLaVA-SP — conv kernel으로 6개 spatial token 추가 (우리의 direct baseline)
- **Special tokens**: Visual Perception Token, Aurora — 추가 토큰으로 visual understanding 강화
- **Dense Connector, MMFuser**: multi-layer ViT feature 융합

### 2.3 Visual Token Efficiency
- FastV, LLaVA-PruMerge, PyramidDrop, VisionZip, ATP-LLaVA, LLaVA-Mini
- DocSP는 token compression과 orthogonal: 추가 토큰이 8개(+3%)로 latency 영향 미미

### 2.4 Frequency-aware Visual Representations
- FcaNet, Octave Conv — high/low frequency 분리 처리의 이론적 배경
- DocPedia — 주파수 도메인을 문서 이해에 최초 적용 (DCT 기반, DocSP는 feature-level 분해)

### 2.5 Document Understanding with MLLMs
- DocPedia, mPLUG-DocOwl, TextMonkey, UReader, TinyChart

### 2.6 Chart and Table Understanding
- ChartQA, DocVQA, ChartLlama, ChartInstruct

---

## 논문별 관련도 순위 (DocSP 기준)

### Tier 1: 가장 직접적 (반드시 비교/인용)
| 논문 | 관련도 | 이유 |
|------|--------|------|
| **LLaVA-SP** [ICCV'25] | ★★★★★ | Direct baseline. SFE+DFI를 DocSP가 확장 |
| **SAEP** [2024] | ★★★★★ | 거의 동일한 동기/접근: conv로 spatial info 강화 |
| **Honeybee** [CVPR'24] | ★★★★☆ | Conv-based projector, locality 보존 |
| **DocPedia** [2024] | ★★★★☆ | 주파수 도메인 + 문서 이해의 교차점 |

### Tier 2: 밀접한 관련 (Related Works에서 논의)
| 논문 | 관련도 | 이유 |
|------|--------|------|
| **TokenPacker** [IJCV'25] | ★★★★ | Projector 재설계, multi-scale region injection |
| **Dense Connector** [NeurIPS'24] | ★★★★ | Multi-layer feature fusion connector |
| **Visual Anchors** [NeurIPS'24] | ★★★★ | ViT 내 정보 구조를 활용한 connector |
| **LLaVA-UHD v2** [2024] | ★★★★ | Hierarchical window + feature pyramid |
| **Visual Perception Token** [ICCV'25] | ★★★★ | 추가 토큰 패러다임 |
| **FastV** [ECCV'24] | ★★★☆ | Token efficiency 대표 연구 |
| **LLaVA-Mini** [ICLR'25] | ★★★☆ | 극단적 압축 — DocSP의 반대 극단 |
| **MMFuser** [2024] | ★★★☆ | Cross-attention multi-layer fusion |

### Tier 3: 배경/맥락 (Introduction에서 언급)
| 논문 | 관련도 | 이유 |
|------|--------|------|
| LLaVA series | ★★★ | 표준 아키텍처 확립 |
| InternVL series | ★★★ | Base model |
| BLIP-2, Flamingo | ★★☆ | Cross-attention VLM의 배경 |
| FPN, SPP | ★★☆ | Multi-scale 이론적 배경 |
| Octave Conv, FcaNet | ★★☆ | Frequency 분해 이론적 배경 |
