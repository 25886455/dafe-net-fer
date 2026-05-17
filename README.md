# DAFE-Net: Facial Expression Recognition with Domain Adaptation

**6G7V0024 Deep Learning Coursework — Manchester Metropolitan University**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/devved/dafe-net-fer)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)]()

---

## Overview

This repository contains two connected experiments in facial expression recognition (FER).

**Phase 1 (`raf-data-aug.ipynb`)** trains four models on balanced RAF-DB:
- Model A: Custom CNN from scratch — 92.97% test accuracy
- Model B: Pretrained ResNet-50 — 98.46%
- Model C: ResNet-50 + CBAM attention — 98.90%
- Model D: ResNet-50 + CBAM + Focal Loss — **98.97% (beats all published SOTA)**

**Phase 2 (`final-dl.ipynb`)** addresses domain shift — the finding that 98.97% benchmark accuracy does not transfer to user-uploaded photographs:
- HybridFER-21k: 49,000 images from RAF-DB + FER-2013 + CK+
- Baseline ResNet-50: 77.96% on HybridFER-21k
- DAFE-Net: 74.65% with substantially better cross-domain calibration and generalisation

---

## Repository Structure
dafe-net-fer/
├── raf-data-aug.ipynb       # Phase 1: RAF-DB balancing + 4-model ablation
├── final-dl.ipynb           # Phase 2: HybridFER-21k + DAFE-Net training
├── checkpoints/
│   ├── Baseline_ResNet50.pt # Trained Phase 2 baseline weights
│   └── DAFENet.pt           # Trained DAFE-Net weights
└── README.md

---

## Results

### Phase 1 — RAF-DB SOTA Comparison

| Method | Accuracy |
|---|---|
| SE-ResNet-18 (Li 2019) | 86.7% |
| FT-CSAT (Zhou 2023) | 88.6% |
| ResNet-50+CBAM (2024) | 91.9% |
| TriCAFFNet (2024) | 92.2% |
| **Model A: Custom CNN (ours)** | **92.97%** |
| **Model D: CBAM+Focal (ours)** | **98.97%** |

### Phase 2 — HybridFER-21k

| Model | Accuracy | F1 | Macro AUC |
|---|---|---|---|
| Baseline ResNet-50 | 77.96% | 0.779 | — |
| DAFE-Net (ours) | 74.65% | 0.745 | 0.949 |

---

## Datasets

### Phase 1
- **RAF-DB**: http://www.whdeng.cn/raf/model1.html (request access)
- Place at `RAW_DATASETS/RAF-DB/train/` and `RAW_DATASETS/RAF-DB/test/`
- Subfolders 1–7 map to: 1=Surprise, 2=Fear, 3=Disgust, 4=Happy, 5=Sad, 6=Angry, 7=Neutral

### Phase 2 (additional)
- **FER-2013**: https://www.kaggle.com/datasets/msambare/fer2013
- **CK+**: https://www.kaggle.com/datasets/shawon10/ckplus
- Place at `RAW_DATASETS/FER2013/` and `RAW_DATASETS/CK+/`

HybridFER-21k is built automatically by running cells 1–8 of `final-dl.ipynb`.

---

## How to Run

### Install dependencies

```bash
pip install torch torchvision torchmetrics torchinfo \
            scikit-learn matplotlib seaborn gradio \
            opencv-python-headless pandas umap-learn
```

### Phase 1

```bash
jupyter notebook raf-data-aug.ipynb
```

1. Update paths in Cell 3 to point to your RAF-DB location
2. Run all cells top to bottom
3. Gradio app launches in the final cell

### Phase 2

```bash
jupyter notebook final-dl.ipynb
```

1. Download all three datasets and update paths in Cell 3
2. Run Cell 8 to parse CK+ CSV into image folders
3. Set `DEBUG_MODE = False` in the debug cell
4. Run all cells — training takes ~3 hours on a GPU
5. Pretrained checkpoints load automatically if present in `checkpoints/`

### Inference only (skip retraining)

```python
import torch
# Load DAFE-Net
checkpoint = torch.load('checkpoints/DAFENet.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## Novel Contributions

### Adaptive Domain Normalisation (ADN)
A learnable per-channel blend of InstanceNorm and BatchNorm inserted after each CBAM block:
ADN(x) = sigmoid(α) · IN(x) + (1 − sigmoid(α)) · BN(x)

`α` is shape `(C, 1, 1)`, initialised at 0.5, learned jointly with all parameters. InstanceNorm removes per-image domain statistics; BatchNorm retains discriminative features.

### Face Region Erasing (FRE)
Randomly occludes the eye region (upper 35% of face bounding box) or mouth region (lower 30%) to simulate glasses, masks, and partial occlusion during training.

### DomainStyleJitter
Random gamma correction (γ ∈ [0.6, 1.8]) plus sharpness variation to simulate different camera sensor response curves.

---

## Live Demo

**https://huggingface.co/spaces/devved/dafe-net-fer**

Upload any face image or use your webcam. Both models predict simultaneously with full 7-class probability distributions.

---

## Citation

```bibtex
@misc{patel2025dafenet,
  title  = {DAFE-Net: FER with Domain Adaptation and Multi-Source Dataset Fusion},
  author = {Patel, Dev},
  year   = {2025},
  url    = {https://github.com/THE-DGP/dafe-net-fer}
}
```

---

## Acknowledgements

RAF-DB (Li et al. CVPR 2019) · FER-2013 (Goodfellow et al. 2013) · CK+ (Lucey et al. 2010) · CBAM (Woo et al. ECCV 2018) · BlurPool (Zhang ICML 2019) · Focal Loss (Lin et al. ICCV 2017)
EOF
