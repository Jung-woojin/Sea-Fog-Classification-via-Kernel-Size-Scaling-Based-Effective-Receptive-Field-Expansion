# 🌊 Sea Fog Classification via Kernel Size Scaling-Based Effective Receptive Field Expansion

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)

---

## 📋 Overview

This project investigates how **Kernel Size Scaling (KSS)**-based ERF expansion affects sea fog classification performance in maritime CCTV environments. Rather than proposing a new SOTA architecture, this work analyzes the **relationship between ERF expansion and classification accuracy** across multiple CNN backbones and port environments.

---
### 🔍 Problem Statement

CNN models with limited ERF exhibit **local texture bias** — over-responding to local noise, haze, and edge components visually similar to sea fog. This causes misclassification between normal visibility, low visibility, and sea fog conditions.

---
### 💡 Approach

We investigate two ERF expansion strategies applied to depthwise convolutions:

- **Type A**: Replace existing DWConv kernel with larger K×K kernel (K ∈ {3, 7, 11, 15})
- **Type B**: Add a large-kernel branch in parallel to the original DWConv (K ∈ {7, 15}), following RepLKNet-style design

Both are evaluated across 4 CNN backbones and 2 port environments.

---

## 🏆 Results

### Yeosu Port

| Model | Macro Precision | Macro Recall | Macro F1 | Params(M) | ΔF1 |
|-------|----------------|--------------|----------|-----------|-----|
| ConvNeXt TypeA_3 (baseline) | 0.819 | 0.727 | 0.688 | 86.85 | — |
| ConvNeXt TypeA_11 | 0.854 | 0.773 | **0.754** | 88.87 | **+0.066** |
| EfficientNet V2 Base (baseline) | 0.827 | 0.738 | 0.712 | 52.86 | — |
| EfficientNet V2 TypeB_7 | 0.815 | 0.759 | **0.748** | 56.15 | **+0.036** |
| MobileNet V3 Base (baseline) | 0.837 | 0.724 | 0.695 | 4.21 | — |
| MobileNet V3 TypeA_11 | 0.857 | 0.829 | **0.833** | 4.72 | **+0.138 ★** |
| Xception Base (baseline) | 0.836 | 0.737 | 0.706 | 20.81 | — |
| Xception TypeB_15 | 0.814 | 0.758 | **0.745** | 25.79 | **+0.039** |
| Swin-T (ImageNet-100) | 0.651 | 0.647 | 0.640 | 86.75 | ref |
| Swin-T (ImageNet-22k) | 0.780 | 0.688 | 0.651 | 86.75 | ref |

→ **4/4 backbones improved**, average ΔMacro F1 = **+0.070**

---

### Haeundae Port

| Model | Macro Precision | Macro Recall | Macro F1 | Params(M) | ΔF1 |
|-------|----------------|--------------|----------|-----------|-----|
| ConvNeXt TypeA_3 (baseline) | 0.746 | 0.728 | 0.707 | 86.85 | — |
| ConvNeXt Base | 0.791 | 0.780 | **0.774** | 87.57 | **+0.067 ★** |
| EfficientNet V2 Base (baseline) | 0.744 | 0.723 | 0.698 | 52.86 | — |
| EfficientNet V2 TypeA_7 | 0.746 | 0.730 | **0.716** | 55.46 | **+0.018** |
| MobileNet V3 Base (baseline) | 0.745 | 0.722 | 0.690 | 4.21 | — |
| MobileNet V3 TypeA_15 | 0.763 | 0.730 | **0.712** | 5.24 | **+0.022** |
| Xception Base (baseline) | 0.688 | 0.670 | 0.624 | 20.81 | — |
| Xception TypeB_7 | 0.770 | 0.758 | **0.748** | 21.99 | **+0.124 ★** |
| Swin-T (ImageNet-100) | 0.655 | 0.538 | 0.517 | 86.75 | ref |
| Swin-T (ImageNet-22k) | 0.767 | 0.734 | 0.713 | 86.75 | ref |

→ **4/4 backbones improved**, average ΔMacro F1 = **+0.058**

---

### Key Findings

**1. ERF expansion consistently improves performance**
- 8/8 cases improved across both ports (best-mode comparison)
- Maximum gain: MobileNet V3 TypeA_11 at Yeosu (+0.138 Macro F1)

**2. Bidirectional validation via ConvNeXt**
- ConvNeXt already uses 7×7 DWConv (inherently large ERF)
- Reducing kernel to 3×3 (TypeA_3) degrades performance: −0.067 at Haeundae
- This confirms ERF expansion as the causal factor, not mere architecture change

**3. Backbone-specific optimal design**
- Optimal mode varies per backbone and port environment
- No single configuration dominates across all settings

**4. Swin-T comparison**
- Swin-T (ImageNet-100): underperforms all CNN baselines
- Swin-T (ImageNet-22k): competitive, but CNN + ERF expansion achieves higher F1 at lower params (MobileNet: 4.72M vs 86.75M)

---

## 🧠 Backbone Architectures

| Backbone | Base DW Kernel | Activation | Notes |
|----------|---------------|------------|-------|
| ConvNeXt | 7×7 | GELU | Already ERF-expanded design |
| EfficientNetV2-M | 3×3 | SiLU | Fused-MBConv (early) + MBConv (late) |
| MobileNetV3 | 3×3 / 5×5 | h-swish | Mixed kernel, SE blocks |
| Xception | 3×3 | ReLU | Depthwise separable |

---

## 📁 Repository Structure

```
├── models_erf.py                # ERF model architecture & builders
├── gradcam_erf_models.py        # Grad-CAM visualization toolkit
├── pretrain.py                  # ImageNet-100 pretraining script
├── erf_analysis.py              # ERF measurement (Luo et al., 2016)
├── summary_erf.csv              # Full experiment results
└── results/                     # Trained models & analysis outputs
```

---

## 🛠️ Usage
 
### Model Building
 
```python
from models_erf import build_erf_model, load_pretrained_for_finetune
 
# Base model
model = build_erf_model("mobilenet", "base", num_classes=3)
 
# Type A (kernel replacement)
model = build_erf_model("mobilenet", "typeA_11", num_classes=3)
 
# Type B (branch addition)
model = build_erf_model("xception", "typeB_7", num_classes=3)
 
# Load pretrained checkpoint for fine-tuning
model = load_pretrained_for_finetune(
    backbone="mobilenet",
    mode="typeA_11",
    pretrain_ckpt="/path/to/pretrain_ckpt/mobilenet_typeA_11/best.pth",
    num_classes=3
)
```
 
### Pretraining (CNN)
 
```bash
# Single backbone/mode
python pretrain.py \
    --backbone mobilenet \
    --mode typeA_11 \
    --data_dir /path/to/imagenet100 \
    --save_dir /path/to/pretrain_ckpt \
    --epochs 90 \
    --batch_size 256 \
    --lr 1e-3
 
# All backbones/modes sequentially (GPU 0)
bash run_pretrain.sh
```
 
### Pretraining (Swin-T)
 
```bash
# From scratch with ImageNet-100
python pretrain_vit.py \
    --backbone swin \
    --data_dir /path/to/imagenet100 \
    --save_dir /path/to/pretrain_ckpt \
    --epochs 90 \
    --batch_size 256
```
 
### Fine-tuning (CNN)
 
```bash
# Single model
CUDA_VISIBLE_DEVICES=1 python train_erf.py \
    --backbone mobilenet \
    --mode typeA_11 \
    --port yeosu \
    --data_csv /path/to/splits.csv \
    --pretrain_ckpt /path/to/pretrain_ckpt/mobilenet_typeA_11/best.pth \
    --output_root /path/to/results/erf \
    --epochs 100 \
    --batch_size 32 \
    --img_size 512 \
    --lr 1e-3 \
    --patience 15
 
# All models sequentially (GPU 1)
bash run_finetune.sh
```
 
### Fine-tuning (Swin-T)
 
```bash
# With ImageNet-100 pretrained checkpoint
CUDA_VISIBLE_DEVICES=1 python train_vit.py \
    --backbone swin \
    --port yeosu \
    --data_csv /path/to/splits.csv \
    --pretrain_ckpt /path/to/pretrain_ckpt/swin_base/best.pth \
    --output_root /path/to/results/erf \
    --epochs 100 \
    --batch_size 32
 
# With ImageNet-22k pretrained weight (timm)
CUDA_VISIBLE_DEVICES=1 python train_vit.py \
    --backbone swin \
    --port yeosu \
    --data_csv /path/to/splits.csv \
    --pretrain_ckpt imagenet21k \
    --output_root /path/to/results/erf \
    --run_name swin_21k \
    --epochs 100 \
    --batch_size 32
```
 
### ERF Measurement
 
```bash
# Step 1: ERF measurement per backbone
python erf_analysis.py --step erf --port yeosu --num_per_class 50
 
# Step 2: ERF vs misclassification correlation
python erf_analysis.py --step correlation --port yeosu --num_per_class 100
 
# Step 4: ERF change before/after expansion
python erf_analysis.py --step delta --port yeosu --num_per_class 50
 
# All steps at once
python erf_analysis.py --step all --port yeosu --num_per_class 50
```
 
### Grad-CAM Visualization
 
```bash
# Paper figures (correct predictions)
CUDA_VISIBLE_DEVICES=1 python gradcam_paper.py \
    --pair xception \
    --port haeundae \
    --num_images 30
 
# Misclassification mining
CUDA_VISIBLE_DEVICES=1 python gradcam_failure.py \
    --pair xception \
    --port haeundae \
    --max_per_case 10
 
# All pairs
CUDA_VISIBLE_DEVICES=1 python gradcam_failure.py \
    --pair all \
    --port haeundae \
    --max_per_case 10
```

## 📊 Experimental Configuration

| Category | Detail |
|----------|--------|
| Backbones | ConvNeXt, EfficientNetV2-M, MobileNetV3, Xception |
| ERF modes | base, typeA_{3,7,11,15}, typeB_{7,15} |
| Total configs | 28 |
| Pretraining data | ImageNet-100 (130,000 train / 5,000 val) |
| Fine-tuning data | KHOA CCTV — Yeosu, Haeundae |
| Train/Val/Test split | 6,300 / 900 / 1,800 (temporal split: 2018-2022 / 2023-2024) |
| Input size | 512×512 (fine-tuning), 224×224 (pretraining) |
| Primary metric | Macro-F1 |
| GPU | NVIDIA H200 |

---

## 🔬 ERF Measurement

ERF is measured following **Luo et al. (2016)**:

1. Feed random noise input (n=50 trials)
2. Target: last spatial feature map center unit
3. Compute input gradient → average over trials
4. Measure **Weighted Spatial Spread (σ)**:

$$\sigma = \sqrt{\sum_{i,j} \tilde{G}(i,j) \cdot d(i,j)^2}$$

This approach is data-independent and measures the structural ERF of the model itself.

---


## 🙏 Acknowledgments

- Korea Hydrographic and Oceanographic Agency (KHOA) for CCTV data
- PyTorch, timm, mmdetection open source communities
