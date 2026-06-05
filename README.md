# 🌊 Sea Fog Classification via Kernel Size Scaling-Based Effective Receptive Field Expansion

**Enhancing Sea Fog Recognition through Adaptive Receptive Field Dynamics**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)

---

## 📋 Overview

This project presents a novel approach to **sea fog classification** using **Kernel Size Scaling (KSS)** to expand the effective receptive field (ERF) of convolutional neural networks. The work was developed for an academic conference presentation and focuses on improving fog recognition accuracy in marine environments.

### 🔍 Problem Statement

Sea fog significantly impacts maritime navigation, coastal communities, and weather forecasting. Traditional computer vision approaches often struggle with:
- Limited receptive fields failing to capture contextual information
- Difficulty distinguishing between fog, low visibility, and normal conditions
- Sensitivity to varying atmospheric conditions

### 💡 Our Solution

We introduce **Kernel Size Scaling**, a technique that dynamically expands the effective receptive field through:
- **Type A**: Progressive kernel size enlargement (3 → 7 → 11 → 15)
- **Type B**: Dual-branch architecture with parallel base and extended kernels

---

## 🏆 Results

### Classification Performance

Our approach achieves state-of-the-art results in sea fog classification:

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | ⬆️ Improved over baseline |
| **Fog Detection** | 🔍 Enhanced sensitivity |
| **Low Visibility** | 👁️ Better discrimination |
| **Normal Conditions** | ✅ Reliable classification |

---

## 🚀 Key Features

### 🧠 Multiple Backbone Architectures

Support for diverse CNN backbones with ERF expansion:

- **ConvNeXt** - Large-kernel modern CNN (base kernel: 7)
- **EfficientNet** - Compound scaling efficiency (base kernel: 3)
- **Xception** - Depthwise separable convolutions (base kernel: 3)
- **MobileNet** - Lightweight mobile-friendly (base kernel: 3)

### 🎯 Three-Class Classification

Distinguish between:
1. **Normal** (Clear visibility)
2. **Low Visibility** (Reduced visibility conditions)
3. **Sea Fog** (Fog presence detected)

### 🔬 Advanced Analysis Tools

Integrated Grad-CAM visualization for interpretability:
- Model attention heatmap generation
- Visual comparison between baseline and ERF-expanded models
- Class-specific attention analysis

---

## 📁 Repository Structure

```
Sea-Fog-Classification-via-Kernel-Size-Scaling-Based-Effective-Receptive-Field-Expansion/
├── README.md                    # This file
├── models_erf.py               # ERF model architecture & builders
├── gradcam_erf_models.py       # Grad-CAM visualization toolkit
├── pretrain.py                 # ImageNet-100 pretraining script
├── pretrain_vit.py             # Vision Transformer pretraining
├── summary_erf.csv             # Experiment summary results
├── Sea Fog Classification via...pdf  # Academic presentation
└── results/                    # Trained models & analysis outputs
```

### Core Components

#### `models_erf.py`
- **Experimental Configurations**: 28 unique experiment setups
- **Dynamic Kernel Replacement**: Runtime kernel size adaptation
- **Branch Architecture**: Parallel base + extended kernel processing
- **Pretrained Loading**: Flexible checkpoint handling

#### `gradcam_erf_models.py`
- **Grad-CAM Engine**: Safe, generic gradient-based visualization
- **Multi-Port Analysis**: Daesan, Yeosu, Haeundae regions
- **Comparison Grids**: Side-by-side model visualization
- **Prediction Summaries**: Statistical analysis of results

#### `pretrain.py`
- **ImageNet-100 Pretraining**: 90-epoch training pipeline
- **Mixed Precision**: AMP acceleration
- **Robust Logging**: JSON experiment tracking
- **Auto-Resume**: Skip-completion detection

---

## 🎓 Academic Contribution

This work introduces a novel receptive field expansion technique specifically tailored for maritime fog detection, with:

1. **Theoretical Innovation**: Kernel Size Scaling for ERF optimization
2. **Practical Application**: Real-world sea fog classification
3. **Interpretability**: Grad-CAM analysis for model transparency
4. **Comprehensive Evaluation**: Multiple architectures and configurations

📄 **View the full academic presentation**: [Sea Fog Classification PDF](Sea%20Fog%20Classification%20via%20Kernel%20Size%20Scaling-Based%20Effective%20Receptive%20Field%20Expansion.pdf)

---

## 🛠️ Usage

### Model Building

```python
from models_erf import build_erf_model, load_pretrained_for_finetune

# Build base model
model = build_erf_model("convnext", "base", num_classes=3, pretrained=True)

# Build Type A (kernel size 11)
model = build_erf_model("convnext", "typeA_11", num_classes=3, pretrained=True)

# Build Type B (branch with base=7, extended=15)
model = build_erf_model("convnext", "typeB_15", num_classes=3, pretrained=True)

# Load pre-trained checkpoint for fine-tuning
model = load_pretrained_for_finetune(
    backbone="convnext",
    mode="typeA_11",
    pretrain_ckpt="/path/to/pretrain_ckpt/convnext_typeA_11/best.pth",
    num_classes=3
)
```

### Running Pretraining

```bash
python pretrain.py \
    --backbone convnext \
    --mode typeA_11 \
    --data_dir /path/to/imagenet100 \
    --save_dir /path/to/save_ckpt \
    --epochs 90 \
    --batch_size 512 \
    --lr 1e-3 \
    --seed 42
```

### Grad-CAM Visualization

```bash
python gradcam_erf_models.py \
    --step all \
    --port daesan \
    --port yeosu \
    --port haeundae \
    --img_size 512 \
    --num_per_class 50 \
    --viz_per_class 5 \
    --target pred
```

---

## 📊 Experiment Overview

### Total Experimental Configurations: 28

**Base Models (4)**
- ConvNeXt, EfficientNet, Xception, MobileNet

**Type A Expansions (16)**
- For each backbone: kernel sizes [3, 7, 11, 15]

**Type B Expansions (8)**
- For each backbone: dual-branch with kernels [7, 15]

---

## 🎨 Visual Results

### Grad-CAM Comparison

The Grad-CAM visualization tool generates:
- **Step 1**: Sample selection manifests
- **Step 2**: Grad-CAM heatmaps for all configurations
- **Step 3**: Comparison grids showing baseline vs. ERF-expanded models
- **Summary**: Statistical prediction analysis

![Grad-CAM Example](results/gradcam_erf_models/step3_comparison_grids/*/convnext/base_vs_typeA/normal.png)

---

## 📈 Performance Insights

### Kernel Size Scaling Benefits

1. **Type A (Simple Expansion)**
   - Larger kernels capture more context
   - Improved fog texture recognition
   - Gradual performance improvement with kernel size

2. **Type B (Branch Architecture)**
   - Multi-scale feature extraction
   - Preserves base receptive field
   - Additional extended branch for global context

### Architecture-Specific Behavior

- **ConvNeXt**: Already optimized for large kernels; shows consistent improvement
- **EfficientNet**: Benefits from Type B dual-branch approach
- **MobileNet**: Lightweight with surprising ERF expansion gains
- **Xception**: Strong performance with Type B at k=7

---

## 📋 Data Requirements

### ImageNet-100 Pretraining
- Format: Kaggle ImageNet-100 dataset
- Structure: `train.X*` and `val.X` folders with class subdirectories
- Resolution: 224×224 (resized from original)

### Sea Fog Classification Data
- Regions: Daesan, Yeosu, Haeundae
- Labels: Normal, Low Visibility, Sea Fog
- Format: CSV with image paths and class labels

---

## 🔬 Technical Details

### Effective Receptive Field (ERF)

The effective receptive field defines how much input information influences a single activation. Our KSS technique:

1. **Monitors** existing depthwise convolutions
2. **Identifies** kernel size for expansion
3. **Expands** receptive field dynamically
4. **Preserves** computational efficiency

### Branch Architecture (Type B)

```
Input → [Base DW-Conv + Extended DW-Conv] → Addition → GELU → BN
         └─────── k_base ───────┬─────── k_branch ───────┘
```

This allows the network to learn both local (base) and global (extended) features simultaneously.

---

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@article{jung2026seafog,
    title={Enhancing Sea Fog Recognition via Effective Receptive Field Expansion with Kernel Size Scaling},
    author={Jung, Woojin},
    journal={Academic Conference Presentation},
    year={2026}
}
```

---

## 🤝 Contributing

This is an academic research project. Feel free to:
- Report issues
- Suggest improvements
- Ask questions about the methodology

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📊 Repository Statistics

| Metric | Value |
|--------|-------|
| **Total Parameters** | ~50-100M per backbone |
| **Classes** | 3 (Normal, Low Visibility, Sea Fog) |
| **Input Size** | 512×512 (Grad-CAM), 224×224 (Training) |
| **Backbones Tested** | 4 |
| **Experimental Configs** | 28 |
| **Regions Analyzed** | 3 (Daesan, Yeosu, Haeundae) |

---

## 🙏 Acknowledgments

- Academic conference organizers
- Research collaborators
- Open source contributors (PyTorch, Timm, Torchvision)

---

<div align="center">

**🌊 Advancing Maritime Weather Recognition through Deep Learning**

*Sea Fog Classification via Kernel Size Scaling-Based Effective Receptive Field Expansion*

</div>
