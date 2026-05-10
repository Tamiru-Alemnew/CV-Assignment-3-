# =============================================================================
# Assignment 3 — Deep Learning for Computer Vision
# ZFNet & LeNet-5 Implementation
#
# Group 2:
#   Abenezer Seifu     UGR/6499/14
#   Tamiru Alemnew     UGR/5857/14
#   Yohannes Alemayehu UGR/2497/14
# =============================================================================

## Project Structure

```
project/
│
├── lenet5.py          # LeNet-5 model definition, training, evaluation
├── zfnet.py           # ZFNet model definition, training, evaluation
├── main.py            # Run both models and compare results
├── requirements.txt   # Python dependencies
│
├── data/              # CIFAR-10 auto-downloaded here on first run
├── checkpoints/       # Saved model weights (.pth files)
└── outputs/           # Training curves, plots, history JSON files
```

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run both models (recommended)

```bash
python main.py
```

This will:
- Download CIFAR-10 automatically
- Train LeNet-5 (32x32 input, 50 epochs)
- Train ZFNet (224x224 input, 50 epochs)
- Plot training curves for both
- Show sample predictions
- Print a comparison summary table

### 3. Run individual models

```bash
python main.py --lenet    # LeNet-5 only
python main.py --zfnet    # ZFNet only
```

### 4. Compare already-trained models

```bash
python main.py --compare  # Uses saved history JSON files
```

### 5. Run each model script directly

```bash
python lenet5.py    # Standalone LeNet-5
python zfnet.py     # Standalone ZFNet
```

---

## Expected Results on CIFAR-10

| Metric               | LeNet-5   | ZFNet     |
|----------------------|-----------|-----------|
| Test Accuracy        | ~70-74%   | ~85-89%   |
| Parameters           | ~60K      | ~58M      |
| Training Time (CPU)  | ~10 min   | ~2-4 hrs  |
| Training Time (GPU)  | ~3 min    | ~20 min   |

> **Tip**: ZFNet trains much faster on a GPU. If you only have CPU,
> consider reducing NUM_EPOCHS to 20 and batch size to 32 in zfnet.py.

---

## Hardware Requirements

- **Minimum** : 8 GB RAM, any modern CPU (slow for ZFNet)
- **Recommended** : NVIDIA GPU with 4+ GB VRAM (CUDA 11+)

---

## Outputs Generated

| File                          | Description                        |
|-------------------------------|------------------------------------|
| `outputs/lenet5_curves.png`   | LeNet-5 loss and accuracy plots    |
| `outputs/zfnet_curves.png`    | ZFNet loss and accuracy plots      |
| `outputs/comparison_curves.png`| Side-by-side comparison           |
| `lenet5_predictions.png`      | Sample predicted images (LeNet-5)  |
| `zfnet_predictions.png`       | Sample predicted images (ZFNet)    |
| `checkpoints/lenet5_cifar10.pth` | Saved LeNet-5 weights           |
| `checkpoints/zfnet_best.pth`  | Best ZFNet weights (by test acc)   |
| `checkpoints/zfnet_final.pth` | Final ZFNet weights                |
