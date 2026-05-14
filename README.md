
# Assignment 3 — Deep Learning for Computer Vision
# ZFNet & LeNet-5 Implementation

## Project Structure

```
project/
│
├── lenet5.py                        # LeNet-5 (canonical annotated implementation)
├── zfnet.py                         # Compact ZFNet-style STL-10 implementation
├── main.py                          # CLI: full training + comparison
├── Assignment3_STL10_Walkthrough.ipynb    # Notebook for the STL-10 setup
├── requirements.txt                 # Python dependencies
│
├── data/              # STL-10 auto-downloaded here on first run
├── checkpoints/       # Saved model weights (.pth files)
└── outputs/           # Training curves, plots, history JSON files
```

**Notebook vs scripts:** The implementation lives in `lenet5.py` and `zfnet.py`. The notebook mirrors the default training configuration used by the scripts, while `python main.py` provides the terminal workflow.

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
- Download STL-10 automatically
- Train LeNet-5 on STL-10 resized to 32x32
- Train ZFNet on STL-10 resized to 64x64
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

### 6. Jupyter walkthrough (optional)

```bash
pip install -r requirements.txt   # includes jupyter / ipykernel
jupyter notebook Assignment3_STL10_Walkthrough.ipynb
```

Or open the `.ipynb` in VS Code / Cursor and select the project `venv` kernel.

The notebook defaults now match `main.py`. If you want shorter exploratory runs, you can reduce `NUM_EPOCHS_LENET` or `NUM_EPOCHS_ZFNET`. Use **`python main.py`** and `outputs/*_history.json` for the main scripted workflow.

---

## Dataset and Runtime

| Item                 | LeNet-5   | ZFNet     |
|----------------------|-----------|-----------|
| Dataset              | STL-10    | STL-10    |
| Input Size           | 32x32     | 64x64     |
| Default Epochs       | 30        | 40        |
| Purpose              | baseline  | compact ZFNet-style CNN |

> **Tip**: This codebase is tuned for CPU-friendly STL-10 runs. The ZFNet path now uses a lighter 64x64 configuration so it converges more reliably on a normal laptop CPU.

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
| `outputs/lenet5_predictions.png` | Sample predicted images (LeNet-5)  |
| `outputs/zfnet_predictions.png`  | Sample predicted images (ZFNet)    |
| `checkpoints/lenet5_stl10.pth`   | Saved LeNet-5 weights              |
| `checkpoints/zfnet_best.pth`  | Best ZFNet weights (by test acc)   |
| `checkpoints/zfnet_stl10_final.pth` | Final ZFNet weights            |
