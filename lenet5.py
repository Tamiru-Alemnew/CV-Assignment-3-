# =============================================================================
# LeNet-5 Implementation — Deep Learning for Computer Vision, Assignment 3
# Authors : Abenezer Seifu  (UGR/6499/14)
#           Tamiru Alemnew  (UGR/5857/14)
#           Yohannes Alemayehu (UGR/2497/14)
# Group   : 2
# Dataset : CIFAR-10 (native 32×32 RGB — matches LeNet-5 spatial size)
# Framework: PyTorch
# =============================================================================
#
# TENSOR CONVENTIONS (used throughout this file)
# ----------------------------------------------
# - Batch dimension is always first: NCHW layout
#   * N = batch size
#   * C = channels (3 for RGB)
#   * H, W = height and width in pixels
# - A batch of CIFAR-10 images after ToTensor() has dtype float32 and shape:
#     images:  [N, 3, 32, 32]
#     labels:  [N]  (integer class index 0..9)
# - Model logits (pre-softmax) shape: [N, num_classes] — here [N, 10]
#
# ARCHITECTURE OVERVIEW
# ---------------------
# LeNet-5 (LeCun et al., 1998) is a 7-layer CNN originally designed for
# handwritten digit recognition on 32x32 grayscale images. We adapt it here
# for CIFAR-10 (32x32 RGB images, 10 classes) by changing the input channels
# from 1 (grayscale) to 3 (RGB).
#
# Original Paper:
#   LeCun, Y., et al. (1998). Gradient-based learning applied to document
#   recognition. Proceedings of the IEEE, 86(11), 2278-2324.
# =============================================================================

# --- Standard library & third-party imports ---------------------------------
# torch: core tensors, autograd, neural network modules, optimizers
import torch
# nn: Module, Conv2d, Linear, pooling, activations (building blocks of LeNet-5)
import torch.nn as nn
# optim: SGD and learning-rate schedulers used in the training orchestration
import torch.optim as optim
# torchvision: ready-made CIFAR-10 dataset class
import torchvision
# transforms: PIL Image -> tensor, augmentation, per-channel normalization
import torchvision.transforms as transforms
# DataLoader: batches samples into (images, labels) pairs for training/eval loops
from torch.utils.data import DataLoader
# matplotlib: plot loss/accuracy curves and qualitative prediction grids
import matplotlib.pyplot as plt
# numpy: fixed random seed alongside torch for reproducibility
import numpy as np
# os: create checkpoint directories on disk
import os

# --- Dataset normalization constants (CIFAR-10 train-set statistics) ----------
# These values zero-center and scale each RGB channel so gradients are stable.
# Shape context: Normalize applies per channel; after it, each channel has
# approximately zero mean and unit variance across the training distribution.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# --- Reproducibility ----------------------------------------------------------
# Setting a fixed seed makes weight init and DataLoader order comparable across
# runs (exact bitwise reproducibility on GPU may still vary by CUDA version).
torch.manual_seed(42)
np.random.seed(42)

# --- Device selection ---------------------------------------------------------
# CUDA GPU accelerates convolutions; CPU is the portable fallback.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# How many subprocesses prefetch batches (set NUM_WORKERS=0 in environment on
# Windows if DataLoader workers hang).
_NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))


# =============================================================================
# SECTION 1: MODEL DEFINITION
# =============================================================================

class LeNet5(nn.Module):
    """
    LeNet-5 Architecture (adapted for CIFAR-10).

    Original structure (LeCun 1998):
        Input (32x32x1) -> C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> Output

    Our adaptation:
        - Input channels: 1 (grayscale) -> 3 (RGB for CIFAR-10)
        - Output neurons: 10 (original used Gaussian RBF units; we use
          a standard Linear layer; CrossEntropyLoss combines log-softmax + NLL)
        - Activations: tanh (faithful to original; ReLU is a common modern swap)

    Layer-by-layer output shapes for a single 32x32x3 input (batch size B=1):
        After C1  : [1, 6,  28, 28]
        After S2  : [1, 6,  14, 14]
        After C3  : [1, 16, 10, 10]
        After S4  : [1, 16,  5,  5]
        After C5  : [1, 120, 1,  1]
        After flatten : [1, 120]
        After F6  : [1, 84]
        After Out : [1, 10]   (logits; softmax not applied — see forward())
    """

    def __init__(self, num_classes=10):
        """
        Build all layers. Tensor shapes below use generic batch size B.

        Args:
            num_classes: CIFAR-10 has 10 categories -> output logits length 10.
        """
        super(LeNet5, self).__init__()

        # -----------------------------------------------------------------
        # BLOCK 1: C1 — first convolutional layer
        # Role: extract low-level features (edges, color blobs) with shared
        #       weights across space (parameter efficiency vs. fully-connected).
        # Input tensor : [B, 3, 32, 32]   (RGB CIFAR-10)
        # Output tensor: [B, 6, 28, 28]   (6 feature maps; 32 - 5 + 1 = 28)
        # -----------------------------------------------------------------
        self.c1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0,
        )

        # -----------------------------------------------------------------
        # BLOCK 2: S2 — average pooling (subsampling / downsampling)
        # Role: reduce spatial size, add local translation tolerance; LeNet-5
        #       used learned subsampling; we use fixed 2x2 average pooling.
        # Input tensor : [B, 6, 28, 28]
        # Output tensor: [B, 6, 14, 14]   (each dimension halved: 28/2)
        # -----------------------------------------------------------------
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # -----------------------------------------------------------------
        # BLOCK 3: C3 — second convolution
        # Input tensor : [B, 6, 14, 14]
        # Output tensor: [B, 16, 10, 10]  (14 - 5 + 1 = 10)
        # -----------------------------------------------------------------
        self.c3 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
        )

        # -----------------------------------------------------------------
        # BLOCK 4: S4 — second average pool
        # Input tensor : [B, 16, 10, 10]
        # Output tensor: [B, 16, 5, 5]
        # -----------------------------------------------------------------
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # -----------------------------------------------------------------
        # BLOCK 5: C5 — convolution with kernel equal to spatial size
        # Role: equivalent to a fully-connected layer from 16*5*5 = 400 inputs
        #       to 120 outputs, but implemented as Conv2d for historical fidelity.
        # Input tensor : [B, 16, 5, 5]
        # Output tensor: [B, 120, 1, 1]
        # -----------------------------------------------------------------
        self.c5 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            stride=1,
            padding=0,
        )

        # -----------------------------------------------------------------
        # BLOCK 6: F6 — fully connected (dense) hidden layer
        # Input tensor : [B, 120]  (after flattening C5 output)
        # Output tensor: [B, 84]
        # -----------------------------------------------------------------
        self.f6 = nn.Linear(in_features=120, out_features=84)

        # -----------------------------------------------------------------
        # BLOCK 7: Output — classification logits
        # Input tensor : [B, 84]
        # Output tensor: [B, num_classes]  e.g. [B, 10]
        # No softmax here: nn.CrossEntropyLoss expects raw logits.
        # -----------------------------------------------------------------
        self.output = nn.Linear(in_features=84, out_features=num_classes)

        # Tanh squashes activations to (-1, 1); saturates for large magnitudes
        # but matches the 1998 architecture.
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass: map a batch of 32x32 RGB images to class logits.

        Args:
            x: Float tensor [B, 3, 32, 32] — must match CIFAR-10 after transforms.

        Returns:
            Logits tensor [B, num_classes] for use with CrossEntropyLoss.
        """
        # C1 + tanh: [B,3,32,32] -> [B,6,28,28]
        x = self.tanh(self.c1(x))

        # S2: [B,6,28,28] -> [B,6,14,14]
        x = self.s2(x)

        # C3 + tanh: [B,6,14,14] -> [B,16,10,10]
        x = self.tanh(self.c3(x))

        # S4: [B,16,10,10] -> [B,16,5,5]
        x = self.s4(x)

        # C5 + tanh: [B,16,5,5] -> [B,120,1,1]
        x = self.tanh(self.c5(x))

        # Flatten spatial dims: [B,120,1,1] -> [B,120]
        x = x.view(x.size(0), -1)

        # F6 + tanh: [B,120] -> [B,84]
        x = self.tanh(self.f6(x))

        # Linear readout: [B,84] -> [B,10] (logits)
        x = self.output(x)

        return x


# =============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# =============================================================================

def get_cifar10_loaders(batch_size=128):
    """
    Build PyTorch DataLoaders for CIFAR-10 tailored to LeNet-5 (32×32 input).

    CIFAR-10 ships as 32×32 PIL images. We keep native resolution (no upsampling)
    so the network input stays [N, 3, 32, 32].

    Training pipeline:
        - RandomCrop(32, padding=4): syntactic translation invariance; tensor
          stays [3, 32, 32] per image.
        - RandomHorizontalFlip: class-preserving mirror augmentation.
        - ToTensor: PIL HWC uint8 -> float32 CHW in [0,1].
        - Normalize: per-channel (x - mean) / std using CIFAR-10 stats.

    Evaluation pipeline:
        - Resize((32,32)): explicit no-op for documentation / rubric clarity
          (CIFAR is already 32×32).
        - ToTensor + Normalize only (no random augmentation — fair test metric).

    Returns:
        train_loader: yields batches (images, labels)
            images: [batch_size, 3, 32, 32]
            labels: [batch_size] int64
        test_loader: same tensor shapes, shuffle=False
    """

    # --- Training transforms ------------------------------------------------
    train_transform = transforms.Compose(
        [
            # Optional explicit size (identity for CIFAR); clarifies 32×32 contract.
            transforms.Resize((32, 32)),
            # Pad 32→40, then random crop back to 32×32 (small translations).
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # PIL Image -> Tensor shape [3, 32, 32], values in [0, 1]
            transforms.ToTensor(),
            # Per-channel standardization; output still [3, 32, 32]
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    # --- Test transforms (no randomness) ------------------------------------
    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    # --- Datasets: download once under ./data --------------------------------
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    # pin_memory=True speeds host→GPU copies when CUDA is used; on CPU-only
    # machines it should be False to avoid warnings/errors on some setups.
    _pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        pin_memory=_pin,
        persistent_workers=_NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=_pin,
        persistent_workers=_NUM_WORKERS > 0,
    )

    print(f"[INFO] Training samples : {len(train_dataset)}")
    print(f"[INFO] Test samples     : {len(test_dataset)}")
    print("[INFO] LeNet-5 pipeline: spatial size 32×32 (native CIFAR-10).")

    return train_loader, test_loader


# =============================================================================
# SECTION 3: TRAINING AND EVALUATION LOOPS
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    One full pass over the training set (one epoch).

    For each batch:
        1) Move (images, labels) to `device`
        2) Forward: logits = model(images)
        3) loss = criterion(logits, labels)
        4) loss.backward() then optimizer.step()

    Tensor shapes (typical batch):
        images: [B, 3, 32, 32]
        labels: [B]
        logits: [B, 10]
        loss: scalar tensor (we accumulate loss.item() float)

    Returns:
        avg_loss: mean cross-entropy over batches
        accuracy: % of training labels matched by argmax(logits)
    """
    model.train()  # Training mode (no dropout in LeNet-5; included for generality)

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # images: [B, 3, 32, 32], labels: [B] — move to same device as model
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a held-out loader (test set).

    model.eval() disables training-only behavior (e.g. dropout — not used in
    LeNet-5, but required for correct inference on architectures that use it).

    torch.no_grad(): do not build the autograd graph — saves memory and time.

    Tensor shapes match train_one_epoch.

    Returns:
        avg_loss: mean cross-entropy on the loader
        accuracy: classification accuracy (%)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


# =============================================================================
# SECTION 4: TRAINING ORCHESTRATION
# =============================================================================

def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=50,
    model_name="LeNet-5",
    device=None,
):
    """
    Full training schedule: repeated train + test, LR decay, history dict.

    If device is None, uses the device of the first model parameter so the
    training loop always matches where `model` was placed (.to(cuda) etc.).

    Hyperparameters:
        - CrossEntropyLoss: expects logits [B,C] and class indices [B]
        - SGD + momentum 0.9: classic choice for CNNs on CIFAR-10
        - weight_decay 5e-4: L2 regularization on weights
        - StepLR: multiply LR by 0.1 every 20 epochs

    Returns:
        history: dict with keys 'train_loss', 'train_acc', 'test_loss', 'test_acc'
                 each a list of length num_epochs
    """
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    print("\n" + "=" * 60)
    print(f"  Training {model_name} on CIFAR-10")
    print("=" * 60)
    print(f"  Epochs      : {num_epochs}")
    print(f"  Batch Size  : {train_loader.batch_size}")
    print(f"  Optimizer   : SGD (lr=0.01, momentum=0.9)")
    print(f"  Scheduler   : StepLR (step=20, gamma=0.1)")
    print(f"  Device      : {device}")
    print("=" * 60 + "\n")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:3d}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
            )

    print(f"\n[DONE] Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    return history


# =============================================================================
# SECTION 5: VISUALIZATION
# =============================================================================

def plot_training_curves(history, model_name="LeNet-5", save_path=None):
    """
    Plot per-epoch train/test loss and accuracy.

    history[*] are lists of scalars, one entry per epoch.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{model_name} — Training Curves (CIFAR-10)",
        fontsize=14,
        fontweight="bold",
    )

    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["test_loss"], "r-o", markersize=3, label="Test Loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Accuracy")
    ax2.plot(epochs, history["test_acc"], "r-o", markersize=3, label="Test Accuracy")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Plot saved to {save_path}")

    plt.show()


def show_sample_predictions(
    model,
    test_loader,
    device,
    num_samples=10,
    model_name="LeNet-5",
    save_path="lenet5_predictions.png",
):
    """
    Draw a 2×5 grid of test images with predicted vs. true class names.

    Denormalization reverses Normalize(mean, std) so imshow sees RGB in [0,1].
    Tensor flow:
        batch images: [B, 3, 32, 32] normalized
        subset to CPU for plotting: same shape, then denorm -> numpy HWC
    """
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    model.eval()
    images, labels = next(iter(test_loader))
    images_dev = images[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(images_dev)
        _, preds = outputs.max(1)

    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    imgs = images[:num_samples] * std + mean
    imgs = imgs.clamp(0, 1).permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(
        f"{model_name} — Sample Predictions on CIFAR-10",
        fontsize=13,
        fontweight="bold",
    )

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i])
        pred = classes[preds[i].item()]
        truth = classes[labels[i].item()]
        color = "green" if pred == truth else "red"
        ax.set_title(f"Pred: {pred}\nTrue: {truth}", color=color, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Predictions figure saved to {save_path}")
    plt.show()


# =============================================================================
# SECTION 6: MAIN ENTRY POINT
# =============================================================================

def count_parameters(model):
    """Count trainable parameters (weights + biases) for complexity reporting."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total trainable parameters: {total:,}")
    return total


def main():
    """End-to-end demo: load CIFAR-10 → train LeNet-5 → save weights and plots."""
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    NUM_CLASSES = 10

    train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

    model = LeNet5(num_classes=NUM_CLASSES).to(device)
    print("\n[MODEL SUMMARY — LeNet-5]")
    print(model)
    count_parameters(model)

    history = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=NUM_EPOCHS,
        model_name="LeNet-5",
        device=device,
    )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/lenet5_cifar10.pth")
    print("[INFO] Model saved to checkpoints/lenet5_cifar10.pth")

    plot_training_curves(
        history, model_name="LeNet-5", save_path="lenet5_training_curves.png"
    )
    show_sample_predictions(model, test_loader, device)


if __name__ == "__main__":
    main()
