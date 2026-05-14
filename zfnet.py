"""Compact ZFNet-style training pipeline for STL-10."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
import os

STL10_MEAN = (0.507, 0.487, 0.441)
STL10_STD = (0.267, 0.256, 0.276)

# Keep runs reproducible for report figures and metric comparisons.
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

_NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))


class ZFNet(nn.Module):
    """Compact ZFNet-style CNN for 64x64 STL-10 inputs."""

    def __init__(self, num_classes=10, input_size=64):
        super().__init__()
        self.input_size = input_size

        # Convolutional feature extractor. BatchNorm stabilizes the deeper model
        # and adaptive pooling keeps the classifier input size fixed.
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=192,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        flatten_dim = self._infer_flatten_dim()

        # Smaller fully connected head than the original ZFNet for CPU-friendly
        # STL-10 training while preserving the conv-to-classifier structure.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        """Map `[batch, 3, 64, 64]` images to `[batch, num_classes]` logits."""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _infer_flatten_dim(self):
        """Infer classifier input width from the current feature extractor."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_size, self.input_size)
            feats = self.pool(self.features(dummy))
            return feats.view(1, -1).size(1)

    def _initialize_weights(self):
        """Initialize layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def get_stl10_loaders_zfnet(batch_size=96, input_size=64):
    """Create STL-10 train and test loaders for ZFNet."""
    # Keep more spatial detail than LeNet-5 while remaining practical on CPU.
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size, padding=6),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ]
    )

    # Test preprocessing mirrors training resolution without random augmentation.
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ]
    )

    train_dataset = torchvision.datasets.STL10(
        root="./data", split="train", download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.STL10(
        root="./data", split="test", download=True, transform=test_transform
    )

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
    print(f"[INFO] ZFNet pipeline: STL-10 at {input_size}×{input_size}.")

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Standard supervised update: forward -> loss -> backward -> optimizer step.
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
    """Evaluate the model on a data loader."""
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


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=40,
    model_name="ZFNet",
    device=None,
):
    """Train ZFNet and return per-epoch loss/accuracy history."""
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()

    # AdamW converges faster than plain SGD for this compact ZFNet variant.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # Cosine decay gradually reduces the learning rate across the full run.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    print("\n" + "=" * 60)
    print(
        f"  Training {model_name} on STL-10 "
        f"({model.input_size}×{model.input_size})"
    )
    print("=" * 60)
    print(f"  Epochs      : {num_epochs}")
    print(f"  Batch Size  : {train_loader.batch_size}")
    print(f"  Optimizer   : AdamW (lr=1e-3, wd=1e-4)")
    print(f"  Scheduler   : CosineAnnealingLR")
    print(f"  Device      : {device}")
    print("=" * 60 + "\n")

    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
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
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/zfnet_best.pth")
            print(f"  [CHECKPOINT] New best test accuracy: {best_acc:.2f}%")

    print(f"\n[DONE] Best Test Accuracy: {best_acc:.2f}%")
    return history


def plot_training_curves(history, model_name="ZFNet", save_path=None, show=True):
    """Plot loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{model_name} — Training Curves (STL-10)",
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

    if show:
        plt.show()
    plt.close(fig)


def show_sample_predictions(
    model,
    test_loader,
    device,
    num_samples=10,
    model_name="ZFNet",
    save_path="zfnet_predictions.png",
    show=True,
):
    """Plot sample predictions from the test set."""
    # STL-10 class order from torchvision.datasets.STL10.
    classes = [
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ]

    model.eval()
    images, labels = next(iter(test_loader))
    images_dev = images[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(images_dev)
        _, preds = outputs.max(1)

    # Undo normalization so matplotlib receives displayable RGB values.
    mean = torch.tensor(STL10_MEAN).view(3, 1, 1)
    std = torch.tensor(STL10_STD).view(3, 1, 1)
    imgs = images[:num_samples] * std + mean
    imgs = imgs.clamp(0, 1).permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(
        f"{model_name} — Sample Predictions on STL-10",
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
    if show:
        plt.show()
    plt.close(fig)


def compare_architectures(
    lenet_history, zfnet_history, save_path="comparison_curves.png", show=True
):
    """Plot LeNet-5 and ZFNet test curves together."""
    epochs_l = range(1, len(lenet_history["test_acc"]) + 1)
    epochs_z = range(1, len(zfnet_history["test_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "LeNet-5 vs ZFNet — Comparison on STL-10",
        fontsize=14,
        fontweight="bold",
    )

    ax1.plot(epochs_l, lenet_history["test_loss"], "b-", label="LeNet-5")
    ax1.plot(epochs_z, zfnet_history["test_loss"], "r-", label="ZFNet")
    ax1.set_title("Test Loss Comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_l, lenet_history["test_acc"], "b-", label="LeNet-5")
    ax2.plot(epochs_z, zfnet_history["test_acc"], "r-", label="ZFNet")
    ax2.set_title("Test Accuracy Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Comparison plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def count_parameters(model):
    """Print and return the number of trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total trainable parameters: {total:,}")
    return total


def main():
    """Run the standalone ZFNet training script."""
    BATCH_SIZE = 96
    NUM_EPOCHS = 40
    NUM_CLASSES = 10
    INPUT_SIZE = 64

    # End-to-end standalone run: data -> train -> save weights/plots/history.
    train_loader, test_loader = get_stl10_loaders_zfnet(
        batch_size=BATCH_SIZE, input_size=INPUT_SIZE
    )

    model = ZFNet(num_classes=NUM_CLASSES, input_size=INPUT_SIZE).to(device)
    print("\n[MODEL SUMMARY — ZFNet]")
    print(model)
    count_parameters(model)

    history = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=NUM_EPOCHS,
        model_name="ZFNet",
        device=device,
    )

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/zfnet_stl10_final.pth")
    print("[INFO] Final model saved to checkpoints/zfnet_stl10_final.pth")

    plot_training_curves(
        history,
        model_name="ZFNet",
        save_path="outputs/zfnet_curves.png",
        show=False,
    )
    show_sample_predictions(
        model,
        test_loader,
        device,
        save_path="outputs/zfnet_predictions.png",
        show=False,
    )

    with open("outputs/zfnet_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
