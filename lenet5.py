"""LeNet-5 training pipeline for STL-10.

The original LeNet-5 was designed for small grayscale images. This project
adapts the architecture to STL-10 by:

- accepting 3-channel RGB inputs,
- resizing images to 32x32,
- keeping the classic conv-pool-conv-pool-conv pattern, and
- using a lightweight classifier head suitable for a CPU-friendly baseline.
"""

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
from typing import TypeAlias

STL10_MEAN = (0.507, 0.487, 0.441)
STL10_STD = (0.267, 0.256, 0.276)
STL10_CLASSES = [
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
History: TypeAlias = dict[str, list[float]]

# Keep runs reproducible for report figures and metric comparisons.
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

_NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))


class LeNet5(nn.Module):
    """LeNet-5 adapted for 32x32 RGB STL-10 inputs.

    The layer naming follows the classic LeNet-5 convention (`c1`, `s2`, ...)
    so the architecture remains easy to map back to the textbook model.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Feature extractor: 32x32 -> 28x28 -> 14x14 -> 10x10 -> 5x5 -> 1x1.
        # We keep tanh activations to stay close to the original LeNet design.
        self.c1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0,
        )

        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.c3 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
        )

        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.c5 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            stride=1,
            padding=0,
        )

        # Classifier head: flatten 120 features -> 84 hidden units -> class logits.
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.output = nn.Linear(in_features=84, out_features=num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map `[batch, 3, 32, 32]` images to `[batch, num_classes]` logits."""
        x = self.tanh(self.c1(x))
        x = self.s2(x)
        x = self.tanh(self.c3(x))
        x = self.s4(x)
        x = self.tanh(self.c5(x))
        x = torch.flatten(x, 1)
        x = self.tanh(self.f6(x))
        x = self.output(x)

        return x


def get_stl10_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Create STL-10 train and test loaders for LeNet-5.

    Training includes simple spatial augmentation to reduce overfitting, while
    evaluation uses deterministic preprocessing so test metrics remain stable.
    """

    # Resize STL-10 to the spatial size expected by the original LeNet-5 design.
    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ]
    )

    # Evaluation uses deterministic preprocessing only.
    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
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
    print("[INFO] LeNet-5 pipeline: STL-10 resized to 32×32.")

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return average loss and accuracy."""
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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a data loader without gradient updates."""
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
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 50,
    model_name: str = "LeNet-5",
    device: torch.device | None = None,
) -> History:
    """Train LeNet-5 and return per-epoch loss/accuracy history.

    The history dictionary is the common contract shared by the notebook,
    plotting functions, and the `main.py --compare` workflow.
    """
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()

    # LeNet-5 uses the simpler SGD setup as the baseline configuration.
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # Decay the learning rate after every 10 epochs.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    print("\n" + "=" * 60)
    print(f"  Training {model_name} on STL-10")
    print("=" * 60)
    print(f"  Epochs      : {num_epochs}")
    print(f"  Batch Size  : {train_loader.batch_size}")
    print(f"  Optimizer   : SGD (lr=0.01, momentum=0.9)")
    print(f"  Scheduler   : StepLR (step=10, gamma=0.1)")
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


def plot_training_curves(
    history: History,
    model_name: str = "LeNet-5",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot training and test loss/accuracy curves for a LeNet run."""
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
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Plot saved to {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def show_sample_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_samples: int = 10,
    model_name: str = "LeNet-5",
    save_path: str = "lenet5_predictions.png",
    show: bool = True,
) -> None:
    """Plot sample predictions from the STL-10 test set."""

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
        pred = STL10_CLASSES[preds[i].item()]
        truth = STL10_CLASSES[labels[i].item()]
        color = "green" if pred == truth else "red"
        ax.set_title(f"Pred: {pred}\nTrue: {truth}", color=color, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Predictions figure saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def count_parameters(model: nn.Module) -> int:
    """Print and return the number of trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total trainable parameters: {total:,}")
    return total


def main() -> None:
    """Run the standalone LeNet-5 training script.

    This entry point is useful when the project needs only the LeNet baseline,
    without invoking the combined workflow in `main.py`.
    """
    BATCH_SIZE = 128
    NUM_EPOCHS = 30
    NUM_CLASSES = 10

    # End-to-end standalone run: data -> train -> save weights/plots/history.
    train_loader, test_loader = get_stl10_loaders(batch_size=BATCH_SIZE)

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
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/lenet5_stl10.pth")
    print("[INFO] Model saved to checkpoints/lenet5_stl10.pth")

    plot_training_curves(
        history,
        model_name="LeNet-5",
        save_path="outputs/lenet5_curves.png",
        show=False,
    )
    show_sample_predictions(
        model,
        test_loader,
        device,
        save_path="outputs/lenet5_predictions.png",
        show=False,
    )

    with open("outputs/lenet5_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
