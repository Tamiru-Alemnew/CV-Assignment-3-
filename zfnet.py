"""ZFNet training pipeline for CIFAR-10."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

_NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))


class ZFNet(nn.Module):
    """
    ZFNet feature extractor + classifier, ImageNet-style, adapted to 10 classes.

    Spatial shape chain for input 224×224 (batch B omitted):
        Conv1+ReLU:  [B, 96, 110, 110]
        Pool1:       [B, 96, 55, 55]
        LRN1:        [B, 96, 55, 55]
        Conv2+ReLU:  [B, 256, 26, 26]
        Pool2:       [B, 256, 13, 13]
        LRN2:        [B, 256, 13, 13]
        Conv3+ReLU:  [B, 384, 13, 13]
        Conv4+ReLU:  [B, 384, 13, 13]
        Conv5+ReLU:  [B, 256, 13, 13]
        Pool5:       [B, 256, 6, 6]
        Flatten:     [B, 9216]
        FC path:     [B, 4096] → [B, 4096] → [B, num_classes]
    """

    def __init__(self, num_classes=10):
        super(ZFNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=7,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] float tensor (normalized CIFAR upsampled).

        Returns:
            Logits [B, num_classes] for CrossEntropyLoss.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        Gaussian weight init (std=0.01) and zero biases for conv/linear layers.
        Simpler than the original AlexNet paper but stable for student-scale runs.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)


def get_cifar10_loaders_zfnet(batch_size=64):
    """
    CIFAR-10 loaders with 224×224 spatial size for ZFNet.

    Why upsample?
        ZFNet/AlexNet stems expect ~ImageNet-scale inputs. CIFAR is 32×32; we
        Resize to 224×224 so the first conv sees a compatible receptive-field
        layout (assignment-appropriate compromise vs. training on full ImageNet).

    Training transforms:
        Resize((224,224), bilinear): PIL 32×32 -> 224×224 (tensor later [3,224,224])
        RandomHorizontalFlip: label-preserving augmentation
        RandomCrop(224, padding=8): small jitters at full resolution
        ToTensor: -> float [0,1], shape [3,224,224]
        Normalize(CIFAR mean/std): same stats as LeNet for comparability

    Test transforms:
        Resize + ToTensor + Normalize (no random geometry)

    Returns:
        train_loader / test_loader yielding:
            images: [batch_size, 3, 224, 224]
            labels: [batch_size]
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=8),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
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
    print("[INFO] ZFNet pipeline: spatial size 224×224 (upsampled from 32×32).")

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Single training epoch. Batch tensors:
        images: [B, 3, 224, 224]
        labels: [B]
        logits: [B, 10]
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
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
    Evaluation loop; dropout disabled via model.eval() so full width is used.
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


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=50,
    model_name="ZFNet",
    device=None,
):
    """
    Train ZFNet with SGD + StepLR; track metrics; save best test checkpoint.

    Checkpoint:
        checkpoints/zfnet_best.pth — weights when test accuracy improved most.

    If device is None, inferred from model parameters.
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
    print(f"  Training {model_name} on CIFAR-10 (224×224)")
    print("=" * 60)
    print(f"  Epochs      : {num_epochs}")
    print(f"  Batch Size  : {train_loader.batch_size}")
    print(f"  Optimizer   : SGD (lr=0.01, momentum=0.9, wd=5e-4)")
    print(f"  Scheduler   : StepLR (step=20, gamma=0.1)")
    print(f"  Device      : {device}")
    print("=" * 60 + "\n")

    best_acc = 0.0

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

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/zfnet_best.pth")
            print(f"  [CHECKPOINT] New best test accuracy: {best_acc:.2f}%")

    print(f"\n[DONE] Best Test Accuracy: {best_acc:.2f}%")
    return history


def plot_training_curves(history, model_name="ZFNet", save_path=None):
    """Plot train/test loss and accuracy vs. epoch (lists in history dict)."""
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
    model_name="ZFNet",
    save_path="zfnet_predictions.png",
):
    """Qualitative grid; images denormalized from CIFAR stats for display."""
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


def compare_architectures(
    lenet_history, zfnet_history, save_path="comparison_curves.png"
):
    """
    Overlay LeNet-5 vs ZFNet test metrics by epoch.

    Note: curves are comparable in *trend*; absolute epoch index assumes both
    histories used the same number of epochs when generating the JSON.
    """
    epochs_l = range(1, len(lenet_history["test_acc"]) + 1)
    epochs_z = range(1, len(zfnet_history["test_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "LeNet-5 vs ZFNet — Comparison on CIFAR-10",
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
    plt.show()


def count_parameters(model):
    """Return and print total trainable parameter count."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total trainable parameters: {total:,}")
    return total


def main():
    """Standalone ZFNet training script entry point."""
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    NUM_CLASSES = 10

    train_loader, test_loader = get_cifar10_loaders_zfnet(batch_size=BATCH_SIZE)

    model = ZFNet(num_classes=NUM_CLASSES).to(device)
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
    torch.save(model.state_dict(), "checkpoints/zfnet_final.pth")
    print("[INFO] Final model saved to checkpoints/zfnet_final.pth")

    plot_training_curves(
        history, model_name="ZFNet", save_path="zfnet_training_curves.png"
    )
    show_sample_predictions(
        model, test_loader, device, save_path="zfnet_predictions.png"
    )


if __name__ == "__main__":
    main()
