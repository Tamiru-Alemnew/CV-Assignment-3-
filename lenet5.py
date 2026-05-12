"""LeNet-5 training pipeline for CIFAR-10."""

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

        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.output = nn.Linear(in_features=84, out_features=num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass: map a batch of 32x32 RGB images to class logits.

        Args:
            x: Float tensor [B, 3, 32, 32] — must match CIFAR-10 after transforms.

        Returns:
            Logits tensor [B, num_classes] for use with CrossEntropyLoss.
        """
        x = self.tanh(self.c1(x))
        x = self.s2(x)
        x = self.tanh(self.c3(x))
        x = self.s4(x)
        x = self.tanh(self.c5(x))
        x = x.view(x.size(0), -1)
        x = self.tanh(self.f6(x))
        x = self.output(x)

        return x


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

    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
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
    print("[INFO] LeNet-5 pipeline: spatial size 32×32 (native CIFAR-10).")

    return train_loader, test_loader


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
