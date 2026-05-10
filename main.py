# =============================================================================
# main.py — End-to-end pipeline: LeNet-5 + ZFNet on CIFAR-10
# Authors : Abenezer Seifu  (UGR/6499/14)
#           Tamiru Alemnew  (UGR/5857/14)
#           Yohannes Alemayehu (UGR/2497/14)
# Group   : 2
#
# This script ties together:
#   1) lenet5.py   — 32×32 CIFAR pipeline + LeNet-5
#   2) zfnet.py    — 224×224 upsampled CIFAR + ZFNet
#   3) Comparison plots + JSON histories under outputs/
#
# Usage:
#   python main.py             # Train both models, then compare
#   python main.py --lenet     # LeNet-5 only
#   python main.py --zfnet     # ZFNet only
#   python main.py --compare   # Load saved JSON histories and plot only
#
# Tensor flow summary:
#   LeNet-5:  batches [N, 3, 32, 32]  -> logits [N, 10]
#   ZFNet:    batches [N, 3, 224, 224] -> logits [N, 10]
# =============================================================================

import argparse
import json
import os

import torch

# LeNet-5 stack (native CIFAR resolution)
from lenet5 import (
    LeNet5,
    get_cifar10_loaders,
    train_model as train_lenet,
    plot_training_curves,
    show_sample_predictions,
    count_parameters,
)

# ZFNet stack (upsampled CIFAR)
from zfnet import (
    ZFNet,
    get_cifar10_loaders_zfnet,
    train_model as train_zfnet,
    compare_architectures,
    count_parameters as count_zf_params,
    plot_training_curves as plot_zf_curves,
    show_sample_predictions as show_zf_predictions,
)

# Device shared by both models in this process (CUDA if available).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_history(history, path):
    """
    Serialize training curves to JSON (lists of floats per epoch).

    Args:
        history: dict with keys train_loss, train_acc, test_loss, test_acc
        path: e.g. outputs/lenet5_history.json
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] History saved to {path}")


def load_history(path):
    """Load a history dict written by save_history."""
    with open(path, "r") as f:
        return json.load(f)


def run_lenet():
    """
    Full LeNet-5 pipeline: data (32×32) -> train -> plot -> checkpoint -> JSON.

    Returns:
        history dict for optional in-memory comparison.
    """
    print("\n" + "#" * 60)
    print("#  LENET-5 TRAINING")
    print("#" * 60)

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    model = LeNet5(num_classes=10).to(device)
    count_parameters(model)

    history = train_lenet(
        model,
        train_loader,
        test_loader,
        num_epochs=50,
        model_name="LeNet-5",
        device=device,
    )

    plot_training_curves(
        history, model_name="LeNet-5", save_path="outputs/lenet5_curves.png"
    )

    show_sample_predictions(
        model,
        test_loader,
        device,
        model_name="LeNet-5",
        save_path="outputs/lenet5_predictions.png",
    )

    torch.save(model.state_dict(), "checkpoints/lenet5_cifar10.pth")
    print("[INFO] Weights saved to checkpoints/lenet5_cifar10.pth")

    os.makedirs("outputs", exist_ok=True)
    save_history(history, "outputs/lenet5_history.json")

    return history


def run_zfnet():
    """
    Full ZFNet pipeline: data (224×224) -> train -> plot -> checkpoint -> JSON.
    """
    print("\n" + "#" * 60)
    print("#  ZFNET TRAINING")
    print("#" * 60)

    train_loader, test_loader = get_cifar10_loaders_zfnet(batch_size=64)

    model = ZFNet(num_classes=10).to(device)
    count_zf_params(model)

    history = train_zfnet(
        model,
        train_loader,
        test_loader,
        num_epochs=50,
        model_name="ZFNet",
        device=device,
    )

    plot_zf_curves(
        history, model_name="ZFNet", save_path="outputs/zfnet_curves.png"
    )

    show_zf_predictions(
        model,
        test_loader,
        device,
        model_name="ZFNet",
        save_path="outputs/zfnet_predictions.png",
    )

    os.makedirs("outputs", exist_ok=True)
    save_history(history, "outputs/zfnet_history.json")

    return history


def print_summary(lenet_history, zfnet_history):
    """
    Text table comparing best/final metrics and architecture metadata.

    Parameter counts are computed from fresh module definitions (no checkpoint
    needed) so the table stays accurate if hyperparameters change.
    """
    n_lenet = sum(p.numel() for p in LeNet5().parameters() if p.requires_grad)
    n_zf = sum(p.numel() for p in ZFNet().parameters() if p.requires_grad)

    print("\n" + "=" * 65)
    print("  FINAL RESULTS SUMMARY — CIFAR-10")
    print("=" * 65)
    print(f"{'Metric':<30} {'LeNet-5':>15} {'ZFNet':>15}")
    print("-" * 65)

    metrics = [
        (
            "Best Test Accuracy (%)",
            f"{max(lenet_history['test_acc']):.2f}",
            f"{max(zfnet_history['test_acc']):.2f}",
        ),
        (
            "Final Train Accuracy (%)",
            f"{lenet_history['train_acc'][-1]:.2f}",
            f"{zfnet_history['train_acc'][-1]:.2f}",
        ),
        (
            "Final Test Loss",
            f"{lenet_history['test_loss'][-1]:.4f}",
            f"{zfnet_history['test_loss'][-1]:.4f}",
        ),
        ("Parameters", f"{n_lenet:,}", f"{n_zf:,}"),
        ("Input Size", "32x32", "224x224"),
        ("Batch Size", "128", "64"),
    ]

    for name, l_val, z_val in metrics:
        print(f"  {name:<28} {l_val:>15} {z_val:>15}")

    print("=" * 65)


def main():
    """Parse CLI flags and dispatch training / comparison."""
    parser = argparse.ArgumentParser(
        description="Train LeNet-5 and/or ZFNet on CIFAR-10"
    )
    parser.add_argument("--lenet", action="store_true", help="Train LeNet-5 only")
    parser.add_argument("--zfnet", action="store_true", help="Train ZFNet only")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare saved histories under outputs/ (no training)",
    )
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if args.compare:
        lenet_h = load_history("outputs/lenet5_history.json")
        zfnet_h = load_history("outputs/zfnet_history.json")
        compare_architectures(
            lenet_h, zfnet_h, save_path="outputs/comparison_curves.png"
        )
        print_summary(lenet_h, zfnet_h)

    elif args.lenet:
        run_lenet()

    elif args.zfnet:
        run_zfnet()

    else:
        lenet_history = run_lenet()
        zfnet_history = run_zfnet()
        compare_architectures(
            lenet_history, zfnet_history, save_path="outputs/comparison_curves.png"
        )
        print_summary(lenet_history, zfnet_history)


if __name__ == "__main__":
    main()
