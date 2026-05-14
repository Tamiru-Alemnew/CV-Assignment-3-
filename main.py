"""End-to-end LeNet-5 and ZFNet training pipeline for STL-10."""

import argparse
import json
import os

import torch

from lenet5 import (
    LeNet5,
    get_stl10_loaders,
    train_model as train_lenet,
    plot_training_curves,
    show_sample_predictions,
    count_parameters,
)

from zfnet import (
    ZFNet,
    get_stl10_loaders_zfnet,
    train_model as train_zfnet,
    compare_architectures,
    count_parameters as count_zf_params,
    plot_training_curves as plot_zf_curves,
    show_sample_predictions as show_zf_predictions,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_history(history, path):
    """Save a history dictionary as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] History saved to {path}")


def load_history(path):
    """Load a saved history dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def run_lenet():
    """Run the full LeNet-5 pipeline."""
    print("\n" + "#" * 60)
    print("#  LENET-5 TRAINING")
    print("#" * 60)

    # Build loaders, train the model, then save plots/weights/history.
    train_loader, test_loader = get_stl10_loaders(batch_size=128)

    model = LeNet5(num_classes=10).to(device)
    count_parameters(model)

    history = train_lenet(
        model,
        train_loader,
        test_loader,
        num_epochs=30,
        model_name="LeNet-5",
        device=device,
    )

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
        model_name="LeNet-5",
        save_path="outputs/lenet5_predictions.png",
        show=False,
    )

    torch.save(model.state_dict(), "checkpoints/lenet5_stl10.pth")
    print("[INFO] Weights saved to checkpoints/lenet5_stl10.pth")

    os.makedirs("outputs", exist_ok=True)
    save_history(history, "outputs/lenet5_history.json")

    return history


def run_zfnet():
    """Run the full ZFNet pipeline."""
    print("\n" + "#" * 60)
    print("#  ZFNET TRAINING")
    print("#" * 60)

    # Build loaders, train the model, then save plots/weights/history.
    train_loader, test_loader = get_stl10_loaders_zfnet(batch_size=96, input_size=64)

    model = ZFNet(num_classes=10, input_size=64).to(device)
    count_zf_params(model)

    history = train_zfnet(
        model,
        train_loader,
        test_loader,
        num_epochs=40,
        model_name="ZFNet",
        device=device,
    )

    plot_zf_curves(
        history,
        model_name="ZFNet",
        save_path="outputs/zfnet_curves.png",
        show=False,
    )

    show_zf_predictions(
        model,
        test_loader,
        device,
        model_name="ZFNet",
        save_path="outputs/zfnet_predictions.png",
        show=False,
    )

    torch.save(model.state_dict(), "checkpoints/zfnet_stl10_final.pth")
    print("[INFO] Final weights saved to checkpoints/zfnet_stl10_final.pth")

    os.makedirs("outputs", exist_ok=True)
    save_history(history, "outputs/zfnet_history.json")

    return history


def print_summary(lenet_history, zfnet_history):
    """Print a summary table for both models."""
    # Recompute parameter counts from the current model definitions.
    n_lenet = sum(p.numel() for p in LeNet5().parameters() if p.requires_grad)
    n_zf = sum(p.numel() for p in ZFNet().parameters() if p.requires_grad)

    print("\n" + "=" * 65)
    print("  FINAL RESULTS SUMMARY — STL-10")
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
        ("Input Size", "32x32", "64x64"),
        ("Batch Size", "128", "96"),
    ]

    for name, l_val, z_val in metrics:
        print(f"  {name:<28} {l_val:>15} {z_val:>15}")

    print("=" * 65)


def main():
    """Parse CLI flags and run the requested workflow."""
    parser = argparse.ArgumentParser(
        description="Train LeNet-5 and/or ZFNet on STL-10"
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

    # Compare-only mode reuses saved JSON histories and skips training.
    if args.compare:
        lenet_h = load_history("outputs/lenet5_history.json")
        zfnet_h = load_history("outputs/zfnet_history.json")
        compare_architectures(
            lenet_h, zfnet_h, save_path="outputs/comparison_curves.png", show=False
        )
        print_summary(lenet_h, zfnet_h)

    elif args.lenet:
        run_lenet()

    elif args.zfnet:
        run_zfnet()

    else:
        # Default behavior runs both models and then produces a comparison plot.
        lenet_history = run_lenet()
        zfnet_history = run_zfnet()
        compare_architectures(
            lenet_history,
            zfnet_history,
            save_path="outputs/comparison_curves.png",
            show=False,
        )
        print_summary(lenet_history, zfnet_history)


if __name__ == "__main__":
    main()
