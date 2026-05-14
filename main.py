"""Command-line entry point for the STL-10 project.

This module wires together the reusable LeNet-5 and ZFNet utilities exposed by
`lenet5.py` and `zfnet.py`. The goal is to keep the training logic inside the
model modules while giving the project a single script that can:

- train LeNet-5 only,
- train ZFNet only,
- train both models and compare them, or
- reload saved histories and regenerate the comparison plot.
"""

import argparse
import json
import os
from typing import TypeAlias

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

History: TypeAlias = dict[str, list[float]]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_history(history: History, path: str) -> None:
    """Persist a training history dictionary to disk.

    Histories are written as indented JSON so they can be inspected manually,
    reused by `--compare`, and loaded inside the notebook without retraining.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] History saved to {path}")


def load_history(path: str) -> History:
    """Load a previously saved history dictionary from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def run_lenet() -> History:
    """Run the full LeNet-5 experiment and save its outputs.

    The function intentionally mirrors the project workflow used in the report:
    data loading, model construction, training, plotting, sample predictions,
    weight export, and metric-history export.
    """
    print("\n" + "#" * 60)
    print("#  LENET-5 TRAINING")
    print("#" * 60)

    # Build the STL-10 input pipeline sized for the classic LeNet-5 network.
    train_loader, test_loader = get_stl10_loaders(batch_size=128)

    # Train the model and collect per-epoch metrics for later reporting.
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

    # Save visual artifacts from the main scripted run.
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

    # Export the final checkpoint and the JSON history used by `--compare`.
    torch.save(model.state_dict(), "checkpoints/lenet5_stl10.pth")
    print("[INFO] Weights saved to checkpoints/lenet5_stl10.pth")

    os.makedirs("outputs", exist_ok=True)
    save_history(history, "outputs/lenet5_history.json")

    return history


def run_zfnet() -> History:
    """Run the full ZFNet experiment and save its outputs.

    ZFNet keeps more spatial detail than LeNet-5, so the scripted workflow uses
    a separate loader configuration, model construction path, and history file.
    """
    print("\n" + "#" * 60)
    print("#  ZFNET TRAINING")
    print("#" * 60)

    # Build the STL-10 pipeline sized for the compact ZFNet-style model.
    train_loader, test_loader = get_stl10_loaders_zfnet(batch_size=96, input_size=64)

    # Train the deeper model and collect per-epoch metrics.
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

    # Save visual artifacts from the main scripted run.
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

    # Export both the final checkpoint and the reusable history JSON.
    torch.save(model.state_dict(), "checkpoints/zfnet_stl10_final.pth")
    print("[INFO] Final weights saved to checkpoints/zfnet_stl10_final.pth")

    os.makedirs("outputs", exist_ok=True)
    save_history(history, "outputs/zfnet_history.json")

    return history


def print_summary(lenet_history: History, zfnet_history: History) -> None:
    """Print a compact side-by-side summary for both architectures."""
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


def main() -> None:
    """Parse CLI flags and dispatch the requested experiment workflow."""
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

    # Compare-only mode is intentionally lightweight: it reuses the histories
    # written by earlier runs and regenerates the shared comparison plot.
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
        # The default project workflow trains both models in sequence and then
        # compares them using the histories collected during the same session.
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
