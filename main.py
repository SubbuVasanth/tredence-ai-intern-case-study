import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from models.prunable_net import PrunableNet
from utils.trainer import run_training


# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)


def get_loaders(batch_size: int = 128):
    """Download CIFAR-10 and return train / test DataLoaders."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),         
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True
    )
    test_loader  = torch.utils.data.DataLoader(
        testset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


def plot_gate_distribution(gate_vals: np.ndarray, lambd: float, path: str):
    """Save a clean histogram of final gate values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gate_vals, bins=80, color="#2563EB", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(x=0.1, color="#EF4444", linestyle="--", linewidth=1.5,
               label="Prune threshold (0.1)")
    ax.set_title(f"Final Gate Value Distribution  (λ = {lambd})", fontsize=13, pad=12)
    ax.set_xlabel("Gate Value", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved {path}")


def plot_tradeoff(results: list, path: str):
    """Save a dual-axis chart showing accuracy and sparsity vs lambda."""
    lambdas    = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    sparsities = [r[2] for r in results]
    x = range(len(lambdas))

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color_acc  = "#2563EB"
    color_spar = "#16A34A"

    ax1.plot(x, accuracies, "o-", color=color_acc,  linewidth=2, markersize=8, label="Test Accuracy (%)")
    ax1.set_ylabel("Test Accuracy (%)", color=color_acc, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(x, sparsities, "s--", color=color_spar, linewidth=2, markersize=8, label="Sparsity Level (%)")
    ax2.set_ylabel("Sparsity Level (%)", color=color_spar, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_spar)
    ax2.set_ylim(0, 100)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in lambdas], fontsize=10)
    ax1.set_xlabel("Lambda (λ)", fontsize=11)
    ax1.set_title("Accuracy vs Sparsity Trade-off", fontsize=13, pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved {path}")


# ── Main

def main():
    device = torch.device("cpu") 
    print(f"Using device: {device}\n")

    train_loader, test_loader = get_loaders(batch_size=128)



    lambdas = [1e-4, 1e-3, 1e-2]
    results = []

    for lambd in lambdas:
        print(f"\n{'='*60}")
        print(f"  Training with λ = {lambd}")
        print(f"{'='*60}")
        model = PrunableNet().to(device)
        acc, sparsity, gate_vals = run_training(
            model, train_loader, test_loader, lambd, device, epochs=15, warmup_epochs=3
        )
        results.append((lambd, acc, sparsity, gate_vals))
        print(f"\n  Result → Accuracy: {acc:.2f}%  |  Sparsity: {sparsity:.2f}%")

    print("\nGenerating plots …")

    best = results[-1]
    plot_gate_distribution(best[3], best[0], "plots/gate_distribution.png")
    plot_tradeoff([(r[0], r[1], r[2]) for r in results], "plots/sparsity_tradeoff.png")


    print("\n" + "="*52)
    print(f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>15}")
    print("-"*52)
    for lambd, acc, sparsity, _ in results:
        print(f"{lambd:<12} {acc:>14.2f}% {sparsity:>14.2f}%")
    print("="*52)


if __name__ == "__main__":
    main()