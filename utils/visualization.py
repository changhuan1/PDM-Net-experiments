from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

from datasets.facial_dataset import denormalize_image


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_lowshot_curve(results: dict[str, dict[str, float]], out_path: str | Path) -> None:
    shots = list(results.keys())
    acc = [results[s]["accuracy"] for s in shots]
    f1 = [results[s]["macro_f1"] for s in shots]
    plt.figure(figsize=(7, 4))
    plt.plot(shots, acc, marker="o", label="Accuracy")
    plt.plot(shots, f1, marker="s", label="Macro-F1")
    plt.xlabel("Shot setting")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_mask_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    class_names: list[str],
    out_path: str | Path,
    max_items: int = 8,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(images.shape[0], max_items)
    fig, axes = plt.subplots(count, 3, figsize=(9, 3 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(count):
        image = denormalize_image(images[idx].detach().cpu()).permute(1, 2, 0).numpy()
        pred = int(preds[idx].detach().cpu())
        label = int(labels[idx].detach().cpu())
        mask = masks[idx, pred].detach().cpu().unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=image.shape[:2], mode="bilinear", align_corners=False)
        mask_np = mask.squeeze().numpy()

        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f"Image / GT: {class_names[label]}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(mask_np, cmap="magma", vmin=0, vmax=1)
        axes[idx, 1].set_title(f"PDM mask: {class_names[pred]}")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(image)
        axes[idx, 2].imshow(mask_np, cmap="magma", alpha=0.45, vmin=0, vmax=1)
        axes[idx, 2].set_title(f"Overlay / Pred: {class_names[pred]}")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
