from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.facial_dataset import FacialAffectDataset, build_transforms
from models import create_model
from utils.metrics import classification_metrics, confusion, save_predictions
from utils.visualization import plot_confusion_matrix, save_mask_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PDM-Net or baseline checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/pdmnet.yaml"))
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--visualize-masks", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[str], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    paths: list[str] = []
    first_images = None
    first_masks = None
    first_labels = None

    for batch in tqdm(loader, desc="test"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs["logits"].argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        paths.extend(batch["path"])
        if first_images is None and "masks" in outputs:
            first_images = images.detach().cpu()
            first_masks = outputs["masks"].detach().cpu()
            first_labels = labels.detach().cpu()

    return y_true, y_pred, paths, first_images, first_masks, first_labels


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint.get("cfg") or load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    class_names = checkpoint.get("class_names")
    if class_names is None:
        class_file = args.split / "class_names.json"
        class_names = json.loads(class_file.read_text(encoding="utf-8"))

    image_size = int(cfg["dataset"].get("image_size", 224))
    batch_size = int(cfg["training"].get("batch_size", 32))
    num_workers = int(cfg["dataset"].get("num_workers", 4))
    test_csv = args.split / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing {test_csv}.")

    dataset = FacialAffectDataset(test_csv, transform=build_transforms(image_size=image_size, train=False))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg["model"].get("name", "pdmnet")
    model = create_model(model_name, num_classes=len(class_names), cfg=cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    y_true, y_pred, paths, first_images, first_masks, first_labels = run_eval(model, loader, device)
    metrics = classification_metrics(y_true, y_pred)
    cm = confusion(y_true, y_pred, len(class_names))

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_predictions(paths, y_true, y_pred, class_names, output_dir / "predictions.csv")
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    if args.visualize_masks and first_images is not None and first_masks is not None and first_labels is not None:
        preds = torch.tensor(y_pred[: first_images.shape[0]])
        save_mask_grid(first_images, first_masks, first_labels, preds, class_names, output_dir / "pdm_masks.png")

    print(json.dumps(metrics, indent=2))
    print(f"Wrote evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
