from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.facial_dataset import FacialAffectDataset


class EmotionCsvDataset(Dataset):
    def __init__(self, csv_path: Path, processor: AutoImageProcessor) -> None:
        self.resolver = FacialAffectDataset(csv_path, transform=None)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.resolver.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.resolver.samples[index]
        image = Image.open(sample.path).convert("RGB")
        return {"image": image, "label": sample.label, "path": str(sample.path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a FER-pretrained Hugging Face image classifier.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", default="models_hf/vit-face-expression")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_class_names(split: Path) -> list[str]:
    class_file = split / "class_names.json"
    if class_file.exists():
        return json.loads(class_file.read_text(encoding="utf-8"))
    df = pd.read_csv(split / "train.csv")
    return [name for _, name in sorted(set(zip(df["label"], df["class_name"])))]


def collate_fn(processor: AutoImageProcessor):
    def _collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        images = [item["image"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        paths = [item["path"] for item in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = labels
        inputs["paths"] = paths
        return inputs

    return _collate


def freeze_visual_backbone(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if "classifier" not in name and "score" not in name:
            parameter.requires_grad = False
    print("Frozen pretrained encoder; training classifier head only.")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        labels = batch.pop("labels").to(device)
        batch.pop("paths", None)
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    label_smoothing: float,
) -> float:
    model.train()
    total = 0.0
    steps = 0
    amp = device.type == "cuda"
    for batch in tqdm(loader, desc="train", leave=False):
        labels = batch.pop("labels").to(device)
        batch.pop("paths", None)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(**batch).logits
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    class_names = load_class_names(args.split)
    id2label = {idx: name for idx, name in enumerate(class_names)}
    label2id = {name: idx for idx, name in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if args.freeze_backbone:
        freeze_visual_backbone(model)

    train_dataset = EmotionCsvDataset(args.split / "train.csv", processor)
    val_dataset = EmotionCsvDataset(args.split / "val.csv", processor)
    test_dataset = EmotionCsvDataset(args.split / "test.csv", processor)
    collate = collate_fn(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    safe_model_name = str(args.model).replace("/", "_").replace("\\", "_")
    output_dir = args.output_dir or Path("outputs") / f"hf_finetuned_{safe_model_name}_{args.split.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = -1.0
    history = []
    best_dir = output_dir / "best_model"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args.label_smoothing)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model = AutoModelForImageClassification.from_pretrained(best_dir).to(device)
    test_metrics = evaluate(model, test_loader, device)
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print("Best model:", best_dir)
    print(json.dumps({"test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
