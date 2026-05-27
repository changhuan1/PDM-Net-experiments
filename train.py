from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.facial_dataset import FacialAffectDataset, build_transforms
from losses.losses import compute_loss
from models import create_model
from utils.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PDM-Net or a baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/pdmnet.yaml"))
    parser.add_argument("--split", type=Path, required=True, help="Directory containing train.csv/val.csv/test.csv.")
    parser.add_argument("--model", type=str, default=None, help="Override model.name in the YAML config.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_class_names(split_dir: Path, train_dataset: FacialAffectDataset, cfg: dict) -> list[str]:
    configured = cfg.get("dataset", {}).get("class_names") or []
    if configured:
        return list(configured)
    class_file = split_dir / "class_names.json"
    if class_file.exists():
        return json.loads(class_file.read_text(encoding="utf-8"))
    return train_dataset.class_names


def make_loader(csv_path: Path, image_size: int, batch_size: int, num_workers: int, train: bool) -> DataLoader:
    dataset = FacialAffectDataset(csv_path, transform=build_transforms(image_size=image_size, train=train))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def freeze_backbone_if_requested(model: torch.nn.Module, cfg: dict) -> None:
    if not bool(cfg.get("training", {}).get("freeze_backbone", False)):
        return
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    print("Backbone frozen: training classifier/prototype layers only.")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs["logits"].argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return classification_metrics(y_true, y_pred)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    loss_cfg: dict,
    amp: bool,
) -> dict[str, float]:
    model.train()
    running: dict[str, float] = {}
    steps = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            outputs = model(images, labels)
            loss, logs = compute_loss(outputs, labels, loss_cfg)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        steps += 1
        for key, value in logs.items():
            running[key] = running.get(key, 0.0) + value
        pbar.set_postfix(loss=running["loss"] / steps)

    return {key: value / max(steps, 1) for key, value in running.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    seed = int(cfg["training"].get("seed", 0))
    set_seed(seed)

    split_dir = args.split
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}. Run scripts/make_splits.py first.")

    image_size = int(cfg["dataset"].get("image_size", 224))
    batch_size = int(cfg["training"].get("batch_size", 32))
    num_workers = int(cfg["dataset"].get("num_workers", 4))
    train_loader = make_loader(train_csv, image_size, batch_size, num_workers, train=True)
    has_val = val_csv.exists() and sum(1 for _ in val_csv.open("r", encoding="utf-8")) > 1
    val_loader = make_loader(val_csv, image_size, batch_size, num_workers, train=False) if has_val else None
    train_dataset = train_loader.dataset
    class_names = load_class_names(split_dir, train_dataset, cfg)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg["model"].get("name", "pdmnet")
    model = create_model(model_name, num_classes=num_classes, cfg=cfg["model"]).to(device)
    freeze_backbone_if_requested(model, cfg)

    optimizer_name = cfg["training"].get("optimizer", "adamw").lower()
    lr = float(cfg["training"].get("learning_rate", 3e-4))
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(cfg["training"].get("epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    split_name = split_dir.name
    default_run = f"{model_name}_{split_name}"
    run_name = cfg.get("output", {}).get("run_name") or default_run
    if run_name == "pdmnet_run":
        run_name = default_run
    elif "{split}" in run_name:
        run_name = run_name.format(split=split_name, model=model_name)
    elif not run_name.endswith(split_name):
        run_name = f"{run_name}_{split_name}"
    output_root = Path(cfg.get("output", {}).get("root", "outputs"))
    output_dir = args.output_dir or (output_root / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        train_logs = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg.get("loss", {}), amp)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else {}
        score = val_metrics.get("macro_f1", -train_logs.get("loss", 0.0))
        row = {"epoch": epoch, "train": train_logs, "val": val_metrics, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        print(json.dumps(row, indent=2))

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                output_dir / "best.pt",
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
