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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.facial_dataset import FacialAffectDataset


class EmotionCsvDataset(Dataset):
    def __init__(self, csv_path: Path) -> None:
        self.resolver = FacialAffectDataset(csv_path, transform=None)

    def __len__(self) -> int:
        return len(self.resolver.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.resolver.samples[index]
        image = Image.open(sample.path).convert("RGB")
        return {"image": image, "label": sample.label, "path": str(sample.path)}


class PDMViT(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        temperature: float = 0.1,
        fusion_alpha: float = 0.7,
        mask_scale: float = 5.0,
        dropout: float = 0.2,
        use_token_mask: bool = True,
        use_prototype_branch: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)
        hidden_size = int(getattr(config, "hidden_size"))
        self.temperature = temperature
        self.fusion_alpha = fusion_alpha
        self.mask_scale = mask_scale
        self.use_token_mask = use_token_mask
        self.use_prototype_branch = use_prototype_branch
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.prototypes = nn.Parameter(torch.randn(num_classes, hidden_size) * 0.02)

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        encoded = self.encoder(pixel_values=pixel_values, return_dict=True)
        tokens = encoded.last_hidden_state
        cls = self.dropout(tokens[:, 0])
        patch_tokens = tokens[:, 1:]

        logits_g = self.classifier(cls)
        local = F.normalize(patch_tokens, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        similarity = torch.einsum("bnh,kh->bkn", local, prototypes)
        if self.use_token_mask:
            masks = torch.sigmoid(self.mask_scale * similarity)
        else:
            masks = torch.ones_like(similarity)

        if self.training and labels is not None:
            selected = masks[torch.arange(pixel_values.shape[0], device=pixel_values.device), labels]
            masked = patch_tokens * selected.unsqueeze(-1)
            z_m = self.dropout(masked.mean(dim=1))
            logits_p = F.normalize(z_m, dim=-1) @ prototypes.t() / self.temperature
            selected_mask = selected
        else:
            masked = patch_tokens.unsqueeze(1) * masks.unsqueeze(-1)
            z_m = F.normalize(masked.mean(dim=2), dim=-1)
            logits_p = (z_m * prototypes.unsqueeze(0)).sum(dim=-1) / self.temperature
            selected_mask = None

        if self.use_prototype_branch:
            logits = logits_g + self.fusion_alpha * logits_p
        else:
            logits_p = torch.zeros_like(logits_g)
            logits = logits_g
        return {"logits": logits, "logits_g": logits_g, "logits_p": logits_p, "masks": masks, "selected_mask": selected_mask}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PDM-ViT with prototype-guided token masking.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", default="models_hf/vit-face-expression")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--lambda-consistency", type=float, default=0.05)
    parser.add_argument("--lambda-mask", type=float, default=0.0005)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--fusion-alpha", type=float, default=0.7)
    parser.add_argument("--mask-scale", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--disable-token-mask", action="store_true", help="Ablation: replace prototype-guided masks with all-one masks.")
    parser.add_argument("--disable-prototype-branch", action="store_true", help="Ablation: use only the global CLS classifier.")
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
    def _collate(batch: list[dict]) -> dict:
        images = [item["image"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        paths = [item["path"] for item in batch]
        inputs = processor(images=images, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"], "labels": labels, "paths": paths}

    return _collate


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(logits_a, dim=1)
    log_b = F.log_softmax(logits_b, dim=1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    return 0.5 * (F.kl_div(log_a, prob_b, reduction="batchmean") + F.kl_div(log_b, prob_a, reduction="batchmean"))


def compute_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    loss_g = F.cross_entropy(outputs["logits_g"], labels, label_smoothing=args.label_smoothing)
    loss_f = F.cross_entropy(outputs["logits"], labels, label_smoothing=args.label_smoothing)
    loss = loss_g + loss_f
    if not args.disable_prototype_branch:
        loss_p = F.cross_entropy(outputs["logits_p"], labels, label_smoothing=args.label_smoothing)
        loss = loss + loss_p
    if args.lambda_consistency > 0 and not args.disable_prototype_branch:
        loss = loss + args.lambda_consistency * symmetric_kl(outputs["logits_g"], outputs["logits_p"])
    selected_mask = outputs.get("selected_mask")
    if args.lambda_mask > 0 and selected_mask is not None:
        loss = loss + args.lambda_mask * selected_mask.mean()
    return loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        labels = batch["labels"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        logits = model(pixel_values=pixel_values)["logits"]
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    total = 0.0
    steps = 0
    amp = device.type == "cuda"
    for batch in tqdm(loader, desc="train", leave=False):
        labels = batch["labels"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = compute_loss(outputs, labels, args)
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
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = PDMViT(
        model_name_or_path=args.model,
        num_classes=len(class_names),
        temperature=args.temperature,
        fusion_alpha=args.fusion_alpha,
        mask_scale=args.mask_scale,
        dropout=args.dropout,
        use_token_mask=not args.disable_token_mask,
        use_prototype_branch=not args.disable_prototype_branch,
    )
    if args.freeze_encoder:
        model.freeze_encoder()
        print("Frozen FER-pretrained ViT encoder; training PDM token head only.")

    collate = collate_fn(processor)
    train_loader = DataLoader(
        EmotionCsvDataset(args.split / "train.csv"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        EmotionCsvDataset(args.split / "val.csv"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    test_loader = DataLoader(
        EmotionCsvDataset(args.split / "test.csv"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    head_params = list(model.classifier.parameters()) + [model.prototypes]
    head_ids = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    safe_model_name = str(args.model).replace("/", "_").replace("\\", "_")
    output_dir = args.output_dir or Path("outputs") / f"pdm_vit_{safe_model_name}_{args.split.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    history = []
    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics, "lr": scheduler.get_last_lr()}
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "class_names": class_names, "args": vars(args)}, best_path)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device)
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print("Best checkpoint:", best_path)
    print(json.dumps({"test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
