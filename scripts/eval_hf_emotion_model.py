from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.facial_dataset import FacialAffectDataset


CANONICAL = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Hugging Face facial-emotion model on a split CSV.")
    parser.add_argument("--split", type=Path, required=True, help="Split directory containing test.csv.")
    parser.add_argument("--csv-name", default="test.csv")
    parser.add_argument("--model", default="trpakov/vit-face-expression")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def canonicalize(label: str) -> str:
    label = str(label).lower().strip()
    return CANONICAL.get(label, label)


def load_class_names(split: Path, df: pd.DataFrame) -> list[str]:
    class_file = split / "class_names.json"
    if class_file.exists():
        return json.loads(class_file.read_text(encoding="utf-8"))
    return [name for _, name in sorted(set(zip(df["label"], df["class_name"])))]


def main() -> None:
    args = parse_args()
    csv_path = args.split / args.csv_name
    df = pd.read_csv(csv_path)
    class_names = load_class_names(args.split, df)
    class_to_idx = {canonicalize(name): idx for idx, name in enumerate(class_names)}

    resolver = FacialAffectDataset(csv_path, transform=None)
    paths = [sample.path for sample in resolver.samples]
    y_true = [int(sample.label) for sample in resolver.samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).to(device)
    model.eval()

    id2label = {int(k): canonicalize(v) for k, v in model.config.id2label.items()}
    y_pred: list[int] = []
    rows = []

    for start in tqdm(range(0, len(paths), args.batch_size), desc="hf-eval"):
        batch_paths = paths[start : start + args.batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = logits.softmax(dim=1)
            pred_ids = probs.argmax(dim=1).cpu().tolist()
            confs = probs.max(dim=1).values.cpu().tolist()

        for path, pred_id, conf in zip(batch_paths, pred_ids, confs):
            pred_name = id2label[pred_id]
            pred_label = class_to_idx.get(pred_name, -1)
            y_pred.append(pred_label)
            rows.append({"path": str(path), "prediction_name": pred_name, "prediction": pred_label, "confidence": conf})

    metrics = {
        "model": args.model,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    output = args.output or (Path("outputs") / f"hf_{args.model.replace('/', '_')}_{args.split.name}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1).to_csv(output / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
