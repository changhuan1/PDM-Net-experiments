from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str


def build_transforms(image_size: int = 224, train: bool = True) -> Callable:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.RandomAffine(degrees=8, translate=(0.03, 0.03), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def denormalize_image(tensor):
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


class FacialAffectDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform: Callable | None = None) -> None:
        self.csv_path = Path(csv_path)
        self.root = self.csv_path.parent
        self.transform = transform
        df = pd.read_csv(self.csv_path)
        required = {"path", "label", "class_name"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

        self.samples = [
            Sample(path=self._resolve_path(row["path"]), label=int(row["label"]), class_name=str(row["class_name"]))
            for _, row in df.iterrows()
        ]
        self.class_names = self._infer_class_names(self.samples)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        candidate = (self.root / path).resolve()
        if candidate.exists():
            return candidate
        return path.resolve()

    @staticmethod
    def _infer_class_names(samples: Iterable[Sample]) -> list[str]:
        pairs = sorted({(sample.label, sample.class_name) for sample in samples})
        return [name for _, name in pairs]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample.label,
            "path": str(sample.path),
            "class_name": sample.class_name,
        }
