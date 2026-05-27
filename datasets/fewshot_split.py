from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CLASS_ALIASES = {
    "angry": ["anger"],
    "anger": ["angry"],
    "happy": ["happiness"],
    "happiness": ["happy"],
    "sad": ["sadness"],
    "sadness": ["sad"],
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    class_name: str
    label: int


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    for root in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(resolved)


def list_class_images(root: Path) -> tuple[list[str], dict[str, list[Path]]]:
    class_dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    class_names = sorted(p.name for p in class_dirs)
    images_by_class: dict[str, list[Path]] = {}
    for class_name in class_names:
        class_dir = root / class_name
        images = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            images_by_class[class_name] = images
    class_names = sorted(images_by_class)
    return class_names, images_by_class


def records_from_class_dir(root: Path, class_to_idx: dict[str, int]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for class_name, label in class_to_idx.items():
        class_dir = root / class_name
        if not class_dir.exists():
            for alias in CLASS_ALIASES.get(class_name, []):
                alias_dir = root / alias
                if alias_dir.exists():
                    class_dir = alias_dir
                    break
        if not class_dir.exists():
            continue
        images = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        records.extend(ImageRecord(path=p.resolve(), class_name=class_name, label=label) for p in images)
    return records


def split_direct_class_folders(
    source: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord], list[str]]:
    class_names, images_by_class = list_class_images(source)
    rng = random.Random(seed)
    train_pool: list[ImageRecord] = []
    val_records: list[ImageRecord] = []
    test_records: list[ImageRecord] = []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        images = list(images_by_class[class_name])
        rng.shuffle(images)
        n_total = len(images)
        n_test = max(1, int(round(n_total * test_ratio))) if n_total >= 3 else 0
        n_val = max(1, int(round(n_total * val_ratio))) if n_total - n_test >= 3 else 0
        test_images = images[:n_test]
        val_images = images[n_test : n_test + n_val]
        train_images = images[n_test + n_val :]
        label = class_to_idx[class_name]
        train_pool.extend(ImageRecord(p.resolve(), class_name, label) for p in train_images)
        val_records.extend(ImageRecord(p.resolve(), class_name, label) for p in val_images)
        test_records.extend(ImageRecord(p.resolve(), class_name, label) for p in test_images)

    return train_pool, val_records, test_records, class_names


def split_predefined_folders(source: Path) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord], list[str]]:
    train_root = source / "train"
    val_root = source / "val"
    test_root = source / "test"
    if not train_root.exists() or not test_root.exists():
        raise ValueError("Predefined layout requires at least train/ and test/ directories.")

    class_names, _ = list_class_images(train_root)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_records = records_from_class_dir(train_root, class_to_idx)
    val_records = records_from_class_dir(val_root, class_to_idx) if val_root.exists() else []
    test_records = records_from_class_dir(test_root, class_to_idx)
    return train_records, val_records, test_records, class_names


def holdout_validation(
    records: list[ImageRecord],
    class_names: list[str],
    val_ratio: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    rng = random.Random(seed)
    train_records: list[ImageRecord] = []
    val_records: list[ImageRecord] = []
    for class_name in class_names:
        class_records = [r for r in records if r.class_name == class_name]
        rng.shuffle(class_records)
        n_val = max(1, int(round(len(class_records) * val_ratio))) if len(class_records) >= 3 else 0
        val_records.extend(class_records[:n_val])
        train_records.extend(class_records[n_val:])
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def sample_shot(records: list[ImageRecord], class_names: list[str], shot: int | str, seed: int) -> list[ImageRecord]:
    if str(shot).lower() == "full":
        return records
    shot_int = int(shot)
    rng = random.Random(seed)
    sampled: list[ImageRecord] = []
    for class_name in class_names:
        class_records = [r for r in records if r.class_name == class_name]
        rng.shuffle(class_records)
        if len(class_records) < shot_int:
            raise ValueError(f"Class '{class_name}' has {len(class_records)} train images, fewer than {shot_int}.")
        sampled.extend(class_records[:shot_int])
    rng.shuffle(sampled)
    return sampled


def write_records(records: list[ImageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [{"path": portable_path(r.path), "label": r.label, "class_name": r.class_name} for r in records]
        if records
        else [],
        columns=["path", "label", "class_name"],
    )
    df.to_csv(path, index=False)


def write_class_names(class_names: list[str], path: Path) -> None:
    path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def make_fewshot_splits(
    source: Path,
    out: Path,
    shots: list[str],
    seeds: list[int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.20,
) -> None:
    source = source.resolve()
    out = out.resolve()
    predefined = (source / "train").exists() and (source / "test").exists()

    for seed in seeds:
        if predefined:
            train_pool, val_records, test_records, class_names = split_predefined_folders(source)
            if not val_records:
                train_pool, val_records = holdout_validation(train_pool, class_names, val_ratio, seed)
        else:
            train_pool, val_records, test_records, class_names = split_direct_class_folders(
                source, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
            )

        for shot in shots:
            train_records = sample_shot(train_pool, class_names, shot, seed)
            split_name = f"{shot}shot_seed{seed}" if str(shot).lower() != "full" else f"full_seed{seed}"
            split_dir = out / split_name
            write_records(train_records, split_dir / "train.csv")
            write_records(val_records, split_dir / "val.csv")
            write_records(test_records, split_dir / "test.csv")
            write_class_names(class_names, split_dir / "class_names.json")
