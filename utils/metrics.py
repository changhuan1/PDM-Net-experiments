from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def confusion(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    labels = list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def save_predictions(
    paths: list[str],
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    out_path: str | Path,
) -> None:
    rows = []
    for path, truth, pred in zip(paths, y_true, y_pred):
        rows.append(
            {
                "path": path,
                "label": truth,
                "label_name": class_names[truth] if truth < len(class_names) else str(truth),
                "prediction": pred,
                "prediction_name": class_names[pred] if pred < len(class_names) else str(pred),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
