"""Loading of the fixed classification CSV shipped with the service."""
from __future__ import annotations

import csv

import torch

LABEL_COLUMN = "label"


def load_classification_csv(path: str):
    """Return (X float32 [n, d], y int64 [n], feature column names).

    Rows are kept in file order; shuffling for training happens per epoch in
    the trainer.  Parsing is fully deterministic.
    """
    xs: list[list[float]] = []
    ys: list[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or LABEL_COLUMN not in reader.fieldnames:
            raise ValueError(f"{path}: CSV must contain a '{LABEL_COLUMN}' column")
        feature_cols = [c for c in reader.fieldnames if c != LABEL_COLUMN]
        if not feature_cols:
            raise ValueError(f"{path}: CSV has no feature columns")
        for row in reader:
            xs.append([float(row[c]) for c in feature_cols])
            ys.append(int(row[LABEL_COLUMN]))
    if not xs:
        raise ValueError(f"{path}: dataset is empty")
    X = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y, feature_cols
