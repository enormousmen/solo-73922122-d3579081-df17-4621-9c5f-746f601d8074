#!/usr/bin/env python3
"""Generate the fixed classification dataset shipped at data/classification.csv.

The CSV is committed to the repo; this script only documents how it was
produced (fixed seed => identical output).  3 Gaussian blobs in 2D,
600 samples, columns: feature_1, feature_2, label.
"""
import csv
import random
from pathlib import Path

SEED = 20240101
SAMPLES_PER_CLASS = 200
CENTERS = [(2.0, 2.0), (-2.0, 2.0), (0.0, -2.5)]
OUT = Path(__file__).resolve().parent.parent / "data" / "classification.csv"


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for label, (cx, cy) in enumerate(CENTERS):
        for _ in range(SAMPLES_PER_CLASS):
            x = rng.gauss(cx, 1.0)
            y = rng.gauss(cy, 1.0)
            rows.append((round(x, 6), round(y, 6), label))
    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_1", "feature_2", "label"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
