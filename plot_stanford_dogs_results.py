"""Create paper-ready per-breed recall and confusion-matrix plots."""
from pathlib import Path

import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path.cwd()
with open(ROOT / "stanford_dogs_per_breed_metrics.csv", newline="") as metrics_file:
    metrics = list(csv.DictReader(metrics_file))
confusion = np.loadtxt(ROOT / "stanford_dogs_confusion_matrix.csv", delimiter=",", dtype=int)
checkpoint_path = ROOT / "stanford_dogs_efficientnet_b0.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
class_names = checkpoint["class_names"]

# The 20 breeds with the lowest recall make class-level weaknesses legible.
worst = sorted(metrics, key=lambda row: float(row["recall"]))[:20]
fig, ax = plt.subplots(figsize=(11, 8))
labels = [row.get("class", row.get("breed", "")).split("-", 1)[-1].replace("_", " ") for row in worst]
recalls = [float(row["recall"]) for row in worst]
ax.barh(labels, recalls, color="#4C78A8")
ax.set_xlim(0, 1)
ax.set_xlabel("Recall on official test split")
ax.set_title("20 Stanford Dogs breeds with lowest recall")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(ROOT / "stanford_dogs_lowest_recall.png", dpi=220)
plt.close(fig)

# Show normalized confusions among breeds with the most test mistakes.
errors = confusion.sum(axis=1) - np.diag(confusion)
selected = np.argsort(errors)[-12:][::-1]
subset = confusion[np.ix_(selected, selected)].astype(float)
row_sums = confusion.sum(axis=1)[selected, None]
subset = np.divide(subset, row_sums, out=np.zeros_like(subset), where=row_sums != 0)
short_names = [class_names[i].split("-", 1)[-1].replace("_", " ") for i in selected]
fig, ax = plt.subplots(figsize=(11, 9))
image = ax.imshow(subset, cmap="magma", vmin=0, vmax=max(0.1, float(subset.max())))
ax.set_xticks(range(len(short_names)), short_names, rotation=65, ha="right")
ax.set_yticks(range(len(short_names)), short_names)
ax.set_xlabel("Predicted breed")
ax.set_ylabel("Actual breed")
ax.set_title("Normalized confusions for 12 breeds with most errors")
fig.colorbar(image, ax=ax, label="Fraction of actual-breed test images")
fig.tight_layout()
fig.savefig(ROOT / "stanford_dogs_confusion_heatmap.png", dpi=220)
plt.close(fig)
print("Saved stanford_dogs_lowest_recall.png and stanford_dogs_confusion_heatmap.png")
