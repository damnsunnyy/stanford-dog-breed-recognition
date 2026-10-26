"""Train and evaluate an ImageNet-pretrained EfficientNet-B0 on Stanford Dogs."""
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 2
ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "stanford_dogs_dataset"
IMAGE_ROOT = DATASET_DIR / "images" / "Images"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def read_list(path):
    vals = loadmat(path)["file_list"].reshape(-1)
    out = []
    for v in vals:
        while isinstance(v, np.ndarray) and v.size == 1:
            v = v.item()
        name = str(v).replace("\\", "/")
        out.append(name.removeprefix("Images/"))
    return out

train_aug = datasets.ImageFolder(IMAGE_ROOT, transform=train_transform)
eval_ds = datasets.ImageFolder(IMAGE_ROOT, transform=eval_transform)
if len(train_aug.classes) != 120 or train_aug.classes != eval_ds.classes:
    raise RuntimeError("Expected matching 120-class ImageFolder datasets")
lookup = {Path(p).relative_to(IMAGE_ROOT).as_posix(): i for i, (p, _) in enumerate(train_aug.samples)}
official_train = [lookup[p] for p in read_list(DATASET_DIR / "train_list.mat")]
official_test = [lookup[p] for p in read_list(DATASET_DIR / "test_list.mat")]
rng = random.Random(SEED)
per_class = [[] for _ in train_aug.classes]
for idx in official_train:
    per_class[train_aug.targets[idx]].append(idx)
fit, valid = [], []
for indices in per_class:
    rng.shuffle(indices)
    n_valid = max(1, round(len(indices) * 0.1))
    valid.extend(indices[:n_valid])
    fit.extend(indices[n_valid:])
loader_args = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=DEVICE.type == "cuda")
train_loader = DataLoader(Subset(train_aug, fit), shuffle=True, **loader_args)
valid_loader = DataLoader(Subset(eval_ds, valid), shuffle=False, **loader_args)
test_loader = DataLoader(Subset(eval_ds, official_test), shuffle=False, **loader_args)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 120)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW([
    {"params": model.features.parameters(), "lr": 1.5e-4},
    {"params": model.classifier.parameters(), "lr": 1e-3},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def epoch(loader, training):
    model.train(training)
    loss_sum = torch.zeros((), device=DEVICE)
    correct = torch.zeros((), device=DEVICE)
    count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.detach() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum()
            count += labels.size(0)
    return (loss_sum / count).item(), (correct / count).item()

print(f"Device={DEVICE}; classes={len(train_aug.classes)}; fit={len(fit)}; validation={len(valid)}; test={len(official_test)}", flush=True)
print(f"Starting EfficientNet-B0, input={IMAGE_SIZE}, batch={BATCH_SIZE}, epochs={EPOCHS}", flush=True)
best = 0.0
start = time.time()
for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc = epoch(train_loader, True)
    va_loss, va_acc = epoch(valid_loader, False)
    scheduler.step()
    print(f"Epoch {ep:02d}/{EPOCHS} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f} elapsed_min={(time.time()-start)/60:.1f}", flush=True)
    if va_acc > best:
        best = va_acc
        torch.save({"model_state_dict": model.state_dict(), "class_names": train_aug.classes,
                    "image_size": IMAGE_SIZE, "architecture": "efficientnet_b0",
                    "val_accuracy": best, "selected_epoch": ep,
                    "training_epochs": EPOCHS, "seed": SEED}, ROOT / "stanford_dogs_efficientnet_b0.pt")

ckpt = torch.load(ROOT / "stanford_dogs_efficientnet_b0.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
confusion = np.zeros((120, 120), dtype=np.int64)
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images.to(DEVICE)).argmax(1).cpu().numpy()
        np.add.at(confusion, (labels.numpy(), preds), 1)
tp = np.diag(confusion).astype(float)
precision = tp / np.maximum(confusion.sum(0), 1)
recall = tp / np.maximum(confusion.sum(1), 1)
f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
accuracy = tp.sum() / confusion.sum()
print(f"BEST_VALID={best:.4f}; OFFICIAL_TEST accuracy={accuracy:.4f} macro_precision={precision.mean():.4f} macro_recall={recall.mean():.4f} macro_f1={f1.mean():.4f}", flush=True)
np.savetxt(ROOT / "stanford_dogs_confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
with open(ROOT / "stanford_dogs_per_breed_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "precision", "recall", "f1", "support"])
    for i, name in enumerate(train_aug.classes):
        w.writerow([name, f"{precision[i]:.6f}", f"{recall[i]:.6f}", f"{f1[i]:.6f}", int(confusion[i].sum())])
