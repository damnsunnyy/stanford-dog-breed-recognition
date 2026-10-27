"""Evaluate the validation-selected EfficientNet-B0 checkpoint on official test."""
import csv
import numpy as np
import torch
from scipy.io import loadmat
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).parent
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
IMAGE_ROOT = ROOT / "stanford_dogs_dataset" / "images" / "Images"
DATASET_DIR = ROOT / "stanford_dogs_dataset"
SIZE = 224
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(SIZE), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
required = [IMAGE_ROOT, DATASET_DIR / "train_list.mat", DATASET_DIR / "test_list.mat"]
missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
if missing:
    raise SystemExit(
        "Stanford Dogs data is missing. Run `python download_stanford_dogs.py` "
        "to download and prepare it. Missing: " + ", ".join(missing)
    )
ds = datasets.ImageFolder(IMAGE_ROOT, transform=transform)
values = loadmat(DATASET_DIR / "test_list.mat")["file_list"].reshape(-1)
paths=[]
for v in values:
    while isinstance(v, np.ndarray) and v.size == 1: v=v.item()
    paths.append(str(v).replace("\\", "/").removeprefix("Images/"))
lookup={Path(p).relative_to(IMAGE_ROOT).as_posix():i for i,(p,_) in enumerate(ds.samples)}
indices=[lookup[p] for p in paths]
loader=DataLoader(Subset(ds,indices),batch_size=32,shuffle=False,num_workers=0)
ckpt=torch.load(ROOT/"stanford_dogs_efficientnet_b0.pt",map_location=DEVICE,weights_only=False)
model=models.efficientnet_b0(weights=None)
model.classifier[1]=nn.Linear(model.classifier[1].in_features,120)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()
cm=np.zeros((120,120),dtype=np.int64)
with torch.no_grad():
    for x,y in loader:
        pred=model(x.to(DEVICE)).argmax(1).cpu().numpy()
        np.add.at(cm,(y.numpy(),pred),1)
tp=np.diag(cm).astype(float)
precision=tp/np.maximum(cm.sum(0),1)
recall=tp/np.maximum(cm.sum(1),1)
f1=2*precision*recall/np.maximum(precision+recall,1e-12)
print(f"Selected epoch {ckpt.get('selected_epoch', 'unknown')} validation accuracy: {ckpt['val_accuracy']:.4f}")
print(f"Official test: accuracy={tp.sum()/cm.sum():.4f} macro_precision={precision.mean():.4f} macro_recall={recall.mean():.4f} macro_f1={f1.mean():.4f}")
np.savetxt(ROOT/"stanford_dogs_confusion_matrix.csv",cm,delimiter=",",fmt="%d")
with open(ROOT/"stanford_dogs_per_breed_metrics.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["class","precision","recall","f1","support"])
    for i,name in enumerate(ds.classes): w.writerow([name,f"{precision[i]:.6f}",f"{recall[i]:.6f}",f"{f1[i]:.6f}",int(cm[i].sum())])
