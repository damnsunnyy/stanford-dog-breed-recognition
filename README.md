# Stanford Dogs breed recognition

Fine-grained classification of 120 dog breeds using the official Stanford Dogs train/test lists. The current model is an ImageNet-pretrained EfficientNet-B0.

## Current result

The validation-selected checkpoint reached **82.50% validation accuracy** at epoch 2. On the untouched official 8,580-image test partition it reached **82.26% top-1 accuracy**, **82.11% macro precision**, **81.84% macro recall**, and **81.22% macro F1**. The earlier ResNet-18 baseline scored 66.08% test accuracy. This is a single-seed result; it is not a controlled architecture-only comparison or a state-of-the-art claim.

## Reproduce

Install dependencies from `requirements.txt`, then run `python train_stanford_dogs.py`. The script uses seed 42, a stratified 10% validation subset of the official training partition, 224-pixel images, batch size 32, RandAugment, label smoothing, AdamW, and a two-epoch cosine schedule. It selects the best validation checkpoint and evaluates the official test split once.

The trained checkpoint is `stanford_dogs_efficientnet_b0.pt`. The run also writes the confusion matrix and per-breed metrics CSV files. `research.ipynb` contains the data preparation, training, evaluation, and single-image prediction workflow.

## Dataset

The dataset archive is not included. Obtain Stanford Dogs and place the image folders and official `train_list.mat` and `test_list.mat` under `stanford_dogs_dataset/` as expected by the script. The held-out test list is used only for final scoring.
