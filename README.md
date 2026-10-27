# Stanford Dogs breed recognition

Fine-grained classification of 120 dog breeds using the official Stanford Dogs train/test lists. The current model is an ImageNet-pretrained EfficientNet-B0.

## Current result

The validation-selected checkpoint reached **82.50% validation accuracy** at epoch 2. On the untouched official 8,580-image test partition it reached **82.26% top-1 accuracy**, **82.11% macro precision**, **81.84% macro recall**, and **81.22% macro F1**. The earlier ResNet-18 baseline scored 66.08% test accuracy. This is a single-seed result; it is not a controlled architecture-only comparison or a state-of-the-art claim.

## Reproduce

Install dependencies from `requirements.txt`, then run `python train_stanford_dogs.py`. The script uses seed 42, a stratified 10% validation subset of the official training partition, 224-pixel images, batch size 32, RandAugment, label smoothing, AdamW, and a two-epoch cosine schedule. It selects the best validation checkpoint and evaluates the official test split once.

The trained checkpoint is `stanford_dogs_efficientnet_b0.pt`. The run also writes the confusion matrix and per-breed metrics CSV files. `research.ipynb` contains the data preparation, training, evaluation, and single-image prediction workflow.

## Dataset

The dataset files are not included in this repository. From the repository directory, run:

```bash
python download_stanford_dogs.py
```

This downloads the official Stanford `images.tar` and `lists.tar` archives and prepares the expected `stanford_dogs_dataset/images/Images/`, `train_list.mat`, and `test_list.mat` paths. The image archive is about 778 MB, so the download can take a while. Interrupted downloads are retried and resumed when the server supports it; rerun the command if it still fails.

If Stanford's server cannot be reached, download [images.tar](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar) and [lists.tar](http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar) in a browser and save both in the repository's `.stanford_dogs_downloads/` folder (create it if needed), keeping the original filenames. Then rerun `python download_stanford_dogs.py`. After preparation, run `python evaluate_stanford_dogs.py`, or run the downloader before `python train_stanford_dogs.py`.

The downloader uses the archives published on the [Stanford Dogs dataset page](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html). The held-out test list is used only for final scoring.
