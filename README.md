# Stanford Dog Breed Recognition

A fine-grained dog breed classification project using the Stanford Dogs dataset and an ImageNet-pretrained ResNet-18.

## Project contents

- `research.ipynb`: data preparation, training, evaluation, and prediction workflow.
- `stanford_dogs_resnet18.pt`: trained 120-class checkpoint (43 MB).
- `final_research_paper.pdf`: two-column research paper.
- `final_research_paper.docx` and `.md`: editable paper versions.
- `stanford_dogs_confusion_matrix.csv` and `stanford_dogs_per_breed_metrics.csv`: held-out test results.
- `stanford_dogs_lowest_recall.png` and `stanford_dogs_confusion_heatmap.png`: result visualizations.
- `plot_stanford_dogs_results.py`: script used to make the visualizations.
- `research_plan.md`: project notes and literature review.
- `papers/README.md`: bibliography source notes.

## Dataset

The dataset archive is not included. Obtain Stanford Dogs from the [official dataset page](http://vision.stanford.edu/aditya86/ImageNetDogs/), and place/extract it as described in the notebook. The official `train_list.mat` and `test_list.mat` split files are required for reproducing the reported benchmark evaluation. Follow the dataset's terms when downloading and using it.

## Reported result

The saved ResNet-18 checkpoint achieved **66.08% top-1 accuracy** and **64.92% macro F1** on the official 8,580-image test partition. This is a single-run baseline, not a state-of-the-art claim. The model was selected using a validation subset drawn from the official training partition.

## Environment

Install dependencies from `requirements.txt`, then run the notebook from top to bottom. Training and inference device selection are handled in the notebook. The checkpoint can be loaded in PyTorch; use the notebook's prediction helper for image preprocessing and class-name handling.
