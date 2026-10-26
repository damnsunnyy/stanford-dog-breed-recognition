# Stanford Dogs research paper: working plan

## Working title

**Fine-Grained Dog Breed Classification on Stanford Dogs: A Reproducible Transfer-Learning Baseline and Error Analysis**

Treat this as a working title. The paper should describe the actual model and measurements after the experiments are complete; do not claim a new architecture unless we implement and test one.

## Dataset found in the workspace

`archive.zip` is a valid Stanford Dogs archive with 20,580 images and 20,580 annotation files (41,160 entries total). It contains the `images/Images/...` and `annotations/Annotation/...` folders. The notebook now extracts `archive.zip` locally rather than downloading a second copy through KaggleHub. The archive does not include Stanford's official train/test lists. I downloaded the official `lists.tar` (470 KB) separately; it contains `train_list.mat` and `test_list.mat` and can support a benchmark-comparable split.

## Research question to investigate

How does an ImageNet-pretrained ResNet transfer-learning baseline perform across all 120 dog breeds when evaluated with a fixed split, and which breed pairs and image conditions account for its errors? A useful extension is to compare a part-attention/fine-grained model against the same baseline, reporting accuracy alongside compute cost.

This is a project question, not a verified field-wide research gap. The original ResNet-18 baseline reached 66.08% test accuracy. A follow-up ImageNet-pretrained EfficientNet-B0 run reached 82.50% validation accuracy and 82.26% official test accuracy (macro F1 81.22%) after two epochs. This is one seed and one training setup, not a state-of-the-art claim or a controlled architecture-only comparison.

## Literature review synthesis (first reading)

Khosla et al. define the benchmark around 120 visually similar dog breeds and explain why it is difficult: low differences between breeds, large variation within a breed, occlusion, pose, and cluttered human-made backgrounds. Each image has a class label and bounding box. The benchmark's fixed split has 12,000 training and 8,580 test images [1].

Early fine-grained work focuses on discriminative local evidence. Yao et al. combine discriminative patch mining with randomized decision trees [3]. Sermanet et al. apply recurrent visual attention directly to Stanford Dogs and report 76.3% mean accuracy for their three-resolution, one-glimpse model, compared with 75.5% for their GoogLeNet baseline; their attention model does not use dog bounding boxes [4]. Bilinear CNNs model local pairwise feature interactions by pooling outer products from two feature streams [5]. FCAN learns class-discriminative glimpses and reports 87.3% for its fine-tuned ResNet-50 baseline, 88.5% with one attention region, and 88.9% with two on Stanford Dogs [7]. NTS-Net uses cooperating Navigator, Teacher, and Scrutinizer modules to propose and classify informative regions on other fine-grained benchmarks [8]; it is relevant methodologically, but its cited experiments do not include Stanford Dogs.

The transformer literature offers a more recent direction. Dosovitskiy et al. show that a transformer over image patches can transfer well from large-scale pretraining [9]. Conde and Turgutlu add multi-stage attention-guided cropping/erasing and report 93.2% accuracy on Stanford Dogs, using the official 12,000/8,580 split [10]. TransFG selects informative patches from transformer attention and adds a contrastive objective; its paper reports 92.3% accuracy on Stanford Dogs and compares against a standard ViT at 91.7% [11]. Cui et al. study a different pipeline that combines features from four CNNs, feature selection, and an SVM on Stanford Dogs [12]. These results are useful literature context, but they are not directly comparable to the current EfficientNet-B0 run without matching the split, input resolution, pretraining, and training protocol.

## Candidate contribution and gap to verify

The literature already contains strong attention and transformer results above 90% on this benchmark. The ResNet-18 remains a transparent baseline, while EfficientNet-B0 provides the current stronger result. A defensible student project can measure the accuracy/compute trade-off and explain errors across all 120 breeds using one fixed split, macro metrics, a per-breed confusion analysis, and visual examples. The current search has not established that this exact evaluation is absent from prior work; do not claim novelty until the related-work search is expanded and the comparison protocol of those papers is checked.

## Methodology and reproducibility record

1. Use the official Stanford train/test lists, preserving the split and class mapping. Keep a validation subset drawn only from the training partition for model selection.
2. Train and compare ImageNet-pretrained ResNet-18 and EfficientNet-B0 models. Save the seed, software versions, preprocessing, augmentations, optimizer, learning rate, epoch count, and hardware.
3. Report top-1 accuracy, macro precision, macro recall, macro F1, per-breed scores, and a confusion matrix on the held-out official test partition. Record training time and inference throughput.
4. If compute permits, compare one fine-grained attention or transformer model using the exact same split and report the same metrics. Do not frame a result as an improvement without repeated or carefully controlled runs.
5. Use Grad-CAM or attention visualizations only as qualitative evidence, not proof that the model reasons correctly.
6. Write the final Abstract, Introduction, Findings, Methodology, Results, Discussion, and Conclusion from the recorded run. The initial paper draft is in `paper_draft.md`; fill in the author name and adjust the section order to match the instructor’s format.

## Starter bibliography (IEEE-style details; verify page/author metadata when importing to Zotero)

[1] A. Khosla, N. Jayadevaprakash, B. Yao, and L. Fei-Fei, “Novel Dataset for Fine-Grained Image Categorization: Stanford Dogs,” in *First Workshop on Fine-Grained Visual Categorization, IEEE CVPR*, 2011. [Paper](https://people.csail.mit.edu/khosla/papers/fgvc2011.pdf).

[2] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A Large-Scale Hierarchical Image Database,” in *IEEE CVPR*, 2009. [Paper](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf).

[3] B. Yao, A. Khosla, and L. Fei-Fei, “Combining Randomization and Discrimination for Fine-Grained Image Categorization,” in *IEEE CVPR*, 2011. [Paper](https://svl.stanford.edu/assets/publications/pdfs/YaoKhoslaFeiFei_CVPR2011.pdf).

[4] P. Sermanet, A. Frome, and E. Real, “Attention for Fine-Grained Categorization,” *ICLR Workshop*, 2015. [arXiv:1412.7054](https://arxiv.org/abs/1412.7054). This work directly studies attention on Stanford Dogs.

[5] T.-Y. Lin, A. RoyChowdhury, and S. Maji, “Bilinear CNN Models for Fine-Grained Visual Recognition,” in *IEEE ICCV*, pp. 1449–1457, 2015. [Paper](https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html).

[6] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in *IEEE CVPR*, pp. 770–778, 2016. [Paper](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). Cite this for the ResNet architecture, not as dog-breed evidence.

[7] X. Liu, T. Xia, J. Wang, Y. Yang, F. Zhou, and Y. Lin, “Fully Convolutional Attention Networks for Fine-Grained Recognition,” in *IEEE CVPR*, 2016. [arXiv:1603.06765](https://arxiv.org/abs/1603.06765). The paper evaluates on Stanford Dogs and other fine-grained datasets.

[8] Z. Yang, T. Luo, D. Wang, Z. Hu, J. Gao, and L. Wang, “Learning to Navigate for Fine-Grained Classification,” in *European Conference on Computer Vision (ECCV)*, pp. 420–435, 2018. [Paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Ze_Yang_Learning_to_Navigate_ECCV_2018_paper.html). NTS-Net studies informative-region selection.

[9] A. Dosovitskiy et al., “An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale,” in *International Conference on Learning Representations (ICLR)*, 2021. [OpenReview](https://openreview.net/forum?id=YicbFdNTTy).

[10] M. V. Conde and K. Turgutlu, “Exploring Vision Transformers for Fine-Grained Classification,” in *IEEE CVPR Workshops (FGVC8)*, 2021. [arXiv:2106.10587](https://arxiv.org/abs/2106.10587). Includes Stanford Dogs and attention-guided augmentations.

[11] J. He, J.-N. Chen, S. Liu, A. Kortylewski, C. Yang, Y. Bai, and C. Wang, “TransFG: A Transformer Architecture for Fine-Grained Recognition,” in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 36, no. 1, pp. 852–860, 2022. [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/19967).

[12] Y. Cui, B. Tang, G. Wu, L. Li, X. Zhang, Z. Du, and W. Zhao, “Classification of Dog Breeds Using Convolutional Neural Network Models and Support Vector Machine,” *Bioengineering*, vol. 11, no. 11, 1157, 2024. [DOI: 10.3390/bioengineering11111157](https://doi.org/10.3390/bioengineering11111157). It reports a multi-CNN feature/SVM pipeline on Stanford Dogs; inspect its split and experimental details before comparing its accuracy.

## Citation practice

Cite the original source for each claim, cite the dataset paper when describing the benchmark, and distinguish reported values from results reproduced in this project. Download and read the papers before writing the literature review; abstracts alone are not enough to support detailed claims.

## Completed experiment outputs

- ResNet-18 baseline: best validation accuracy 65.50%; official test accuracy 66.08%, macro precision 68.24%, macro recall 65.57%, macro F1 64.92%.
- EfficientNet-B0 follow-up: best validation accuracy 82.50% at epoch 2; official test accuracy 82.26%, macro precision 82.11%, macro recall 81.84%, macro F1 81.22%.
- The EfficientNet-B0 run used two epochs and a validation-selected checkpoint. Results are from one seed; additional seeds are needed to estimate run-to-run variation.
- Per-class scores: `stanford_dogs_per_breed_metrics.csv`. Confusion counts: `stanford_dogs_confusion_matrix.csv`.
- Paper charts: `stanford_dogs_lowest_recall.png` and `stanford_dogs_confusion_heatmap.png`; generation code: `plot_stanford_dogs_results.py`.
- EfficientNet-B0 run environment: Python 3.14.6, PyTorch 2.14.1, torchvision 0.29.1, SciPy 1.18.1, Apple MPS. Seed 42. Batch size 32, 2 epochs, 224-pixel input, RandAugment, label smoothing, AdamW, cosine decay.
- The run is one seed with no uncertainty intervals or latency benchmark. The two model runs use different recipes, so they do not isolate the effect of architecture.
