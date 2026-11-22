# Industrial Surface Defect Inspection Platform

Individual academic project, Deep Learning, Semester V, B.Tech CSE (AIML), UPES.
Author: Aadrika Verma.

Classifies and localises six steel-surface defect types (crazing, inclusion, patches,
pitted_surface, rolled-in_scale, scratches) on the NEU-DET dataset.

## Models
- Classifier (main): ResNet18, ImageNet-pretrained, all layers fine-tuned (`resnet18_seed42.pt`)
- Classifier (baseline): 4-block CNN trained from scratch, 98,406 parameters (`baseline_cnn_seed*.pt`)
- Detector (prototype): Faster R-CNN ResNet50-FPN, COCO-pretrained, 7-way head (`fasterrcnn_seed42.pt`)

## Data and split
NEU-DET (Kaggle copy by kaustubhdikshit of the NEU surface defect database).
1,800 grayscale images, 200x200, 4,189 Pascal VOC boxes. One exact duplicate removed (patches_105).
Stratified 70/15/15 split, seed 42: 1,259 train / 270 val / 270 test (see `splits.csv`).
The test set was evaluated once, after model selection on validation.

## Training setup
| | Baseline CNN | ResNet18 | Faster R-CNN |
|---|---|---|---|
| Loss | CrossEntropy | CrossEntropy | sum of 4 detection losses |
| Optimiser | Adam, lr 1e-3 | Adam, lr 3e-4 | SGD, lr 0.01, momentum 0.9, weight decay 1e-4, StepLR |
| Batch / epochs | 32 / 30 | 32 / 30 | 8 / up to 12 |
| Best epoch chosen by | validation loss | validation loss | validation mAP@0.5 |

Augmentation: random horizontal and vertical flips (classifiers), horizontal flip with boxes (detector),
training data only. Seeds 42 / 7 / 123 for classifiers, 42 for the detector.

## Results (test set, evaluated once)
The full pipeline was run twice on the identical split (GPU runs are not bit-identical).
The CSV files and saved weights belong to Run 1. The notebook's saved outputs are from the later Run 2.

| Model | Run 1 | Run 2 |
|---|---|---|
| ResNet18, 3 seeds, macro F1 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Baseline CNN, 3 seeds, macro F1 | 0.9828 +/- 0.0093 (errors 2/7/5 of 270) | 0.9853 +/- 0.0074 (errors 2/6/4 of 270) |
| Faster R-CNN, seed 42, mAP@0.5 | 0.782 | 0.796 |
| Faster R-CNN, mAP@[.5:.95] | 0.430 | 0.439 |
| Faster R-CNN precision / recall (score 0.5, IoU 0.5) | 0.567 / 0.862 | 0.611 / 0.855 |

ResNet18 scoring 1.0000 shows it reaches the ceiling of this small, clean benchmark. It does not
show the system is ready for a factory line.

## Limitations
- No "normal" class: the model always names a defect type.
- Small, controlled benchmark; results will not transfer directly to production images.
- Detection is a prototype (one seed). Crazing and rolled-in scale are the weakest classes.
- 123 of 1,800 images contain more than one defect type; the folder label is used for classification.
- No large hyperparameter search was run.

## Ethics, citation and AI-assistance disclosure
- The model is meant to support human inspectors, not replace them: a missed defect is the costly error.
- Dataset: NEU surface defect database. Please cite the original authors, K. Song and Y. Yan,
  "A noise robust method based on completed local binary patterns for hot-rolled steel strip
  surface defects", Applied Surface Science 285 (2013) 858-864 (check the dataset page for the
  citation it requests).
- An AI assistant was used as a mentor for explaining and debugging code. All experiments were run
  and measured by the author on Kaggle.

## Inference
1. Load the checkpoint (`torch.load`) and rebuild the model: ResNet18 with a 1-channel first
   convolution and a 6-way final layer.
2. Convert the image to 200x200 grayscale, scale to [0, 1].
3. Normalise with ImageNet mean 0.449 and std 0.226 (ResNet18). The baseline uses the
   training-set mean 0.5035 and std 0.2129 (see `preprocess_config.json`).
4. Apply softmax to the output scores to get class confidences.
5. For boxes, run Faster R-CNN (6 classes + background, input upscaled to 400 px) and keep
   predictions with score >= 0.5.

## Files
- `01-dataset-audit.ipynb`: the full pipeline (audit, split, baseline, ResNet18, detection, platform).
  Final models were trained in Cell B (Cell 20 needs internet access and is kept as the first attempt).
- `splits.csv`, `annotations.csv`, `preprocess_config.json`, `exp_val_results.csv`, `test_results.csv`
- `requirements.txt`
- Model checkpoints are not stored in this repository (the Faster R-CNN file is about 160 MB).

## Reproducibility
Python 3.13, PyTorch 2.11, torchvision 0.26, Kaggle Tesla T4. Seeds 42 / 7 / 123.
Models were retrained after a Kaggle session reset on the identical split.
