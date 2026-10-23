# Skin Lesion Image Classification

An academic computer-vision project exploring binary classification of dermoscopic images into benign and malignant classes. It compares a compact custom CNN with ResNet-18 and studies image augmentation and transfer learning using PyTorch.

## Methods

- Custom convolutional neural network baseline and ResNet-18 classifier
- Training-only flips, rotations, and color/contrast augmentation
- ImageNet channel normalization and AdamW optimization
- Held-out evaluation with accuracy, macro precision/recall, confusion matrix, and ROC-AUC when both classes are present

## Dataset

The dataset is not included. Arrange the images in class-named folders:

```text
data/melanoma-cancer-dataset/
├── train/
│   ├── Benign/
│   └── Malignant/
└── test/
    ├── Benign/
    └── Malignant/
```

Install the dependencies and run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py --data-dir data/melanoma-cancer-dataset --model resnet18 --pretrained
```

Use `--model cnn` for the custom CNN baseline. ResNet-18 pretrained weights are downloaded by torchvision when `--pretrained` is set. The script evaluates the held-out test split once after training and writes the checkpoint and metrics to `runs/`.

## Scope

This is an educational experiment, not a diagnostic tool. Results depend on the dataset and training setup; the models are not intended for clinical use.
