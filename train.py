"""Train and evaluate a CNN or ResNet-18 on a folder-based image dataset."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/melanoma-cancer-dataset"))
    parser.add_argument("--model", choices=("cnn", "resnet18"), default="resnet18")
    parser.add_argument("--pretrained", action="store_true", help="Initialize ResNet-18 with ImageNet weights")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(data_dir: Path, batch_size: int, workers: int):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected class folders under {train_dir} and {test_dir}. "
            "See the README for the required dataset structure."
        )

    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalization,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalization,
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    if train_data.classes != test_data.classes:
        raise ValueError("Train and test folders must contain the same class names.")

    loader_options = {"batch_size": batch_size, "num_workers": workers}
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    return train_loader, test_loader, train_data.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return {"loss": total_loss / total_samples, "accuracy": correct / total_samples}


@torch.inference_mode()
def evaluate(model, loader, device, class_names):
    model.eval()
    all_labels, all_predictions, positive_scores = [], [], []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        total_samples += labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
        if len(class_names) == 2:
            positive_scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())

    metrics = {
        "test_loss": total_loss / total_samples,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision_macro": precision_score(all_labels, all_predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(
            all_labels, all_predictions, labels=list(range(len(class_names)))
        ).tolist(),
        "class_names": class_names,
    }
    if len(class_names) == 2 and len(set(all_labels)) == 2:
        metrics["roc_auc"] = roc_auc_score(all_labels, positive_scores)
    return metrics


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs and batch size must be positive; learning rate must be greater than zero.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, class_names = make_loaders(
        args.data_dir, args.batch_size, args.workers
    )
    model = build_model(args.model, len(class_names), args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history.append(metrics)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={metrics['loss']:.4f} train_accuracy={metrics['accuracy']:.4f}"
        )

    # The held-out test split is evaluated once, after the final training epoch.
    test_metrics = evaluate(model, test_loader, device, class_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "architecture": args.model, "class_names": class_names},
        args.output_dir / "model.pt",
    )
    report = {"model": args.model, "epochs": args.epochs, "training_history": history, "test": test_metrics}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
