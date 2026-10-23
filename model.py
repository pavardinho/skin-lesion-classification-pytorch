"""Models used for the skin-lesion image-classification project."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SimpleCNN(nn.Module):
    """Compact CNN baseline for RGB image classification."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(name: str, num_classes: int, pretrained: bool = False):
    if name == "cnn":
        if pretrained:
            raise ValueError("The custom CNN does not use pretrained weights.")
        return SimpleCNN(num_classes)

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {name}")
