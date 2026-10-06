import torch
import torch.nn as nn
from torchvision import models


class ImageBaselineResNet(nn.Module):
    """
    Image-Only Disease Classification Baseline using Pretrained ResNet18.
    
    Extracts high-level spatial visual representations via Global Average Pooling
    before classification.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)

        # Retain feature extractor up to avgpool
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool
        self.feature_dim = base_model.fc.in_features  # 512 for ResNet18

        # Classification head with regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional visual embedding vector."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning class logits."""
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits
