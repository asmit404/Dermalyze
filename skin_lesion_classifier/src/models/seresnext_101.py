"""
SE-ResNeXt-101 model module for skin lesion classification.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .efficientnet import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    get_loss_function,
)


class SkinLesionSEResNeXt101Classifier(nn.Module):
    """
    SE-ResNeXt-101 based classifier for skin lesion classification.

    This model uses a pretrained SE-ResNeXt-101 (32x4d) backbone with a custom
    classification head optimized for the HAM10000 dataset.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        freeze_layers: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
    ):
        """
        Initialize the skin lesion classifier with SE-ResNeXt-101 backbone.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze all backbone layers
            freeze_layers: Number of backbone layers to freeze (from the start)
            use_gradient_checkpointing: Whether to use gradient checkpointing (saves memory)
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Load pretrained backbone (SE-ResNeXt-101)
        self.backbone, self.feature_dim = self._create_backbone(pretrained)

        # Custom classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        # Initialize classifier weights
        self._initialize_classifier()

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        elif freeze_layers is not None:
            self._freeze_layers(freeze_layers)

    def _create_backbone(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Create the SE-ResNeXt-101 backbone."""
        backbone = timm.create_model(
            'seresnext101_32x4d',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='',  # Remove global pooling
        )
        feature_dim = 2048

        return backbone, feature_dim

    def _initialize_classifier(self) -> None:
        """Initialize classifier weights using Kaiming initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _freeze_layers(self, num_layers: int) -> None:
        """
        Freeze the first N layers of the backbone.

        For SE-ResNeXt, this freezes stages in order: stem, layer1, layer2, layer3, layer4.
        """
        stage_modules = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]

        for i, module in enumerate(stage_modules[:num_layers]):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (for fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    @torch.jit.unused
    def _forward_with_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Forward backbone with gradient checkpointing (Python-only path)."""
        return checkpoint(self.backbone, x, use_reentrant=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            features = self.backbone(x)
        elif self.use_gradient_checkpointing and self.training:
            features = self._forward_with_checkpoint(x)
        else:
            features = self.backbone(x)

        # Global average pooling if features are spatial
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, 1)
            features = torch.flatten(features, 1)

        # Classify
        logits = self.classifier(features)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Predicted class indices of shape (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model_seresnext101(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    use_gradient_checkpointing: bool = False,
) -> SkinLesionSEResNeXt101Classifier:
    """Factory function to create a SE-ResNeXt-101 skin lesion classifier."""
    return SkinLesionSEResNeXt101Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


__all__ = [
    "SkinLesionSEResNeXt101Classifier",
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "create_model_seresnext101",
    "get_loss_function",
]
