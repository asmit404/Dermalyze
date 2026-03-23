"""
EfficientNet-B5 model module for skin lesion classification.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B5_Weights

from .efficientnet import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    SkinLesionClassifier,
    get_loss_function,
)


class SkinLesionClassifierB5(SkinLesionClassifier):
    """EfficientNet-B5 based classifier for skin lesion classification."""

    def _create_backbone(self, pretrained: bool) -> tuple[nn.Module, int]:
        """Create the EfficientNet-B5 backbone."""
        weights = EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b5(weights=weights)
        feature_dim = 2048

        backbone.classifier = nn.Identity()
        return backbone, feature_dim


def create_model_b5(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    use_gradient_checkpointing: bool = False,
) -> SkinLesionClassifierB5:
    """Factory function to create an EfficientNet-B5 skin lesion classifier."""
    return SkinLesionClassifierB5(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


__all__ = [
    "SkinLesionClassifierB5",
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "create_model_b5",
    "get_loss_function",
]
