"""
ConvNeXt-Tiny Model for Skin Lesion Classification.

This module provides the model architecture based on ConvNeXt-Tiny with
an MLP classification head for the HAM10000 seven-class classification task.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

from .efficientnet import FocalLoss, LabelSmoothingCrossEntropy


class SkinLesionConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Tiny based classifier for skin lesion classification.

    This model uses a pretrained ConvNeXt-Tiny backbone with a custom
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
        super().__init__()

        self.num_classes = num_classes
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.backbone, self.feature_dim = self._create_backbone(pretrained)

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

        self._initialize_classifier()

        if freeze_backbone:
            self._freeze_backbone()
        elif freeze_layers is not None:
            self._freeze_layers(freeze_layers)

    def _create_backbone(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Create the ConvNeXt-Tiny backbone."""
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        feature_dim = 768

        # Keep ConvNeXt pooled features and flatten to [B, 768].
        # Using Identity here returns [B, 768, 1, 1] in some torchvision versions,
        # which breaks the first Linear layer of our custom classifier.
        backbone.classifier = nn.Flatten(1)

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
        """Freeze the first N feature layers of the backbone."""
        layers = list(self.backbone.features.children())
        for layer in layers[:num_layers]:
            for param in layer.parameters():
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
        """Forward pass through ConvNeXt backbone and custom head."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            features = self.backbone(x)
        elif self.use_gradient_checkpointing and self.training:
            features = self._forward_with_checkpoint(x)
        else:
            features = self.backbone(x)

        if features.ndim > 2:
            features = torch.flatten(features, 1)

        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    use_gradient_checkpointing: bool = False,
) -> SkinLesionConvNeXtClassifier:
    """Factory function to create a ConvNeXt skin lesion classifier."""
    return SkinLesionConvNeXtClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


def get_loss_function(
    loss_type: Literal["cross_entropy", "focal", "label_smoothing"] = "focal",
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Get the loss function for training."""
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal":
        alpha = focal_alpha if focal_alpha is not None else class_weights
        return FocalLoss(alpha=alpha, gamma=focal_gamma)
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(
            smoothing=label_smoothing, weight=class_weights
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
