"""EfficientNet classifier variants used by the inference service."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
)


_EFFICIENTNET_VARIANTS = {
    "efficientnet_b0": ("efficientnet_b0", 1280, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b1": ("efficientnet_b1", 1280, EfficientNet_B1_Weights.IMAGENET1K_V1),
    "efficientnet_b2": ("efficientnet_b2", 1408, EfficientNet_B2_Weights.IMAGENET1K_V1),
    "efficientnet_b3": ("efficientnet_b3", 1536, EfficientNet_B3_Weights.IMAGENET1K_V1),
    "efficientnet_b4": ("efficientnet_b4", 1792, EfficientNet_B4_Weights.IMAGENET1K_V1),
    "efficientnet_b5": ("efficientnet_b5", 2048, EfficientNet_B5_Weights.IMAGENET1K_V1),
    "efficientnet_b6": ("efficientnet_b6", 2304, EfficientNet_B6_Weights.IMAGENET1K_V1),
    "efficientnet_b7": ("efficientnet_b7", 2560, EfficientNet_B7_Weights.IMAGENET1K_V1),
    "efficientnetv2_s": ("efficientnet_v2_s", 1280, EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    "efficientnetv2_m": ("efficientnet_v2_m", 1280, EfficientNet_V2_M_Weights.IMAGENET1K_V1),
    "efficientnetv2_l": ("efficientnet_v2_l", 1280, EfficientNet_V2_L_Weights.IMAGENET1K_V1),
}


def normalize_efficientnet_variant(variant: str) -> str:
    """Normalize common EfficientNet backbone aliases to canonical keys."""
    normalized = str(variant or "efficientnet_b0").strip().lower().replace("-", "_")
    alias_map = {
        "efficientnet": "efficientnet_b0",
        "efficientnet_v2_s": "efficientnetv2_s",
        "efficientnet_v2_m": "efficientnetv2_m",
        "efficientnet_v2_l": "efficientnetv2_l",
    }
    return alias_map.get(normalized, normalized)


class SkinLesionClassifier(nn.Module):
    """EfficientNet family model with a custom MLP head for 7-way classification."""

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        backbone_variant: str = "efficientnet_b0",
    ):
        super().__init__()

        variant_key = normalize_efficientnet_variant(backbone_variant)
        if variant_key not in _EFFICIENTNET_VARIANTS:
            raise ValueError(
                f"Unsupported EfficientNet variant: {backbone_variant!r}. "
                f"Supported variants: {sorted(_EFFICIENTNET_VARIANTS.keys())}"
            )

        model_name, feature_dim, pretrained_weights = _EFFICIENTNET_VARIANTS[variant_key]
        weights = pretrained_weights if pretrained else None
        self.backbone = getattr(models, model_name)(weights=weights)

        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
