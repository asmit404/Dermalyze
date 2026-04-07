"""Multi-input model wrapper for image + metadata fusion inference."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiInputClassifier(nn.Module):
    """Fuse image features and metadata features before classification."""

    def __init__(
        self,
        image_model: nn.Module,
        metadata_dim: int,
        num_classes: int = 7,
        metadata_hidden_dim: int = 64,
        fusion_hidden_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.image_model = image_model
        self.metadata_dim = metadata_dim
        self.num_classes = num_classes
        self.image_feature_dim = self._get_image_feature_dim()

        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, metadata_hidden_dim),
            nn.BatchNorm1d(metadata_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(metadata_hidden_dim, metadata_hidden_dim),
            nn.BatchNorm1d(metadata_hidden_dim),
            nn.ReLU(inplace=True),
        )

        fusion_input_dim = self.image_feature_dim + metadata_hidden_dim
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def _get_image_feature_dim(self) -> int:
        if hasattr(self.image_model, "classifier"):
            classifier = self.image_model.classifier
            if isinstance(classifier, nn.Sequential):
                for module in classifier:
                    if isinstance(module, nn.Linear):
                        return module.in_features
            elif isinstance(classifier, nn.Linear):
                return classifier.in_features

        if hasattr(self.image_model, "feature_dim"):
            return int(self.image_model.feature_dim)

        return 2048

    def _extract_image_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.image_model, "backbone"):
            features = self.image_model.backbone(x)
            if features.ndim == 4:
                features = F.adaptive_avg_pool2d(features, 1)
                features = torch.flatten(features, 1)
            return features

        if hasattr(self.image_model, "forward_features"):
            features = self.image_model.forward_features(x)
            return torch.flatten(features, 1)

        raise AttributeError(
            "Image model must expose either 'backbone' or 'forward_features'"
        )

    def _compute_image_logits(self, image_features: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute image-only logits from the wrapped image model head when available."""
        classifier = getattr(self.image_model, "classifier", None)
        if not isinstance(classifier, nn.Module):
            return None
        return classifier(image_features)

    def forward(
        self,
        image: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image_features = self._extract_image_features(image)
        image_logits = self._compute_image_logits(image_features)

        if metadata is None:
            batch_size = image.shape[0]
            metadata = torch.zeros(
                batch_size,
                self.metadata_dim,
                device=image.device,
                dtype=image.dtype,
            )

        metadata_features = self.metadata_mlp(metadata)
        fused_features = torch.cat([image_features, metadata_features], dim=1)
        fusion_logits = self.fusion_classifier(fused_features)

        if image_logits is None:
            return fusion_logits

        if image_logits.shape != fusion_logits.shape:
            raise RuntimeError(
                "Image and fusion logits shape mismatch: "
                f"image={tuple(image_logits.shape)}, fusion={tuple(fusion_logits.shape)}"
            )

        return fusion_logits + image_logits
