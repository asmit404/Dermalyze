"""
Multi-input model for skin lesion classification with image and metadata fusion.

This module provides a wrapper that combines any image backbone with metadata
(age, sex, anatomical site) to improve classification performance.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiInputClassifier(nn.Module):
    """
    Multi-input classifier that fuses image features with patient metadata.

    Architecture:
        1. Image backbone (any CNN) extracts image features
        2. Metadata MLP processes tabular data (age, sex, localization)
        3. Features are concatenated
        4. Final classifier predicts lesion type
    """

    def __init__(
        self,
        image_model: nn.Module,
        metadata_dim: int,
        num_classes: int = 7,
        metadata_hidden_dim: int = 64,
        fusion_hidden_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize the multi-input classifier.

        Args:
            image_model: Base image model (should output features, not logits)
            metadata_dim: Dimension of encoded metadata features
            num_classes: Number of output classes
            metadata_hidden_dim: Hidden dimension for metadata MLP
            fusion_hidden_dim: Hidden dimension for fusion layer
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        self.image_model = image_model
        self.metadata_dim = metadata_dim
        self.num_classes = num_classes

        # Get feature dimension from image model
        # The image model should have a 'classifier' attribute we can inspect
        self.image_feature_dim = self._get_image_feature_dim()

        # Metadata MLP
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, metadata_hidden_dim),
            nn.BatchNorm1d(metadata_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(metadata_hidden_dim, metadata_hidden_dim),
            nn.BatchNorm1d(metadata_hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Fusion classifier
        fusion_input_dim = self.image_feature_dim + metadata_hidden_dim
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        # Initialize new layers
        self._initialize_weights()

    def _get_image_feature_dim(self) -> int:
        """
        Infer the feature dimension from the image model.

        The image model should have a classifier that starts with a Linear layer.
        """
        # Try to get from classifier
        if hasattr(self.image_model, "classifier"):
            classifier = self.image_model.classifier
            if isinstance(classifier, nn.Sequential):
                for module in classifier:
                    if isinstance(module, nn.Linear):
                        return module.in_features
            elif isinstance(classifier, nn.Linear):
                return classifier.in_features

        # Fallback: get from feature_dim attribute if available
        if hasattr(self.image_model, "feature_dim"):
            return self.image_model.feature_dim

        # Common default for most models
        return 2048

    def _initialize_weights(self) -> None:
        """Initialize weights for new layers."""
        for module in [self.metadata_mlp, self.fusion_classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def _extract_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the image model without the final classifier.

        Args:
            x: Input image tensor (batch_size, 3, H, W)

        Returns:
            Image features (batch_size, image_feature_dim)
        """
        # Forward through backbone
        if hasattr(self.image_model, "backbone"):
            features = self.image_model.backbone(x)

            # Global average pooling if features are spatial
            if features.ndim == 4:
                features = F.adaptive_avg_pool2d(features, 1)
                features = torch.flatten(features, 1)
        else:
            # Fallback: use forward_features if available (timm models)
            if hasattr(self.image_model, "forward_features"):
                features = self.image_model.forward_features(x)
                features = torch.flatten(features, 1)
            else:
                raise AttributeError(
                    "Image model must have 'backbone' or 'forward_features' method"
                )

        return features

    def forward(
        self,
        image: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the multi-input network.

        Args:
            image: Input image tensor (batch_size, 3, H, W)
            metadata: Input metadata tensor (batch_size, metadata_dim), optional

        Returns:
            Logits tensor (batch_size, num_classes)
        """
        # Extract image features
        image_features = self._extract_image_features(image)

        # If no metadata provided, use image-only prediction
        if metadata is None:
            # Use only image features with a learned projection
            # Create a dummy metadata tensor of zeros
            batch_size = image.shape[0]
            metadata = torch.zeros(
                batch_size, self.metadata_dim, device=image.device, dtype=image.dtype
            )

        # Process metadata
        metadata_features = self.metadata_mlp(metadata)

        # Fuse features
        fused_features = torch.cat([image_features, metadata_features], dim=1)

        # Final classification
        logits = self.fusion_classifier(fused_features)

        return logits

    def predict_proba(
        self,
        image: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get probability predictions.

        Args:
            image: Input image tensor
            metadata: Input metadata tensor (optional)

        Returns:
            Probability tensor (batch_size, num_classes)
        """
        logits = self.forward(image, metadata)
        return F.softmax(logits, dim=1)

    def predict(
        self,
        image: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get class predictions.

        Args:
            image: Input image tensor
            metadata: Input metadata tensor (optional)

        Returns:
            Predicted class indices (batch_size,)
        """
        logits = self.forward(image, metadata)
        return torch.argmax(logits, dim=1)

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_multi_input_model(
    image_model_factory: Callable[..., nn.Module],
    image_model_kwargs: dict[str, Any],
    metadata_dim: int,
    num_classes: int = 7,
    metadata_hidden_dim: int = 64,
    fusion_hidden_dim: int = 256,
    dropout_rate: float = 0.3,
) -> MultiInputClassifier:
    """
    Factory function to create a multi-input classifier.

    Args:
        image_model_factory: Factory function that creates the image model
        image_model_kwargs: Keyword arguments for the image model factory
        metadata_dim: Dimension of encoded metadata features
        num_classes: Number of output classes
        metadata_hidden_dim: Hidden dimension for metadata MLP
        fusion_hidden_dim: Hidden dimension for fusion layer
        dropout_rate: Dropout rate

    Returns:
        MultiInputClassifier instance

    Example:
        >>> from skin_lesion_classifier.src.models import create_model_b0
        >>> model = create_multi_input_model(
        ...     image_model_factory=create_model_b0,
        ...     image_model_kwargs={'pretrained': True, 'num_classes': 7},
        ...     metadata_dim=15,  # e.g., 1 (age) + 3 (sex) + 11 (localization)
        ... )
    """
    # Create base image model
    image_model = image_model_factory(**image_model_kwargs)

    # Wrap in multi-input classifier
    return MultiInputClassifier(
        image_model=image_model,
        metadata_dim=metadata_dim,
        num_classes=num_classes,
        metadata_hidden_dim=metadata_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout_rate=dropout_rate,
    )


__all__ = [
    "MultiInputClassifier",
    "create_multi_input_model",
]
