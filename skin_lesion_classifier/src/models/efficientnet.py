"""
EfficientNet-V2 Small Model for Skin Lesion Classification.

This module provides the model architecture based on EfficientNet-V2 Small with
a custom classification head for the HAM10000 seven-class classification task.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights


class SkinLesionClassifier(nn.Module):
    """
    EfficientNet-V2 Small based classifier for skin lesion classification.
    
    This model uses a pretrained EfficientNet-V2 backbone with a custom
    classification head optimized for the HAM10000 dataset.
    
    Attributes:
        backbone: EfficientNet-V2 feature extractor
        classifier: Custom classification head
        num_classes: Number of output classes (default: 7 for HAM10000)
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        freeze_layers: Optional[int] = None,
    ):
        """
        Initialize the skin lesion classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze all backbone layers
            freeze_layers: Number of backbone layers to freeze (from the start)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained backbone (EfficientNet-V2 Small)
        self.backbone, self.feature_dim = self._create_backbone(pretrained)
        
        # Lightweight classification head — the pretrained backbone already
        # produces strong features; a single hidden layer is sufficient for
        # 7-class classification and reduces overfitting risk.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )
        # Initialize classifier weights
        self._initialize_classifier()
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        elif freeze_layers is not None:
            self._freeze_layers(freeze_layers)
    
    def _create_backbone(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Create the EfficientNet-V2 Small backbone."""
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_v2_s(weights=weights)
        feature_dim = 1280
        
        # Remove the original classifier
        backbone.classifier = nn.Identity()
        
        return backbone, feature_dim
    
    def _initialize_classifier(self) -> None:
        """Initialize classifier weights using Kaiming initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
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
        """Freeze the first N layers of the backbone."""
        layers = list(self.backbone.features.children())
        for layer in layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (for fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
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


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard misclassified examples.
    This is particularly useful for imbalanced datasets like HAM10000.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights tensor of shape (num_classes,)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: How to reduce the loss ("mean", "sum", or "none")
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # pt is the probability of being correct
        
        # Apply focal weighting
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def create_model(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
) -> SkinLesionClassifier:
    """
    Factory function to create a skin lesion classifier.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for regularization
        freeze_backbone: Whether to freeze backbone layers
        
    Returns:
        Configured SkinLesionClassifier model (EfficientNet-V2 Small backbone)
    """
    return SkinLesionClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )


def get_loss_function(
    loss_type: Literal["cross_entropy", "focal"] = "focal",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Get the loss function for training.
    
    Args:
        loss_type: Type of loss function ("cross_entropy" or "focal")
        class_weights: Optional class weights for imbalanced data (for cross_entropy)
        focal_gamma: Focusing parameter for focal loss
        focal_alpha: Alpha weights for focal loss (if None, uses class_weights)
        
    Returns:
        Loss function module
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal":
        alpha = focal_alpha if focal_alpha is not None else class_weights
        return FocalLoss(alpha=alpha, gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose 'cross_entropy' or 'focal'.")
