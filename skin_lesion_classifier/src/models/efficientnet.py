"""
EfficientNet-V2 Model for Skin Lesion Classification.

This module provides the model architecture based on EfficientNet-V2 with
a custom classification head for the HAM10000 seven-class classification task.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_L_Weights,
)


class SkinLesionClassifier(nn.Module):
    """
    EfficientNet-V2 based classifier for skin lesion classification.
    
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
        model_size: Literal["small", "medium", "large"] = "small",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        freeze_layers: Optional[int] = None,
        head_type: Literal["simple", "acrnn"] = "simple",
    ):
        """
        Initialize the skin lesion classifier.
        
        Args:
            num_classes: Number of output classes
            model_size: Size of EfficientNet-V2 model (small, medium, large)
            pretrained: Whether to use pretrained ImageNet weights
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze all backbone layers
            freeze_layers: Number of backbone layers to freeze (from the start)
            head_type: Type of classification head ("simple" or "acrnn")
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_size = model_size
        self.head_type = head_type
        
        # Load pretrained backbone
        self.backbone, self.feature_dim = self._create_backbone(
            model_size, pretrained
        )
        
        if head_type == "acrnn":
            from .acrnn import ACRNN
            self.classifier = ACRNN(input_dim=self.feature_dim, num_classes=num_classes)
        else:
            # Custom classification head with dropout for regularization
            # Using consistent dropout throughout for stronger regularization
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),  # Increased from dropout_rate/2
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),  # Increased from dropout_rate/2
                nn.Linear(256, num_classes),
            )
            # Initialize classifier weights for simple head
            self._initialize_classifier()
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        elif freeze_layers is not None:
            self._freeze_layers(freeze_layers)
    
    def _create_backbone(
        self, model_size: str, pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """Create the EfficientNet-V2 backbone."""
        if model_size == "small":
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_v2_s(weights=weights)
            feature_dim = 1280
        elif model_size == "medium":
            weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_v2_m(weights=weights)
            feature_dim = 1280
        elif model_size == "large":
            weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_v2_l(weights=weights)
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
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


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.
    
    Label smoothing prevents the model from becoming overconfident
    and improves generalization.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """
        Initialize label smoothing cross entropy.
        
        Args:
            smoothing: Smoothing factor (0 = no smoothing, 1 = uniform)
            weight: Class weights tensor
            reduction: How to reduce the loss
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross entropy loss.
        
        Args:
            inputs: Logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Loss value
        """
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        with torch.no_grad():
            smoothed_targets = torch.zeros_like(log_probs)
            smoothed_targets.fill_(self.smoothing / (num_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Compute loss
        loss = -smoothed_targets * log_probs
        
        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
            weight = weight.unsqueeze(0).expand_as(loss)
            loss = loss * weight
        
        loss = loss.sum(dim=1)
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def create_model(
    num_classes: int = 7,
    model_size: Literal["small", "medium", "large"] = "small",
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    head_type: Literal["simple", "acrnn"] = "simple",
) -> SkinLesionClassifier:
    """
    Factory function to create a skin lesion classifier.
    
    Args:
        num_classes: Number of output classes
        model_size: Size of EfficientNet-V2 backbone
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for regularization
        freeze_backbone: Whether to freeze backbone layers
        head_type: Type of classification head ("simple" or "acrnn")
        
    Returns:
        Configured SkinLesionClassifier model
    """
    return SkinLesionClassifier(
        num_classes=num_classes,
        model_size=model_size,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        head_type=head_type,
    )


def get_loss_function(
    loss_type: Literal["cross_entropy", "focal", "label_smoothing"] = "focal",
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Get the loss function for training.
    
    Args:
        loss_type: Type of loss function
        class_weights: Optional class weights for imbalanced data (for cross_entropy and label_smoothing)
        label_smoothing: Smoothing factor for label smoothing loss
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
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(
            smoothing=label_smoothing, weight=class_weights
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
