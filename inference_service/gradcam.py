"""Grad-CAM implementation for model explainability.

Generates heatmaps showing which regions of an image most influenced
the model's prediction.
"""

from __future__ import annotations

import io
import base64
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM implementation using PyTorch hooks.

    Captures activations and gradients from a target layer to generate
    class activation maps highlighting important image regions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """Initialize GradCAM with model and target layer.

        Args:
            model: The neural network model
            target_layer: The convolutional layer to compute Grad-CAM from
                         (typically the last conv layer before pooling)
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Forward hook to capture activations."""
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        """Backward hook to capture gradients."""
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove registered hooks to free resources."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self) -> None:
        """Cleanup hooks on deletion."""
        try:
            self.remove_hooks()
        except Exception:
            pass

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for the input.

        Args:
            input_tensor: Preprocessed input tensor (1, C, H, W)
            target_class: Class index to generate CAM for. If None, uses
                         the predicted class.

        Returns:
            Normalized heatmap as numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()

        # Enable gradients for this computation
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        return self.generate_from_output(output, target_class)

    def generate_from_output(
        self,
        output: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap from an already-computed model output.

        Args:
            output: Model logits from a forward pass where hooks captured activations
            target_class: Class index to generate CAM for. If None, uses
                         the predicted class.

        Returns:
            Normalized heatmap as numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        return self._compute_cam_from_captured_tensors()

    def _compute_cam_from_captured_tensors(self) -> np.ndarray:
        """Compute normalized CAM from hook-captured activations and gradients."""
        activations = self.activations  # (1, C, H, W)
        gradients = self.gradients  # (1, C, H, W)

        if activations is None or gradients is None:
            raise RuntimeError("Failed to capture activations or gradients")

        # Global average pooling of gradients to get per-channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # Keep only positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def get_target_layer(model: nn.Module) -> nn.Module:
    """Get the appropriate target layer for Grad-CAM based on model architecture.

    Args:
        model: The neural network model (EfficientNet or ConvNeXt based)

    Returns:
        The target convolutional layer for Grad-CAM
    """
    # Check model type by looking at backbone structure
    backbone = getattr(model, 'backbone', None)
    if backbone is None:
        raise ValueError("Model must have a 'backbone' attribute")

    # Both EfficientNet and ConvNeXt use features[-1] as last conv block
    features = getattr(backbone, 'features', None)
    if features is None:
        raise ValueError("Model backbone must have a 'features' attribute")

    # Return the last feature block
    return features[-1]


def apply_colormap(heatmap: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Apply a colormap to a grayscale heatmap (vectorized for speed).

    Args:
        heatmap: Normalized heatmap (H, W) with values in [0, 1]
        colormap: Colormap name ('jet', 'turbo', 'grayscale')

    Returns:
        Colored heatmap as RGB array (H, W, 3) with values in [0, 255]
    """
    if colormap == "jet":
        # Vectorized jet colormap: blue -> cyan -> green -> yellow -> red
        v = heatmap
        r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
        colored = np.stack([r, g, b], axis=-1)
        return (colored * 255).astype(np.uint8)

    elif colormap == "turbo":
        # Vectorized turbo-like colormap
        v = heatmap
        r = np.where(v > 0.5, 0.5 + 2.0 * (v - 0.5), 2.0 * v)
        r = np.clip(r, 0.0, 1.0)
        g = np.clip(1.0 - 2.0 * np.abs(v - 0.5), 0.0, 1.0)
        b = np.where(v < 0.5, 1.0 - 2.0 * v, 0.0)
        b = np.clip(b, 0.0, 1.0)
        colored = np.stack([r, g, b], axis=-1)
        return (colored * 255).astype(np.uint8)

    else:
        # Default grayscale
        gray = (heatmap * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def create_heatmap_overlay(
    original_image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """Create a heatmap overlay on the original image.

    Args:
        original_image: Original PIL image or numpy array
        heatmap: Grad-CAM heatmap (H, W) with values in [0, 1]
        alpha: Opacity of heatmap overlay (0 = invisible, 1 = opaque)
        colormap: Colormap to use ('jet', 'turbo', 'grayscale')

    Returns:
        PIL Image with heatmap overlay
    """
    # Convert to PIL if needed
    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(original_image)

    # Ensure RGB
    original_image = original_image.convert("RGB")
    original_size = original_image.size  # (W, H)

    # Resize heatmap to match original image
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            original_size, resample=Image.BILINEAR
        )
    ) / 255.0

    # Apply colormap
    colored_heatmap = apply_colormap(heatmap_resized, colormap)

    # Blend with original
    original_array = np.array(original_image)
    blended = (1 - alpha) * original_array + alpha * colored_heatmap
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def heatmap_to_base64(
    original_image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
    format: str = "PNG",
) -> str:
    """Generate heatmap overlay and encode as base64 string.

    Args:
        original_image: Original PIL image or numpy array
        heatmap: Grad-CAM heatmap (H, W) with values in [0, 1]
        alpha: Opacity of heatmap overlay
        colormap: Colormap to use
        format: Image format ('PNG' or 'JPEG')

    Returns:
        Base64-encoded image string (without data URI prefix)
    """
    overlay = create_heatmap_overlay(original_image, heatmap, alpha, colormap)

    buffer = io.BytesIO()
    overlay.save(buffer, format=format)
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")
