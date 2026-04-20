"""Inference-only predictor module for skin lesion classification."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from .metadata import (
        CLASS_LABELS,
        IDX_TO_LABEL,
        IMAGENET_MEAN,
        IMAGENET_STD,
        preprocess_image,
    )
    from .metadata_encoder import MetadataEncoder
    from .models.convnext import SkinLesionConvNeXtClassifier
    from .models.efficientnet import SkinLesionClassifier, normalize_efficientnet_variant
    from .models.multi_input import MultiInputClassifier
    from .tta_constants import TTA_AUG_COUNTS
    from .gradcam import GradCAM, get_target_layer, heatmap_to_base64
except ImportError:
    from metadata import (
        CLASS_LABELS,
        IDX_TO_LABEL,
        IMAGENET_MEAN,
        IMAGENET_STD,
        preprocess_image,
    )
    from metadata_encoder import MetadataEncoder
    from models.convnext import SkinLesionConvNeXtClassifier
    from models.efficientnet import SkinLesionClassifier, normalize_efficientnet_variant
    from models.multi_input import MultiInputClassifier
    from tta_constants import TTA_AUG_COUNTS
    from gradcam import GradCAM, get_target_layer, heatmap_to_base64

try:
    import cv2
except ImportError:
    cv2 = None


def apply_clahe_to_pil(
    image: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> Image.Image:
    """Apply CLAHE to a PIL RGB image using LAB luminance channel."""
    if cv2 is None:
        raise RuntimeError(
            "CLAHE-TTA requested but OpenCV is not installed. "
            "Install opencv-python or opencv-python-headless."
        )

    image_rgb = np.array(image.convert("RGB"))
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    l_channel = clahe.apply(l_channel)
    image_lab = cv2.merge((l_channel, a_channel, b_channel))
    image_rgb_clahe = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(image_rgb_clahe)


def get_base_tta_transform(image_size: int = 300) -> transforms.Compose:
    """Get the canonical inference transform used as one TTA branch."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_tta_transforms(image_size: int = 300) -> list[transforms.Compose]:
    """Return the common TTA transform set (8 branches)."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    base_transform = get_base_tta_transform(image_size)

    return [
        base_transform,
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=(180, 180)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=(270, 270)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    ]


def get_full_extra_tta_transforms(image_size: int = 300) -> list[transforms.Compose]:
    """Get corner crop branches used only in full TTA mode."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    scaled_size = int(image_size * 1.1)

    def _corner_crop(index: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((scaled_size, scaled_size)),
                transforms.FiveCrop(image_size),
                transforms.Lambda(lambda crops, idx=index: crops[idx]),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return [_corner_crop(0), _corner_crop(1), _corner_crop(2), _corner_crop(3)]


class SkinLesionPredictor:
    """High-level predictor class for API-friendly model inference."""

    DISCLAIMER = (
        "EDUCATIONAL USE ONLY: This system is for educational and research "
        "purposes only. It does not provide medical diagnosis or clinical "
        "decision-making. Always consult a qualified healthcare professional "
        "for medical advice."
    )

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        image_size: int = 300,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.image_size = image_size

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model, self.config, self.metadata_encoder = self._load_model()
        self.uses_metadata = self.metadata_encoder is not None
        self.class_names = list(sorted(CLASS_LABELS.keys()))
        self.class_descriptions = CLASS_LABELS

    @staticmethod
    def _normalize_backbone(backbone: str) -> str:
        normalized = str(backbone or "efficientnet_b0").strip().lower().replace("-", "_")
        alias_map = {
            "convnext": "convnext_tiny",
            "efficientnet": "efficientnet_b0",
        }
        return alias_map.get(normalized, normalized)

    def _build_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create model architecture that matches checkpoint training config."""
        backbone = self._normalize_backbone(
            str(model_config.get("backbone", "efficientnet_b0"))
        )
        num_classes = int(model_config.get("num_classes", 7))
        dropout_rate = float(model_config.get("dropout_rate", 0.3))

        if backbone == "convnext_tiny":
            return SkinLesionConvNeXtClassifier(
                num_classes=num_classes,
                pretrained=False,
                dropout_rate=dropout_rate,
            )

        if backbone.startswith("efficientnet"):
            return SkinLesionClassifier(
                num_classes=num_classes,
                pretrained=False,
                dropout_rate=dropout_rate,
                backbone_variant=normalize_efficientnet_variant(backbone),
            )

        raise ValueError(
            "Unsupported backbone for inference service: "
            f"{backbone!r}. Supported: convnext_tiny and EfficientNet variants."
        )

    def _load_model(self) -> Tuple[nn.Module, Dict[str, Any], Optional[MetadataEncoder]]:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        model_config = config.get("model", {})
        metadata_encoder_state = checkpoint.get("metadata_encoder_state")

        image_model = self._build_model(model_config)
        metadata_encoder: Optional[MetadataEncoder] = None

        if isinstance(metadata_encoder_state, dict):
            metadata_encoder = MetadataEncoder.from_state(metadata_encoder_state)
            model = MultiInputClassifier(
                image_model=image_model,
                metadata_dim=metadata_encoder.get_metadata_dim(),
                num_classes=int(model_config.get("num_classes", 7)),
                metadata_hidden_dim=int(model_config.get("metadata_hidden_dim", 64)),
                fusion_hidden_dim=int(model_config.get("fusion_hidden_dim", 256)),
                dropout_rate=float(model_config.get("dropout_rate", 0.3)),
            )
        else:
            model = image_model

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        return model, config, metadata_encoder

    def _prepare_metadata_tensor(
        self,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        if self.metadata_encoder is None or metadata is None:
            return None
        metadata_tensor = self.metadata_encoder.encode_metadata_dict(metadata)
        return metadata_tensor.unsqueeze(0).to(self.device)

    def _forward_model(
        self,
        image_tensor: torch.Tensor,
        metadata_tensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.uses_metadata:
            return self.model(image_tensor, metadata_tensor)
        return self.model(image_tensor)

    def preprocess(self, image: Union[str, Path, Image.Image, np.ndarray, bytes]) -> torch.Tensor:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        return preprocess_image(image, self.image_size).to(self.device)

    def _to_pil_image(
        self, image: Union[str, Path, Image.Image, np.ndarray, bytes]
    ) -> Image.Image:
        """Convert various image inputs to PIL Image."""
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def generate_gradcam(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
        target_class: Optional[int] = None,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> str:
        """Generate Grad-CAM heatmap overlay as base64 string.

        Args:
            image: Input image in any supported format
            target_class: Class index to generate CAM for (None = predicted class)
            alpha: Heatmap overlay opacity (0-1)
            colormap: Colormap to use ('jet', 'turbo', 'grayscale')

        Returns:
            Base64-encoded PNG image of heatmap overlay
        """
        # Get PIL image for overlay
        pil_image = self._to_pil_image(image)

        # Preprocess for model
        tensor = self.preprocess(image)

        # Get target layer and create GradCAM
        target_layer = get_target_layer(self.model)
        gradcam = GradCAM(self.model, target_layer)

        try:
            # Generate heatmap
            heatmap = gradcam.generate(tensor, target_class)

            # Create overlay and encode as base64
            gradcam_base64 = heatmap_to_base64(
                pil_image, heatmap, alpha=alpha, colormap=colormap
            )
            return gradcam_base64
        finally:
            # Always clean up hooks
            gradcam.remove_hooks()

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
        metadata: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        include_disclaimer: bool = True,
        include_gradcam: bool = False,
        gradcam_alpha: float = 0.4,
        gradcam_colormap: str = "jet",
    ) -> Dict[str, Any]:
        """Run inference on an image.

        Args:
            image: Input image in any supported format
            metadata: Optional metadata dictionary for metadata-fusion models
            top_k: Number of top predictions to return
            include_disclaimer: Whether to include educational disclaimer
            include_gradcam: Whether to generate Grad-CAM heatmap
            gradcam_alpha: Heatmap overlay opacity (0-1)
            gradcam_colormap: Colormap to use ('jet', 'turbo', 'grayscale')

        Returns:
            Dictionary with predictions and optionally gradcam_image
        """
        # Keep PIL image for gradcam if needed
        pil_image = self._to_pil_image(image) if include_gradcam else None

        tensor = self.preprocess(image)
        metadata_tensor = self._prepare_metadata_tensor(metadata)

        # For gradcam we need gradients, so use a separate path
        if include_gradcam:
            target_layer = get_target_layer(self.model)
            gradcam = GradCAM(self.model, target_layer)
            try:
                # Forward pass with gradients enabled
                tensor.requires_grad_(True)
                logits = self._forward_model(tensor, metadata_tensor)
                probs = F.softmax(logits, dim=1)[0]

                probs_np = probs.detach().cpu().numpy()
                predicted_idx = int(np.argmax(probs_np))

                # Generate heatmap from the same forward pass to keep hooks aligned
                heatmap = gradcam.generate_from_output(logits, predicted_idx)
                gradcam_base64 = heatmap_to_base64(
                    pil_image, heatmap, alpha=gradcam_alpha, colormap=gradcam_colormap
                )
            finally:
                gradcam.remove_hooks()
        else:
            with torch.no_grad():
                logits = self._forward_model(tensor, metadata_tensor)
                probs = F.softmax(logits, dim=1)[0]
            probs_np = probs.cpu().numpy()
            predicted_idx = int(np.argmax(probs_np))
            gradcam_base64 = None

        if self.device.type == "mps":
            torch.mps.synchronize()

        predicted_class = IDX_TO_LABEL[predicted_idx]

        all_probs = {
            self.class_names[i]: float(probs_np[i])
            for i in range(len(self.class_names))
        }

        top_k_indices = np.argsort(probs_np)[::-1][:top_k]
        top_k_predictions = [
            {
                "class": IDX_TO_LABEL[int(idx)],
                "description": CLASS_LABELS[IDX_TO_LABEL[int(idx)]],
                "probability": float(probs_np[idx]),
            }
            for idx in top_k_indices
        ]

        result = {
            "predicted_class": predicted_class,
            "predicted_class_description": CLASS_LABELS[predicted_class],
            "confidence": float(probs_np[predicted_idx]),
            "probabilities": all_probs,
            "top_k_predictions": top_k_predictions,
        }

        if include_gradcam and gradcam_base64:
            result["gradcam_image"] = gradcam_base64

        if include_disclaimer:
            result["disclaimer"] = self.DISCLAIMER
        return result

    @torch.no_grad()
    def predict_with_tta(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
        metadata: Optional[Dict[str, Any]] = None,
        tta_mode: Literal["light", "medium", "full"] = "medium",
        aggregation: Literal["mean", "geometric_mean", "max"] = "mean",
        use_clahe_tta: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
        top_k: int = 3,
        include_disclaimer: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        all_transforms = get_tta_transforms(self.image_size)
        if tta_mode == "light":
            tta_transforms = all_transforms[:4]
        elif tta_mode == "medium":
            tta_transforms = all_transforms
        elif tta_mode == "full":
            tta_transforms = all_transforms + get_full_extra_tta_transforms(self.image_size)
        else:
            valid_modes = ", ".join(TTA_AUG_COUNTS.keys())
            raise ValueError(f"Invalid tta_mode: {tta_mode!r}. Expected one of: {valid_modes}.")

        if use_clahe_tta and cv2 is None:
            raise RuntimeError(
                "CLAHE-TTA requested but OpenCV is not installed. "
                "Install opencv-python or opencv-python-headless."
            )

        metadata_tensor = self._prepare_metadata_tensor(metadata)

        probs_collection = []
        for transform in tta_transforms:
            tensor = transform(image).unsqueeze(0).to(self.device)
            logits = self._forward_model(tensor, metadata_tensor)
            probs_collection.append(F.softmax(logits, dim=1)[0].cpu().numpy())

        if use_clahe_tta:
            clahe_image = apply_clahe_to_pil(
                image,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_grid_size,
            )
            base_transform = get_base_tta_transform(self.image_size)
            tensor = base_transform(clahe_image).unsqueeze(0).to(self.device)
            logits = self._forward_model(tensor, metadata_tensor)
            probs_collection.append(F.softmax(logits, dim=1)[0].cpu().numpy())

        probs_array = np.array(probs_collection)
        if aggregation == "mean":
            final_probs = np.mean(probs_array, axis=0)
        elif aggregation == "geometric_mean":
            final_probs = np.exp(np.mean(np.log(probs_array + 1e-10), axis=0))
            final_probs = final_probs / final_probs.sum()
        else:
            final_probs = np.max(probs_array, axis=0)

        predicted_idx = int(np.argmax(final_probs))
        predicted_class = IDX_TO_LABEL[predicted_idx]

        all_probs_dict = {
            self.class_names[i]: float(final_probs[i])
            for i in range(len(self.class_names))
        }

        top_k_indices = np.argsort(final_probs)[::-1][:top_k]
        top_k_predictions = [
            {
                "class": IDX_TO_LABEL[int(idx)],
                "description": CLASS_LABELS[IDX_TO_LABEL[int(idx)]],
                "probability": float(final_probs[idx]),
            }
            for idx in top_k_indices
        ]

        result = {
            "predicted_class": predicted_class,
            "predicted_class_description": CLASS_LABELS[predicted_class],
            "confidence": float(final_probs[predicted_idx]),
            "probabilities": all_probs_dict,
            "top_k_predictions": top_k_predictions,
            "tta_mode": tta_mode,
            "tta_augmentations": len(tta_transforms) + (1 if use_clahe_tta else 0),
            "aggregation_method": aggregation,
            "use_clahe_tta": bool(use_clahe_tta),
        }

        if include_disclaimer:
            result["disclaimer"] = self.DISCLAIMER
        return result
