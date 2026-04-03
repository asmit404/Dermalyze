"""
Inference Module for Skin Lesion Classification.

This module provides a clean interface for model inference, suitable for
integration with REST APIs and server-side deployment.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from src.data.dataset import (
    CLASS_LABELS,
    IDX_TO_LABEL,
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_image,
)
from src.data.metadata_encoder import MetadataEncoder
from src.models.convnext import SkinLesionConvNeXtClassifier
from src.models.convnext import create_model as create_convnext_tiny_model
from src.models.efficientnet import SkinLesionClassifier
from src.models.efficientnet import create_model as create_efficientnet_b0_model
from src.models.efficientnet_b1 import SkinLesionClassifierB1, create_model_b1
from src.models.efficientnet_b2 import SkinLesionClassifierB2, create_model_b2
from src.models.efficientnet_b3 import SkinLesionClassifierB3, create_model_b3
from src.models.efficientnet_b4 import SkinLesionClassifierB4, create_model_b4
from src.models.efficientnet_b5 import SkinLesionClassifierB5, create_model_b5
from src.models.efficientnet_b6 import SkinLesionClassifierB6, create_model_b6
from src.models.efficientnet_b7 import SkinLesionClassifierB7, create_model_b7
from src.models.efficientnetv2_l import SkinLesionClassifierV2L, create_model_v2l
from src.models.efficientnetv2_m import SkinLesionClassifierV2M, create_model_v2m
from src.models.efficientnetv2_s import SkinLesionClassifierV2S, create_model_v2s
from src.models.multi_input import create_multi_input_model
from src.models.resnest_101 import (
    SkinLesionResNeSt101Classifier,
    create_model_resnest101,
)
from src.models.seresnext_101 import (
    SkinLesionSEResNeXt101Classifier,
    create_model_seresnext101,
)
from src.tta_constants import TTA_AUG_COUNTS
from torchvision import transforms

try:
    import cv2 as _cv2

    cv2: Any = _cv2
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


def get_base_tta_transform(image_size: int = 224) -> transforms.Compose:
    """Get the base (non-augmented) TTA transform used for canonical inference."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_tta_transforms(image_size: int = 224) -> List[transforms.Compose]:
    """
    Get Test-Time Augmentation transforms.

    Returns a list of transforms that preserve semantic content:
    - Original
    - Horizontal flip
    - Vertical flip
    - Horizontal + Vertical flip
    - 90° rotation
    - 180° rotation
    - 270° rotation
    - Center crop with different scales

    Args:
        image_size: Target image size

    Returns:
        List of transform compositions
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    base_transform = get_base_tta_transform(image_size)

    tta_transforms = [
        # Original
        base_transform,
        # Horizontal flip
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        # Vertical flip
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        # Both flips
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        # 90° rotation
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        # 180° rotation
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=(180, 180)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        # 270° rotation
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=(270, 270)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        # Multi-scale crops
        transforms.Compose(
            [
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    ]

    return tta_transforms


def get_full_extra_tta_transforms(image_size: int = 224) -> List[transforms.Compose]:
    """Get additional crop-based TTA transforms used only in full mode."""
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

    # FiveCrop order: top-left, top-right, bottom-left, bottom-right, center
    return [
        _corner_crop(0),
        _corner_crop(1),
        _corner_crop(2),
        _corner_crop(3),
    ]


class SkinLesionPredictor:
    """
    High-level inference class for skin lesion classification.

    This class handles model loading, image preprocessing, and prediction
    with a simple interface suitable for API integration.

    Example:
        >>> predictor = SkinLesionPredictor("checkpoint_best.pt")
        >>> result = predictor.predict("image.jpg")
        >>> print(result["predicted_class"], result["confidence"])
    """

    # Educational disclaimer
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
        image_size: int = 224,
    ):
        """
        Initialize the predictor.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or 'mps')
            image_size: Input image size (should match training)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.image_size = image_size

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model, self.config, self.metadata_encoder = self._load_model()

        # Class information
        self.class_names = list(sorted(CLASS_LABELS.keys()))
        self.class_descriptions = CLASS_LABELS

    def _load_model(
        self,
    ) -> Tuple[nn.Module, Dict[str, Any], Optional[MetadataEncoder]]:
        """Load the model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        config = checkpoint.get("config", {})
        metadata_encoder_state = checkpoint.get("metadata_encoder_state")
        metadata_encoder = (
            MetadataEncoder.from_state(metadata_encoder_state)
            if metadata_encoder_state is not None
            else None
        )

        # Create model
        model_config = config.get("model", {})
        backbone_constructors: List[Tuple[str, Any, Any]] = [
            ("efficientnet_b0", SkinLesionClassifier, create_efficientnet_b0_model),
            ("efficientnet_b1", SkinLesionClassifierB1, create_model_b1),
            ("efficientnet_b2", SkinLesionClassifierB2, create_model_b2),
            ("efficientnet_b3", SkinLesionClassifierB3, create_model_b3),
            ("efficientnet_b4", SkinLesionClassifierB4, create_model_b4),
            ("efficientnet_b5", SkinLesionClassifierB5, create_model_b5),
            ("efficientnet_b6", SkinLesionClassifierB6, create_model_b6),
            ("efficientnet_b7", SkinLesionClassifierB7, create_model_b7),
            ("efficientnetv2_s", SkinLesionClassifierV2S, create_model_v2s),
            ("efficientnetv2_m", SkinLesionClassifierV2M, create_model_v2m),
            ("efficientnetv2_l", SkinLesionClassifierV2L, create_model_v2l),
            ("convnext_tiny", SkinLesionConvNeXtClassifier, create_convnext_tiny_model),
            ("resnest_101", SkinLesionResNeSt101Classifier, create_model_resnest101),
            (
                "seresnext_101",
                SkinLesionSEResNeXt101Classifier,
                create_model_seresnext101,
            ),
        ]

        preferred_backbone_raw = str(model_config.get("backbone", "")).strip().lower()
        backbone_aliases = {
            "efficientnet": "efficientnet_b0",
            "efficientnet-b0": "efficientnet_b0",
            "efficientnet_b0": "efficientnet_b0",
            "efficientnet-b1": "efficientnet_b1",
            "efficientnet_b1": "efficientnet_b1",
            "efficientnet-b2": "efficientnet_b2",
            "efficientnet_b2": "efficientnet_b2",
            "efficientnet-b3": "efficientnet_b3",
            "efficientnet_b3": "efficientnet_b3",
            "efficientnet-b4": "efficientnet_b4",
            "efficientnet_b4": "efficientnet_b4",
            "efficientnet-b5": "efficientnet_b5",
            "efficientnet_b5": "efficientnet_b5",
            "efficientnet-b6": "efficientnet_b6",
            "efficientnet_b6": "efficientnet_b6",
            "efficientnet-b7": "efficientnet_b7",
            "efficientnet_b7": "efficientnet_b7",
            "efficientnetv2_s": "efficientnetv2_s",
            "efficientnet_v2_s": "efficientnetv2_s",
            "efficientnet-v2-s": "efficientnetv2_s",
            "efficientnetv2_m": "efficientnetv2_m",
            "efficientnet_v2_m": "efficientnetv2_m",
            "efficientnet-v2-m": "efficientnetv2_m",
            "efficientnetv2_l": "efficientnetv2_l",
            "efficientnet_v2_l": "efficientnetv2_l",
            "efficientnet-v2-l": "efficientnetv2_l",
            "convnext": "convnext_tiny",
            "convnext-tiny": "convnext_tiny",
            "convnext_tiny": "convnext_tiny",
            "resnest101": "resnest_101",
            "resnest-101": "resnest_101",
            "resnest_101": "resnest_101",
            "seresnext101": "seresnext_101",
            "seresnext-101": "seresnext_101",
            "seresnext_101": "seresnext_101",
            "se-resnext-101": "seresnext_101",
            "se_resnext_101": "seresnext_101",
        }
        preferred_backbone = backbone_aliases.get(preferred_backbone_raw, "")

        model_constructors = backbone_constructors
        if preferred_backbone:
            model_constructors = [
                item for item in backbone_constructors if item[0] == preferred_backbone
            ] + [
                item for item in backbone_constructors if item[0] != preferred_backbone
            ]

        model: Optional[nn.Module] = None
        load_errors: List[str] = []

        for architecture_name, model_class, model_factory in model_constructors:
            if metadata_encoder is not None:
                candidate_model = create_multi_input_model(
                    image_model_factory=model_factory,
                    image_model_kwargs={
                        "num_classes": model_config.get("num_classes", 7),
                        "pretrained": False,
                        "dropout_rate": model_config.get("dropout_rate", 0.3),
                        "freeze_backbone": False,
                        "use_gradient_checkpointing": False,
                    },
                    metadata_dim=metadata_encoder.get_metadata_dim(),
                    num_classes=model_config.get("num_classes", 7),
                    metadata_hidden_dim=int(
                        model_config.get("metadata_hidden_dim", 64)
                    ),
                    fusion_hidden_dim=int(model_config.get("fusion_hidden_dim", 256)),
                    dropout_rate=float(model_config.get("dropout_rate", 0.3)),
                )
            else:
                candidate_model = model_class(
                    num_classes=model_config.get("num_classes", 7),
                    pretrained=False,
                    dropout_rate=model_config.get("dropout_rate", 0.3),
                )

            try:
                candidate_model.load_state_dict(checkpoint["model_state_dict"])
                model = candidate_model
                break
            except RuntimeError as exc:
                load_errors.append(f"{architecture_name}: {exc}")

        if model is None:
            attempted = ", ".join(name for name, _, _ in model_constructors)
            raise RuntimeError(
                "Could not load checkpoint with supported architectures "
                f"({attempted}). Config backbone='{preferred_backbone_raw or 'unknown'}'.\n"
                + "\n".join(load_errors)
            )

        model = model.to(self.device)
        model.eval()

        return model, config, metadata_encoder

    def preprocess(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
    ) -> torch.Tensor:
        """
        Preprocess an image for inference.

        Args:
            image: Image as path, PIL Image, numpy array, or bytes

        Returns:
            Preprocessed image tensor
        """
        # Handle bytes input (e.g., from API upload)
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        tensor = preprocess_image(image, self.image_size).to(self.device)

        return tensor

    def _encode_metadata(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
    ) -> Optional[torch.Tensor]:
        """Encode optional metadata dict into model input tensor."""
        if self.metadata_encoder is None:
            return None

        if metadata is None:
            return None

        encoded = self.metadata_encoder.encode_metadata_dict(metadata).to(self.device)
        return encoded.unsqueeze(0).repeat(batch_size, 1)

    def _forward(
        self,
        images: torch.Tensor,
        metadata_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through either image-only or metadata-enabled model."""
        if metadata_tensor is not None:
            return self.model(images, metadata_tensor)
        return self.model(images)

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
        metadata: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        include_disclaimer: bool = True,
    ) -> Dict[str, Any]:
        """
        Make a prediction for a single image.

        Args:
            image: Input image (path, PIL Image, numpy array, or bytes)
            top_k: Number of top predictions to return
            include_disclaimer: Whether to include educational disclaimer

        Returns:
            Dictionary containing:
                - predicted_class: Most likely class label
                - predicted_class_description: Full description of predicted class
                - confidence: Confidence score for predicted class
                - probabilities: Dictionary of all class probabilities
                - top_k_predictions: List of top K predictions
                - disclaimer: Educational disclaimer (if requested)
        """
        # Preprocess
        tensor = self.preprocess(image)
        metadata_tensor = self._encode_metadata(metadata=metadata, batch_size=1)

        # Forward pass
        logits = self._forward(tensor, metadata_tensor)
        probs = F.softmax(logits, dim=1)[0]

        # Synchronize MPS for accurate results
        if self.device.type == "mps":
            torch.mps.synchronize()

        # Get predictions
        probs_np = probs.cpu().numpy()
        predicted_idx = int(np.argmax(probs_np))
        predicted_class = IDX_TO_LABEL[predicted_idx]
        confidence = float(probs_np[predicted_idx])

        # All probabilities
        all_probs = {
            self.class_names[i]: float(probs_np[i])
            for i in range(len(self.class_names))
        }

        # Top K predictions
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
            "confidence": confidence,
            "probabilities": all_probs,
            "top_k_predictions": top_k_predictions,
        }

        if include_disclaimer:
            result["disclaimer"] = self.DISCLAIMER

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray, bytes]],
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple images.

        Args:
            images: List of input images
            top_k: Number of top predictions per image

        Returns:
            List of prediction dictionaries
        """
        # Preprocess all images
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)

        metadata_tensor = None
        if self.metadata_encoder is not None and metadata is not None:
            if len(metadata) != len(images):
                raise ValueError(
                    "If metadata is provided for batch prediction, its length must match images."
                )
            encoded_rows = [
                self.metadata_encoder.encode_metadata_dict(item or {})
                for item in metadata
            ]
            metadata_tensor = torch.stack(encoded_rows, dim=0).to(self.device)

        # Forward pass
        logits = self._forward(batch, metadata_tensor)
        probs = F.softmax(logits, dim=1)

        # Process each prediction
        results = []
        for i in range(len(images)):
            probs_np = probs[i].cpu().numpy()
            predicted_idx = int(np.argmax(probs_np))
            predicted_class = IDX_TO_LABEL[predicted_idx]

            # Top K predictions
            top_k_indices = np.argsort(probs_np)[::-1][:top_k]
            top_k_predictions = [
                {
                    "class": IDX_TO_LABEL[int(idx)],
                    "description": CLASS_LABELS[IDX_TO_LABEL[int(idx)]],
                    "probability": float(probs_np[idx]),
                }
                for idx in top_k_indices
            ]

            results.append(
                {
                    "predicted_class": predicted_class,
                    "predicted_class_description": CLASS_LABELS[predicted_class],
                    "confidence": float(probs_np[predicted_idx]),
                    "probabilities": {
                        self.class_names[j]: float(probs_np[j])
                        for j in range(len(self.class_names))
                    },
                    "top_k_predictions": top_k_predictions,
                }
            )

        return results

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
        """
        Make prediction with Test-Time Augmentation.

        Args:
            image: Input image
            tta_mode: TTA complexity (light: 4 augs, medium: 8 augs, full: all)
            aggregation: How to aggregate predictions (mean, geometric_mean, max)
            use_clahe_tta: Whether to add CLAHE-processed branch during TTA
            clahe_clip_limit: CLAHE clip limit
            clahe_grid_size: CLAHE tile grid size
            top_k: Number of top predictions to return
            include_disclaimer: Whether to include educational disclaimer

        Returns:
            Prediction dictionary with TTA-enhanced probabilities
        """
        # Load image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        # Get TTA transforms based on mode
        all_transforms = get_tta_transforms(self.image_size)

        if tta_mode == "light":
            # 4 branches: original + flips
            tta_transforms = all_transforms[:4]
        elif tta_mode == "medium":
            # 8 branches: base set (flips + rotations + center zoom crop)
            tta_transforms = all_transforms
        elif tta_mode == "full":
            # 12 branches: medium + extra corner zoom-crop branches
            tta_transforms = all_transforms + get_full_extra_tta_transforms(
                self.image_size
            )
        else:
            valid_modes = ", ".join(TTA_AUG_COUNTS.keys())
            raise ValueError(
                f"Invalid tta_mode: {tta_mode!r}. Expected one of: {valid_modes}."
            )

        if use_clahe_tta and cv2 is None:
            raise RuntimeError(
                "CLAHE-TTA requested but OpenCV is not installed. "
                "Install opencv-python or opencv-python-headless."
            )

        # Collect predictions from all augmentations
        all_probs = []
        metadata_tensor = self._encode_metadata(metadata=metadata, batch_size=1)
        for transform in tta_transforms:
            tensor = transform(image).unsqueeze(0).to(self.device)
            logits = self._forward(tensor, metadata_tensor)
            probs = F.softmax(logits, dim=1)[0]
            all_probs.append(probs.cpu().numpy())

        if use_clahe_tta:
            clahe_image = apply_clahe_to_pil(
                image,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_grid_size,
            )
            base_transform = get_base_tta_transform(self.image_size)
            tensor = base_transform(clahe_image).unsqueeze(0).to(self.device)
            logits = self._forward(tensor, metadata_tensor)
            probs = F.softmax(logits, dim=1)[0]
            all_probs.append(probs.cpu().numpy())

        # Aggregate predictions
        all_probs = np.array(all_probs)  # Shape: (n_augmentations, n_classes)

        if aggregation == "mean":
            final_probs = np.mean(all_probs, axis=0)
        elif aggregation == "geometric_mean":
            # Geometric mean is better for probabilities
            final_probs = np.exp(np.mean(np.log(all_probs + 1e-10), axis=0))
            final_probs = final_probs / final_probs.sum()  # Normalize
        else:  # max
            final_probs = np.max(all_probs, axis=0)

        # Get predictions
        predicted_idx = int(np.argmax(final_probs))
        predicted_class = IDX_TO_LABEL[predicted_idx]
        confidence = float(final_probs[predicted_idx])

        # All probabilities
        all_probs_dict = {
            self.class_names[i]: float(final_probs[i])
            for i in range(len(self.class_names))
        }

        # Top K predictions
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
            "confidence": confidence,
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

    def get_class_info(self) -> Dict[str, Any]:
        """Get information about all classes."""
        return {
            "classes": self.class_descriptions.copy(),
            "num_classes": len(self.class_descriptions),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        model_config = self.config.get("model", {})
        metadata_columns = None
        if self.metadata_encoder is not None:
            metadata_columns = [
                self.metadata_encoder.age_column,
                self.metadata_encoder.sex_column,
                self.metadata_encoder.localization_column,
            ]
        get_total_params_fn = getattr(self.model, "get_total_params", None)
        total_parameters = (
            int(get_total_params_fn())
            if callable(get_total_params_fn)
            else int(sum(param.numel() for param in self.model.parameters()))
        )
        return {
            "model_name": model_config.get("backbone", "auto"),
            "num_classes": model_config.get("num_classes", 7),
            "image_size": self.image_size,
            "device": str(self.device),
            "checkpoint": str(self.checkpoint_path),
            "total_parameters": total_parameters,
            "metadata_enabled": self.metadata_encoder is not None,
            "metadata_columns": metadata_columns,
        }


class EnsemblePredictor:
    """
    Ensemble predictor that combines predictions from multiple models.

    Supports multiple ensemble strategies:
    - Mean averaging of probabilities
    - Weighted averaging with custom weights
    - Voting (hard or soft)
    - Can be combined with TTA for each model

    Example:
        >>> ensemble = EnsemblePredictor([
        ...     "model1_best.pt",
        ...     "model2_best.pt",
        ...     "model3_best.pt"
        ... ])
        >>> result = ensemble.predict("image.jpg", use_tta=True)
    """

    DISCLAIMER = SkinLesionPredictor.DISCLAIMER

    def __init__(
        self,
        checkpoint_paths: List[Union[str, Path]],
        weights: Optional[List[float]] = None,
        device: Optional[str] = None,
        image_size: int = 224,
    ):
        """
        Initialize ensemble predictor.

        Args:
            checkpoint_paths: List of paths to model checkpoints
            weights: Optional weights for each model (default: equal weights)
            device: Device for inference
            image_size: Input image size
        """
        if len(checkpoint_paths) == 0:
            raise ValueError("Must provide at least one checkpoint")

        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.image_size = image_size

        # Set weights
        if weights is None:
            self.weights = np.ones(len(checkpoint_paths)) / len(checkpoint_paths)
        else:
            if len(weights) != len(checkpoint_paths):
                raise ValueError("Number of weights must match number of checkpoints")
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()  # Normalize

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load all models
        self.predictors = []
        for checkpoint_path in self.checkpoint_paths:
            predictor = SkinLesionPredictor(
                checkpoint_path=checkpoint_path,
                device=str(self.device),
                image_size=image_size,
            )
            self.predictors.append(predictor)

        # Class information (same for all models)
        self.class_names = self.predictors[0].class_names
        self.class_descriptions = self.predictors[0].class_descriptions

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
        metadata: Optional[Dict[str, Any]] = None,
        aggregation: Literal[
            "mean", "weighted_mean", "geometric_mean", "max"
        ] = "weighted_mean",
        use_tta: bool = False,
        tta_mode: Literal["light", "medium", "full"] = "medium",
        tta_aggregation: Literal["mean", "geometric_mean", "max"] = "mean",
        use_clahe_tta: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
        top_k: int = 3,
        include_disclaimer: bool = True,
    ) -> Dict[str, Any]:
        """
        Make ensemble prediction.

        Args:
            image: Input image
            aggregation: How to combine model predictions
            use_tta: Whether to use TTA for each model
            tta_mode: TTA complexity if use_tta=True
            tta_aggregation: How to aggregate TTA predictions
            use_clahe_tta: Whether to add CLAHE-processed branch during TTA
            clahe_clip_limit: CLAHE clip limit
            clahe_grid_size: CLAHE tile grid size
            top_k: Number of top predictions to return
            include_disclaimer: Whether to include disclaimer

        Returns:
            Ensemble prediction dictionary
        """
        # Collect predictions from all models
        all_model_probs = []

        for predictor in self.predictors:
            if use_tta:
                result = predictor.predict_with_tta(
                    image=image,
                    metadata=metadata,
                    tta_mode=tta_mode,
                    aggregation=tta_aggregation,
                    use_clahe_tta=use_clahe_tta,
                    clahe_clip_limit=clahe_clip_limit,
                    clahe_grid_size=clahe_grid_size,
                    include_disclaimer=False,
                )
            else:
                result = predictor.predict(
                    image=image,
                    metadata=metadata,
                    include_disclaimer=False,
                )

            # Extract probabilities as array
            probs = np.array([result["probabilities"][cls] for cls in self.class_names])
            all_model_probs.append(probs)

        all_model_probs = np.array(all_model_probs)  # Shape: (n_models, n_classes)

        # Aggregate model predictions
        if aggregation == "mean":
            final_probs = np.mean(all_model_probs, axis=0)
        elif aggregation == "weighted_mean":
            final_probs = np.average(all_model_probs, axis=0, weights=self.weights)
        elif aggregation == "geometric_mean":
            final_probs = np.exp(np.mean(np.log(all_model_probs + 1e-10), axis=0))
            final_probs = final_probs / final_probs.sum()
        else:  # max
            final_probs = np.max(all_model_probs, axis=0)

        # Get predictions
        predicted_idx = int(np.argmax(final_probs))
        predicted_class = IDX_TO_LABEL[predicted_idx]
        confidence = float(final_probs[predicted_idx])

        # All probabilities
        all_probs_dict = {
            self.class_names[i]: float(final_probs[i])
            for i in range(len(self.class_names))
        }

        # Top K predictions
        top_k_indices = np.argsort(final_probs)[::-1][:top_k]
        top_k_predictions = [
            {
                "class": IDX_TO_LABEL[int(idx)],
                "description": CLASS_LABELS[IDX_TO_LABEL[int(idx)]],
                "probability": float(final_probs[idx]),
            }
            for idx in top_k_indices
        ]

        # Calculate prediction variance across models (uncertainty measure)
        prediction_std = np.std(all_model_probs, axis=0)

        result = {
            "predicted_class": predicted_class,
            "predicted_class_description": CLASS_LABELS[predicted_class],
            "confidence": confidence,
            "probabilities": all_probs_dict,
            "top_k_predictions": top_k_predictions,
            "ensemble_size": len(self.predictors),
            "aggregation_method": aggregation,
            "use_tta": use_tta,
            "prediction_uncertainty": float(
                prediction_std[predicted_idx]
            ),  # Std of predicted class
            "mean_uncertainty": float(
                np.mean(prediction_std)
            ),  # Mean std across all classes
        }

        if use_tta:
            result["tta_mode"] = tta_mode
            result["tta_aggregation"] = tta_aggregation

        if include_disclaimer:
            result["disclaimer"] = self.DISCLAIMER

        return result

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble."""
        return {
            "num_models": len(self.predictors),
            "model_weights": self.weights.tolist(),
            "checkpoints": [str(p) for p in self.checkpoint_paths],
            "device": str(self.device),
            "image_size": self.image_size,
        }


def load_predictor(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None,
) -> SkinLesionPredictor:
    """
    Factory function to create a predictor instance.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device for inference

    Returns:
        Configured SkinLesionPredictor instance
    """
    return SkinLesionPredictor(checkpoint_path, device)


def load_ensemble_predictor(
    checkpoint_paths: List[Union[str, Path]],
    weights: Optional[List[float]] = None,
    device: Optional[str] = None,
) -> EnsemblePredictor:
    """
    Factory function to create an ensemble predictor instance.

    Args:
        checkpoint_paths: List of paths to model checkpoints
        weights: Optional weights for each model
        device: Device for inference

    Returns:
        Configured EnsemblePredictor instance
    """
    return EnsemblePredictor(checkpoint_paths, weights, device)


# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test inference module")
    parser.add_argument("--checkpoint", type=Path, help="Single model checkpoint")
    parser.add_argument(
        "--ensemble", type=Path, nargs="+", help="Multiple checkpoints for ensemble"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--use-tta", action="store_true", help="Use test-time augmentation"
    )
    parser.add_argument(
        "--tta-mode", choices=["light", "medium", "full"], default="medium"
    )
    args = parser.parse_args()

    if args.ensemble:
        # Ensemble prediction
        print(f"Loading ensemble with {len(args.ensemble)} models...")
        predictor = load_ensemble_predictor(args.ensemble)
        print(f"Ensemble info: {predictor.get_ensemble_info()}")

        result = predictor.predict(
            args.image,
            use_tta=args.use_tta,
            tta_mode=args.tta_mode,
        )
        print(f"\n[ENSEMBLE] Prediction: {result['predicted_class']}")
        print(f"Uncertainty: {result['prediction_uncertainty']:.4f}")
    else:
        # Single model prediction
        predictor = load_predictor(args.checkpoint)
        print(f"Model info: {predictor.get_model_info()}")

        if args.use_tta:
            result = predictor.predict_with_tta(
                args.image,
                tta_mode=args.tta_mode,
            )
            print(
                f"\n[TTA-{args.tta_mode.upper()}] Prediction: {result['predicted_class']}"
            )
            print(f"Augmentations: {result['tta_augmentations']}")
        else:
            result = predictor.predict(args.image)
            print(f"\nPrediction: {result['predicted_class']}")

    print(f"Description: {result['predicted_class_description']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nTop predictions:")
    for pred in result["top_k_predictions"]:
        print(f"  {pred['class']}: {pred['probability']:.4f}")

    if "disclaimer" in result:
        print(f"\n{result['disclaimer']}")
