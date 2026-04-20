"""Class metadata and preprocessing helpers for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CLASS_LABELS: Dict[str, str] = {
    "akiec": "Actinic keratoses / Intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

LABEL_TO_IDX = {label: idx for idx, label in enumerate(sorted(CLASS_LABELS.keys()))}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform(image_size: int = 300) -> transforms.Compose:
    """Return deterministic inference transform matching validation preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_image(
    image: Image.Image | np.ndarray | str | Path,
    image_size: int = 300,
) -> torch.Tensor:
    """Preprocess a single image and return a batch tensor shaped (1, C, H, W)."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")

    tensor = get_inference_transform(image_size)(image)
    return tensor.unsqueeze(0)
