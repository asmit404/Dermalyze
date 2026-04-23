"""Service-compatible Swin predictor for Dermalyze inference.

This module wraps a Hugging Face Transformers image-classification model in the
same API shape as ``inference_service.predictor.SkinLesionPredictor``.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoConfig, AutoModelForImageClassification

try:
    from transformers import AutoImageProcessor
except ImportError:  # transformers 4.20-era configs still use feature extractors.
    AutoImageProcessor = None

try:
    from transformers import AutoFeatureExtractor
except ImportError:
    AutoFeatureExtractor = None

try:
    from ..metadata import CLASS_LABELS
except ImportError:
    try:
        from inference_service.metadata import CLASS_LABELS
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from metadata import CLASS_LABELS


HF_MODEL_ID = "gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

CANONICAL_CLASS_IDS = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

_LABEL_ALIASES = {
    "akiec": "akiec",
    "actinic keratoses": "akiec",
    "actinic keratosis": "akiec",
    "intraepithelial carcinoma": "akiec",
    "bcc": "bcc",
    "basal cell carcinoma": "bcc",
    "bkl": "bkl",
    "benign keratosis like lesions": "bkl",
    "benign keratosis": "bkl",
    "df": "df",
    "dermatofibroma": "df",
    "mel": "mel",
    "melanoma": "mel",
    "nv": "nv",
    "melanocytic nevi": "nv",
    "melanocytic nevus": "nv",
    "nevi": "nv",
    "nevus": "nv",
    "vasc": "vasc",
    "vascular lesions": "vasc",
    "vascular lesion": "vasc",
}


def _normalize_label(label: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def _canonical_label(raw_label: object, fallback_index: int) -> str:
    normalized = _normalize_label(raw_label)
    if normalized in _LABEL_ALIASES:
        return _LABEL_ALIASES[normalized]

    if normalized.startswith("label ") and fallback_index < len(CANONICAL_CLASS_IDS):
        return CANONICAL_CLASS_IDS[fallback_index]
    if "actinic" in normalized or "akiec" in normalized:
        return "akiec"
    if "basal" in normalized or "bcc" in normalized:
        return "bcc"
    if "benign" in normalized or "bkl" in normalized:
        return "bkl"
    if "dermatofibroma" in normalized:
        return "df"
    if "melanocytic" in normalized or "nevi" in normalized or "nevus" in normalized:
        return "nv"
    if "melanoma" in normalized:
        return "mel"
    if "vascular" in normalized or "vasc" in normalized:
        return "vasc"

    if fallback_index < len(CANONICAL_CLASS_IDS):
        return CANONICAL_CLASS_IDS[fallback_index]
    raise ValueError(f"Unsupported Swin label {raw_label!r} at index {fallback_index}.")


def _detect_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _local_model_dir_is_complete() -> bool:
    return (
        (LOCAL_MODEL_DIR / "config.json").exists()
        and (LOCAL_MODEL_DIR / "preprocessor_config.json").exists()
        and (LOCAL_MODEL_DIR / "pytorch_model.bin").exists()
    )


def _load_processor(model_source: str):
    if AutoImageProcessor is not None:
        return AutoImageProcessor.from_pretrained(model_source)
    if AutoFeatureExtractor is not None:
        return AutoFeatureExtractor.from_pretrained(model_source)
    raise RuntimeError(
        "Transformers image preprocessing support is unavailable. "
        "Install a version with AutoImageProcessor or AutoFeatureExtractor."
    )


def _extract_state_dict(raw_checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(raw_checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict"):
            nested = raw_checkpoint.get(key)
            if isinstance(nested, Mapping):
                return nested
        if all(isinstance(key, str) for key in raw_checkpoint.keys()):
            return raw_checkpoint
    raise ValueError("Swin weights must be a PyTorch state_dict or contain model_state_dict.")


def _clean_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    has_hf_keys = any(
        key.startswith(("swin.", "classifier."))
        for key in cleaned
    )
    if not has_hf_keys and any(key.startswith("model.") for key in cleaned):
        cleaned = {
            key.removeprefix("model."): value
            for key, value in cleaned.items()
        }

    return cleaned


def _center_crop_scaled(image: Image.Image, scale: float = 1.1) -> Image.Image:
    width, height = image.size
    scaled = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = max(0, (scaled.width - width) // 2)
    top = max(0, (scaled.height - height) // 2)
    return scaled.crop((left, top, left + width, top + height))


def _corner_crops_scaled(image: Image.Image, scale: float = 1.1) -> list[Image.Image]:
    width, height = image.size
    scaled = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BICUBIC,
    )
    boxes = [
        (0, 0, width, height),
        (scaled.width - width, 0, scaled.width, height),
        (0, scaled.height - height, width, scaled.height),
        (scaled.width - width, scaled.height - height, scaled.width, scaled.height),
    ]
    return [scaled.crop(box) for box in boxes]


class SwinSkinLesionPredictor:
    """High-level predictor compatible with the inference service contract."""

    DISCLAIMER = (
        "EDUCATIONAL USE ONLY: This system is for educational and research "
        "purposes only. It does not provide medical diagnosis or clinical "
        "decision-making. Always consult a qualified healthcare professional "
        "for medical advice."
    )

    def __init__(
        self,
        model_source: Union[str, Path, None] = None,
        weights_path: Union[str, Path, None] = None,
        device: Optional[str] = None,
        image_size: Optional[int] = None,
    ):
        self.device = _detect_device(device)
        self.image_size = image_size
        self.model_source, self.weights_path = self._resolve_model_inputs(
            model_source=model_source,
            weights_path=weights_path,
        )

        self.processor = _load_processor(self.model_source)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        self.index_to_class_id = self._build_index_to_class_id()
        self.class_names = list(CANONICAL_CLASS_IDS)
        self.class_descriptions = CLASS_LABELS
        self.uses_metadata = False

    @staticmethod
    def _resolve_model_inputs(
        model_source: Union[str, Path, None],
        weights_path: Union[str, Path, None],
    ) -> tuple[str, Optional[Path]]:
        source = str(
            model_source
            or os.environ.get("SWIN_MODEL_SOURCE")
            or os.environ.get("SWIN_MODEL_ID")
            or (LOCAL_MODEL_DIR if _local_model_dir_is_complete() else HF_MODEL_ID)
        )
        weights = Path(weights_path).expanduser() if weights_path else None

        source_path = Path(source).expanduser()
        if source_path.is_file():
            weights = source_path
            source = str(
                os.environ.get("SWIN_CONFIG_SOURCE")
                or os.environ.get("SWIN_MODEL_ID")
                or HF_MODEL_ID
            )
        elif source_path.is_dir():
            has_config = (source_path / "config.json").exists()
            has_weights = (source_path / "pytorch_model.bin").exists()
            if not has_config and has_weights and weights is None:
                weights = source_path / "pytorch_model.bin"
                source = str(
                    os.environ.get("SWIN_CONFIG_SOURCE")
                    or os.environ.get("SWIN_MODEL_ID")
                    or HF_MODEL_ID
                )

        if weights is not None and not weights.exists():
            raise FileNotFoundError(f"Swin weights not found: {weights}")

        return source, weights

    def _load_model(self):
        if self.weights_path is None:
            return AutoModelForImageClassification.from_pretrained(self.model_source)

        config = AutoConfig.from_pretrained(self.model_source)
        model = AutoModelForImageClassification.from_config(config)
        raw_checkpoint = torch.load(self.weights_path, map_location="cpu")
        state_dict = _clean_state_dict_keys(_extract_state_dict(raw_checkpoint))
        model.load_state_dict(state_dict, strict=True)
        return model

    def _build_index_to_class_id(self) -> list[str]:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        num_labels = int(getattr(self.model.config, "num_labels", len(id2label) or 0))
        if num_labels != len(CANONICAL_CLASS_IDS):
            raise ValueError(
                "Swin model must expose exactly 7 labels for the Dermalyze API; "
                f"got {num_labels}."
            )

        index_to_class_id = []
        for index in range(num_labels):
            raw_label = id2label.get(index, id2label.get(str(index), f"LABEL_{index}"))
            index_to_class_id.append(_canonical_label(raw_label, index))

        if set(index_to_class_id) != set(CANONICAL_CLASS_IDS):
            raise ValueError(
                "Swin labels do not map cleanly to Dermalyze class ids: "
                f"{index_to_class_id}"
            )
        return index_to_class_id

    def _to_pil_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
    ) -> Image.Image:
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _predict_probabilities(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = F.softmax(logits, dim=-1)[0]

        if self.device.type == "mps":
            torch.mps.synchronize()
        return probabilities.detach().cpu().numpy()

    def _format_prediction(
        self,
        probabilities: np.ndarray,
        top_k: int,
        include_disclaimer: bool,
    ) -> Dict[str, Any]:
        all_probs = {class_id: 0.0 for class_id in CANONICAL_CLASS_IDS}
        for index, probability in enumerate(probabilities):
            all_probs[self.index_to_class_id[index]] = float(probability)

        predicted_index = int(np.argmax(probabilities))
        predicted_class = self.index_to_class_id[predicted_index]
        top_k_indices = np.argsort(probabilities)[::-1][: max(1, min(top_k, len(probabilities)))]

        result: Dict[str, Any] = {
            "predicted_class": predicted_class,
            "predicted_class_description": CLASS_LABELS[predicted_class],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": all_probs,
            "top_k_predictions": [
                {
                    "class": self.index_to_class_id[int(index)],
                    "description": CLASS_LABELS[self.index_to_class_id[int(index)]],
                    "probability": float(probabilities[index]),
                }
                for index in top_k_indices
            ],
        }

        if include_disclaimer:
            result["disclaimer"] = self.DISCLAIMER
        return result

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
        del metadata, include_gradcam, gradcam_alpha, gradcam_colormap
        pil_image = self._to_pil_image(image)
        probabilities = self._predict_probabilities(pil_image)
        return self._format_prediction(probabilities, top_k, include_disclaimer)

    def _tta_images(self, image: Image.Image, tta_mode: str) -> list[Image.Image]:
        base_images = [
            image,
            ImageOps.mirror(image),
            ImageOps.flip(image),
            ImageOps.flip(ImageOps.mirror(image)),
        ]

        if tta_mode == "light":
            return base_images

        medium_images = base_images + [
            image.rotate(90, expand=True),
            image.rotate(180, expand=True),
            image.rotate(270, expand=True),
            _center_crop_scaled(image),
        ]
        if tta_mode == "medium":
            return medium_images
        if tta_mode == "full":
            return medium_images + _corner_crops_scaled(image)

        raise ValueError("Invalid tta_mode: {!r}. Expected light, medium, or full.".format(tta_mode))

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
        del metadata, use_clahe_tta, clahe_clip_limit, clahe_grid_size
        pil_image = self._to_pil_image(image)
        probs_collection = [
            self._predict_probabilities(tta_image)
            for tta_image in self._tta_images(pil_image, tta_mode)
        ]
        probs_array = np.array(probs_collection)

        if aggregation == "mean":
            final_probs = np.mean(probs_array, axis=0)
        elif aggregation == "geometric_mean":
            final_probs = np.exp(np.mean(np.log(probs_array + 1e-10), axis=0))
            final_probs = final_probs / final_probs.sum()
        elif aggregation == "max":
            final_probs = np.max(probs_array, axis=0)
            final_probs = final_probs / final_probs.sum()
        else:
            raise ValueError(
                "Invalid aggregation: {!r}. Expected mean, geometric_mean, or max.".format(
                    aggregation
                )
            )

        result = self._format_prediction(final_probs, top_k, include_disclaimer)
        result.update(
            {
                "tta_mode": tta_mode,
                "tta_augmentations": len(probs_collection),
                "aggregation_method": aggregation,
                "use_clahe_tta": False,
            }
        )
        return result

    def generate_gradcam(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
        target_class: Optional[int] = None,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> None:
        del image, target_class, alpha, colormap
        return None


def collect_images(path: str) -> list[str]:
    input_path = Path(path)
    if input_path.is_file():
        return [str(input_path)]
    if input_path.is_dir():
        return sorted(
            str(file_path)
            for file_path in input_path.rglob("*")
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise FileNotFoundError(f"Path not found: {path}")


def save_csv(rows: list[dict[str, object]], output_path: str) -> None:
    fieldnames = ["file", "rank", "class", "description", "probability"]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify skin lesion images with the Dermalyze Swin model.",
    )
    parser.add_argument("input", help="Path to an image file or directory.")
    parser.add_argument(
        "--model-source",
        default=None,
        help="Local model directory or Hugging Face model id. Defaults to inference_service/swin.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional standalone pytorch_model.bin path when config/processor come from model source.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps. Auto-detected by default.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = SwinSkinLesionPredictor(
        model_source=args.model_source,
        weights_path=args.weights,
        device=args.device,
    )

    rows: list[dict[str, object]] = []
    for image_path in collect_images(args.input):
        result = predictor.predict(image_path, top_k=args.top_k, include_disclaimer=False)
        print(image_path)
        for index, prediction in enumerate(result["top_k_predictions"], start=1):
            print(
                "  {rank}. {label}: {probability:.2%}".format(
                    rank=index,
                    label=prediction["description"],
                    probability=prediction["probability"],
                )
            )
            rows.append(
                {
                    "file": image_path,
                    "rank": index,
                    "class": prediction["class"],
                    "description": prediction["description"],
                    "probability": prediction["probability"],
                }
            )

    if args.output:
        save_csv(rows, args.output)


if __name__ == "__main__":
    main()
