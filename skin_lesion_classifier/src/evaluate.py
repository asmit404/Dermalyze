"""
Evaluation Module for Skin Lesion Classification.

This script provides comprehensive evaluation of trained models including:
- Per-class and macro-averaged metrics
- Confusion matrix analysis
- Model calibration assessment
- Visualization of results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import (
    HAM10000Dataset,
    get_transforms,
    CLASS_LABELS,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.models.efficientnet import SkinLesionClassifier
from src.tta_constants import TTA_AUG_COUNTS


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None


def _apply_clahe_batch(
    images: torch.Tensor,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> torch.Tensor:
    """Apply CLAHE to a normalized batch tensor (N, C, H, W)."""
    if cv2 is None:
        raise RuntimeError(
            "CLAHE-TTA requested but OpenCV is not installed. "
            "Install opencv-python or opencv-python-headless."
        )

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )

    device = images.device
    dtype = images.dtype

    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)

    denorm = (images * std + mean).clamp(0.0, 1.0)
    denorm_np = (denorm.detach().permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(
        np.uint8
    )

    clahe_batch = []
    for image_rgb in denorm_np:
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(image_lab)
        l_channel = clahe.apply(l_channel)
        image_lab = cv2.merge((l_channel, a_channel, b_channel))
        image_rgb_clahe = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
        clahe_batch.append(image_rgb_clahe)

    clahe_np = np.stack(clahe_batch).astype(np.float32) / 255.0
    clahe_tensor = torch.from_numpy(clahe_np).permute(0, 3, 1, 2).to(device=device)

    return ((clahe_tensor - mean) / std).to(dtype=dtype)


def _zoom_crop_batch(
    images: torch.Tensor,
    position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"],
    scale: float = 1.1,
) -> torch.Tensor:
    """Apply zoom-in then crop back to original size at a given position."""
    _, _, height, width = images.shape
    zoom_h = max(int(round(height * scale)), height)
    zoom_w = max(int(round(width * scale)), width)

    zoomed = F.interpolate(
        images,
        size=(zoom_h, zoom_w),
        mode="bilinear",
        align_corners=False,
    )

    if position == "center":
        y0 = (zoom_h - height) // 2
        x0 = (zoom_w - width) // 2
    elif position == "top_left":
        y0 = 0
        x0 = 0
    elif position == "top_right":
        y0 = 0
        x0 = zoom_w - width
    elif position == "bottom_left":
        y0 = zoom_h - height
        x0 = 0
    else:  # bottom_right
        y0 = zoom_h - height
        x0 = zoom_w - width

    return zoomed[:, :, y0 : y0 + height, x0 : x0 + width]


def get_tta_aug_count(tta_mode: Literal["light", "medium", "full"]) -> int:
    """Return number of tensor-space TTA branches for a mode (excluding CLAHE)."""
    try:
        return TTA_AUG_COUNTS[tta_mode]
    except KeyError as exc:
        valid_modes = ", ".join(TTA_AUG_COUNTS.keys())
        raise ValueError(
            f"Invalid tta_mode: {tta_mode!r}. Expected one of: {valid_modes}."
        ) from exc


def compute_ensemble_weights_from_metrics(
    metrics_list: List[Dict[str, float]],
    metric_name: str = "val_acc",
) -> np.ndarray:
    """
    Compute ensemble weights based on validation metrics.

    Args:
        metrics_list: List of metric dictionaries from checkpoints
        metric_name: Metric to use for weighting (val_acc or val_loss)

    Returns:
        Normalized weights array
    """
    # Extract the metric values
    metric_values = []
    for metrics in metrics_list:
        value = metrics.get(metric_name)
        if value is None:
            # Try alternate metric names
            if metric_name == "val_acc":
                value = metrics.get("val_accuracy", metrics.get("accuracy"))
            elif metric_name == "val_loss":
                value = metrics.get("loss")

        if value is None:
            # If no metrics found, fall back to uniform weights
            logger.warning(
                f"Metric '{metric_name}' not found in checkpoint. "
                "Using uniform weights instead."
            )
            return np.ones(len(metrics_list)) / len(metrics_list)

        metric_values.append(float(value))

    metric_values = np.array(metric_values)

    # Compute weights based on metric
    if "loss" in metric_name:
        # For loss, lower is better - use inverse
        # Add small epsilon to avoid division by zero
        weights = 1.0 / (metric_values + 1e-8)
    else:
        # For accuracy, higher is better - use directly
        weights = metric_values

    # Normalize to sum to 1
    weights = weights / weights.sum()

    return weights


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[SkinLesionClassifier, Dict[str, Any], Dict[str, float]]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, config, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    metrics = checkpoint.get("metrics", {})

    # Create model with config
    model_config = config.get("model", {})
    model = SkinLesionClassifier(
        num_classes=model_config.get("num_classes", 7),
        pretrained=False,  # We're loading weights from checkpoint
        dropout_rate=model_config.get("dropout_rate", 0.3),
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config, metrics


@torch.no_grad()
def get_predictions(
    model: SkinLesionClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions for a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run inference on

    Returns:
        Tuple of (true_labels, predicted_labels, predicted_probabilities)
    """
    all_targets = []
    all_preds = []
    all_probs = []

    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)

        # Get predictions
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_targets.extend(targets.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
    )


@torch.no_grad()
def get_predictions_with_tta(
    model: SkinLesionClassifier,
    dataloader: DataLoader,
    device: torch.device,
    tta_mode: Literal["light", "medium", "full"] = "medium",
    aggregation: Literal["mean", "geometric_mean", "max"] = "mean",
    use_clahe_tta: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions with Test-Time Augmentation.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run inference on
        tta_mode: TTA complexity (light, medium, full)
        aggregation: How to aggregate TTA predictions
        use_clahe_tta: Whether to add a CLAHE-augmented branch during TTA
        clahe_clip_limit: CLAHE clip limit
        clahe_grid_size: CLAHE tile grid size

    Returns:
        Tuple of (true_labels, predicted_labels, predicted_probabilities)
    """
    all_targets = []
    all_preds = []
    all_probs = []

    if use_clahe_tta and cv2 is None:
        raise RuntimeError(
            "CLAHE-TTA requested but OpenCV is not installed. "
            "Install opencv-python or opencv-python-headless."
        )

    for images, targets in tqdm(dataloader, desc=f"Evaluating with TTA ({tta_mode})"):
        batch_size = images.size(0)

        # Collect TTA predictions for each image in batch
        batch_tta_probs = []

        images_device = images.to(device)

        total_aug_count = get_tta_aug_count(tta_mode)

        for aug_idx in range(total_aug_count):
            # Apply TTA transform to each image
            # Note: We need to denormalize, apply transform, and renormalize
            # For simplicity, we'll use the original images and standard augmentations

            aug_images = images_device

            # Simple augmentations that can be done in tensor space
            if aug_idx == 1:  # Horizontal flip
                aug_images = torch.flip(aug_images, dims=[3])
            elif aug_idx == 2:  # Vertical flip
                aug_images = torch.flip(aug_images, dims=[2])
            elif aug_idx == 3:  # Both flips
                aug_images = torch.flip(aug_images, dims=[2, 3])
            elif aug_idx == 4:  # 90° rotation
                aug_images = torch.rot90(aug_images, k=1, dims=[2, 3])
            elif aug_idx == 5:  # 180° rotation
                aug_images = torch.rot90(aug_images, k=2, dims=[2, 3])
            elif aug_idx == 6:  # 270° rotation
                aug_images = torch.rot90(aug_images, k=3, dims=[2, 3])
            elif aug_idx == 7:  # center zoom crop
                aug_images = _zoom_crop_batch(aug_images, position="center", scale=1.1)
            elif aug_idx == 8:  # top-left zoom crop (full mode only)
                aug_images = _zoom_crop_batch(
                    aug_images, position="top_left", scale=1.1
                )
            elif aug_idx == 9:  # top-right zoom crop (full mode only)
                aug_images = _zoom_crop_batch(
                    aug_images, position="top_right", scale=1.1
                )
            elif aug_idx == 10:  # bottom-left zoom crop (full mode only)
                aug_images = _zoom_crop_batch(
                    aug_images, position="bottom_left", scale=1.1
                )
            elif aug_idx == 11:  # bottom-right zoom crop (full mode only)
                aug_images = _zoom_crop_batch(
                    aug_images, position="bottom_right", scale=1.1
                )
            # aug_idx == 0: use original

            logits = model(aug_images)
            probs = F.softmax(logits, dim=1)
            batch_tta_probs.append(probs.cpu().numpy())

        if use_clahe_tta:
            clahe_images = _apply_clahe_batch(
                images,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_grid_size,
            ).to(device)
            logits = model(clahe_images)
            probs = F.softmax(logits, dim=1)
            batch_tta_probs.append(probs.cpu().numpy())

        # Aggregate TTA predictions
        batch_tta_probs = np.array(
            batch_tta_probs
        )  # Shape: (n_augs, batch_size, n_classes)
        batch_tta_probs = np.transpose(
            batch_tta_probs, (1, 0, 2)
        )  # Shape: (batch_size, n_augs, n_classes)

        if aggregation == "mean":
            final_probs = np.mean(batch_tta_probs, axis=1)
        elif aggregation == "geometric_mean":
            final_probs = np.exp(np.mean(np.log(batch_tta_probs + 1e-10), axis=1))
            final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
        else:  # max
            final_probs = np.max(batch_tta_probs, axis=1)

        preds = np.argmax(final_probs, axis=1)

        all_targets.extend(targets.numpy())
        all_preds.extend(preds)
        all_probs.extend(final_probs)

    return (
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
    )


@torch.no_grad()
def get_ensemble_predictions(
    models: List[SkinLesionClassifier],
    dataloader: DataLoader,
    device: torch.device,
    weights: Optional[List[float]] = None,
    aggregation: Literal["mean", "weighted_mean", "geometric_mean"] = "weighted_mean",
    use_tta: bool = False,
    tta_mode: Literal["light", "medium", "full"] = "medium",
    tta_aggregation: Literal["mean", "geometric_mean", "max"] = "mean",
    use_clahe_tta: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ensemble predictions from multiple models.

    Args:
        models: List of trained models
        dataloader: DataLoader for evaluation
        device: Device to run inference on
        weights: Optional weights for each model
        aggregation: How to combine model predictions
        use_tta: Whether to use TTA for each model
        tta_mode: TTA complexity if use_tta=True
        tta_aggregation: How to aggregate TTA predictions
        use_clahe_tta: Whether to add a CLAHE-augmented branch during TTA
        clahe_clip_limit: CLAHE clip limit
        clahe_grid_size: CLAHE tile grid size

    Returns:
        Tuple of (true_labels, predicted_labels, predicted_probabilities)
    """
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()

    all_targets = []
    all_preds = []
    all_probs = []

    desc = f"Ensemble evaluation ({len(models)} models"
    if use_tta:
        desc += f", TTA-{tta_mode}"
    desc += ")"

    for images, targets in tqdm(dataloader, desc=desc):
        # Collect predictions from all models
        model_probs_list = []
        clahe_images = None
        # Precompute once per batch only when it is guaranteed to be used.
        if use_tta and use_clahe_tta:
            clahe_images = _apply_clahe_batch(
                images,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_grid_size,
            ).to(device)

        for model in models:
            if use_tta:
                # Use TTA for this model
                # For batch processing, we'll use a simpler approach
                images_device = images.to(device)

                # Collect TTA predictions
                tta_probs = []
                total_aug_count = get_tta_aug_count(tta_mode)

                for aug_idx in range(total_aug_count):
                    aug_images = images_device

                    if aug_idx == 1:  # H flip
                        aug_images = torch.flip(aug_images, dims=[3])
                    elif aug_idx == 2:  # V flip
                        aug_images = torch.flip(aug_images, dims=[2])
                    elif aug_idx == 3:  # Both flips
                        aug_images = torch.flip(aug_images, dims=[2, 3])
                    elif aug_idx == 4 and tta_mode in ["medium", "full"]:  # 90°
                        aug_images = torch.rot90(aug_images, k=1, dims=[2, 3])
                    elif aug_idx == 5 and tta_mode in ["medium", "full"]:  # 180°
                        aug_images = torch.rot90(aug_images, k=2, dims=[2, 3])
                    elif aug_idx == 6 and tta_mode in ["medium", "full"]:  # 270°
                        aug_images = torch.rot90(aug_images, k=3, dims=[2, 3])
                    elif aug_idx == 7 and tta_mode in [
                        "medium",
                        "full",
                    ]:  # center zoom crop
                        aug_images = _zoom_crop_batch(
                            aug_images, position="center", scale=1.1
                        )
                    elif aug_idx == 8 and tta_mode == "full":  # top-left zoom crop
                        aug_images = _zoom_crop_batch(
                            aug_images, position="top_left", scale=1.1
                        )
                    elif aug_idx == 9 and tta_mode == "full":  # top-right zoom crop
                        aug_images = _zoom_crop_batch(
                            aug_images, position="top_right", scale=1.1
                        )
                    elif aug_idx == 10 and tta_mode == "full":  # bottom-left zoom crop
                        aug_images = _zoom_crop_batch(
                            aug_images, position="bottom_left", scale=1.1
                        )
                    elif aug_idx == 11 and tta_mode == "full":  # bottom-right zoom crop
                        aug_images = _zoom_crop_batch(
                            aug_images, position="bottom_right", scale=1.1
                        )

                    logits = model(aug_images)
                    probs = F.softmax(logits, dim=1)
                    tta_probs.append(probs.cpu().numpy())

                # Aggregate TTA
                tta_probs = np.array(tta_probs)  # (n_augs, batch_size, n_classes)
                tta_probs = np.transpose(
                    tta_probs, (1, 0, 2)
                )  # (batch_size, n_augs, n_classes)

                if use_clahe_tta and clahe_images is not None:
                    logits = model(clahe_images)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    probs = probs[:, np.newaxis, :]  # (batch_size, 1, n_classes)
                    tta_probs = np.concatenate([tta_probs, probs], axis=1)

                if tta_aggregation == "mean":
                    final_probs = np.mean(tta_probs, axis=1)
                elif tta_aggregation == "geometric_mean":
                    final_probs = np.exp(np.mean(np.log(tta_probs + 1e-10), axis=1))
                    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
                else:  # max
                    final_probs = np.max(tta_probs, axis=1)

                model_probs_list.append(final_probs)
            else:
                # Standard prediction
                images_device = images.to(device)
                logits = model(images_device)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                model_probs_list.append(probs)

        # Aggregate model predictions
        model_probs_list = np.array(
            model_probs_list
        )  # (n_models, batch_size, n_classes)
        model_probs_list = np.transpose(
            model_probs_list, (1, 0, 2)
        )  # (batch_size, n_models, n_classes)

        if aggregation == "mean":
            final_probs = np.mean(model_probs_list, axis=1)
        elif aggregation == "weighted_mean":
            final_probs = np.average(model_probs_list, axis=1, weights=weights)
        else:  # geometric_mean
            final_probs = np.exp(np.mean(np.log(model_probs_list + 1e-10), axis=1))
            final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)

        preds = np.argmax(final_probs, axis=1)

        all_targets.extend(targets.numpy())
        all_preds.extend(preds)
        all_probs.extend(final_probs)

    return (
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        class_names: List of class names

    Returns:
        Dictionary of metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Macro-averaged metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Weighted metrics
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    # ROC-AUC (one-vs-rest)
    try:
        # For multi-class, compute OvR AUC
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        per_class_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average=None)
    except ValueError:
        roc_auc = None
        per_class_auc = [None] * len(class_names)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Per-class metrics dictionary
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
            "roc_auc": (
                float(per_class_auc[i]) if per_class_auc[i] is not None else None
            ),
        }

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "roc_auc_macro": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "classification_report": report,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        output_path: Path to save the plot
        normalize: Whether to normalize the matrix
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix"

    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix to: {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    output_path: Path,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """
    Plot ROC curves for all classes.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: List of class names
        output_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Binarize true labels for multi-class ROC
    n_classes = len(class_names)
    y_true_bin = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_true_bin[i, label] = 1

    # Plot ROC curve for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{class_name} (AUC = {auc:.3f})",
        )

    # Plot diagonal
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curves to: {output_path}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 8),
) -> Dict[str, float]:
    """
    Plot reliability diagram (calibration curve).

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save the plot
        n_bins: Number of bins for calibration
        figsize: Figure size

    Returns:
        Dictionary with calibration metrics
    """
    plt.figure(figsize=figsize)

    # Get max probabilities and correctness
    y_prob_max = np.max(y_prob, axis=1)
    y_pred = np.argmax(y_prob, axis=1)
    correct = (y_pred == y_true).astype(int)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(correct, y_prob_max, n_bins=n_bins)

    # Calculate Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob_max >= bin_boundaries[i]) & (y_prob_max < bin_boundaries[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(correct[mask])
            bin_conf = np.mean(y_prob_max[mask])
            ece += np.sum(mask) * np.abs(bin_acc - bin_conf)
    ece /= len(y_true)

    # Plot
    plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfectly Calibrated")
    plt.plot(
        prob_pred,
        prob_true,
        "o-",
        color="tab:blue",
        linewidth=2,
        markersize=8,
        label=f"Model (ECE = {ece:.4f})",
    )

    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives", fontsize=12)
    plt.title("Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved calibration curve to: {output_path}")

    return {
        "expected_calibration_error": float(ece),
        "mean_confidence": float(np.mean(y_prob_max)),
        "accuracy": float(np.mean(correct)),
    }


def plot_per_class_metrics(
    metrics: Dict[str, Any],
    class_names: List[str],
    output_path: Path,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """
    Plot per-class precision, recall, and F1-score.

    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        output_path: Path to save the plot
        figsize: Figure size
    """
    per_class = metrics["per_class_metrics"]

    # Extract metrics
    precision = [per_class[c]["precision"] for c in class_names]
    recall = [per_class[c]["recall"] for c in class_names]
    f1 = [per_class[c]["f1_score"] for c in class_names]

    # Create grouped bar chart
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width, precision, width, label="Precision", color="tab:blue")
    bars2 = ax.bar(x, recall, width, label="Recall", color="tab:orange")
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="tab:green")

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved per-class metrics to: {output_path}")


def evaluate(
    checkpoint_path: Union[Path, List[Path]],
    test_csv: Path,
    images_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    use_tta: bool = False,
    tta_mode: Literal["light", "medium", "full"] = "medium",
    tta_aggregation: Literal["mean", "geometric_mean", "max"] = "mean",
    use_clahe_tta: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8,
    use_ensemble: bool = False,
    ensemble_weights: Optional[List[float]] = None,
    ensemble_aggregation: Literal[
        "mean", "weighted_mean", "geometric_mean"
    ] = "weighted_mean",
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        checkpoint_path: Path to model checkpoint (or list for ensemble)
        test_csv: Path to test CSV file
        images_dir: Path to images directory
        output_dir: Output directory for results
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        use_tta: Whether to use test-time augmentation
        tta_mode: TTA complexity (light/medium/full)
        tta_aggregation: How to aggregate TTA predictions
        use_clahe_tta: Whether to add CLAHE as an extra TTA branch
        clahe_clip_limit: CLAHE clip limit
        clahe_grid_size: CLAHE tile grid size
        use_ensemble: Whether to use ensemble evaluation
        ensemble_weights: Optional custom weights for models. If None and
            ensemble_aggregation='weighted_mean', weights are automatically
            computed from validation accuracy in checkpoints.
        ensemble_aggregation: How to aggregate ensemble predictions
            - 'mean': Uniform averaging
            - 'weighted_mean': Weight by val accuracy (auto-computed if no weights)
            - 'geometric_mean': Geometric average

    Returns:
        Dictionary of evaluation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load model(s)
    if use_ensemble or isinstance(checkpoint_path, list):
        if not isinstance(checkpoint_path, list):
            raise ValueError("Ensemble mode requires list of checkpoint paths")

        logger.info(f"Loading ensemble of {len(checkpoint_path)} models...")
        models = []
        metrics_list = []
        for i, cp in enumerate(checkpoint_path):
            model, config, metrics = load_model(Path(cp), device)
            models.append(model)
            metrics_list.append(metrics)
            logger.info(f"  Model {i+1}: {cp}")

        # Compute weights
        if ensemble_weights:
            logger.info(f"Using custom ensemble weights: {ensemble_weights}")
        elif ensemble_aggregation == "weighted_mean":
            # Automatically compute weights from validation metrics
            logger.info("Computing ensemble weights from validation metrics...")
            ensemble_weights = compute_ensemble_weights_from_metrics(
                metrics_list, metric_name="val_acc"
            )
            logger.info(f"Auto-computed ensemble weights: {ensemble_weights}")
            # Log which model has highest weight
            best_idx = np.argmax(ensemble_weights)
            logger.info(
                f"  Highest weight (model {best_idx+1}): {ensemble_weights[best_idx]:.4f}"
            )

        eval_mode = "ensemble"
        if use_tta:
            eval_mode += f" + TTA-{tta_mode}"
    else:
        logger.info(f"Loading model from: {checkpoint_path}")
        model, config, metrics = load_model(checkpoint_path, device)
        eval_mode = "standard"
        if use_tta:
            eval_mode = f"TTA-{tta_mode}"

    logger.info(f"Evaluation mode: {eval_mode}")
    if use_tta and use_clahe_tta:
        logger.info(
            "CLAHE-TTA enabled (clip_limit=%.2f, grid_size=%d)",
            clahe_clip_limit,
            clahe_grid_size,
        )

    # Load test data
    logger.info(f"Loading test data from: {test_csv}")
    test_df = pd.read_csv(test_csv)

    # Create test dataset
    image_size = config.get("model", {}).get("image_size", 224)
    test_dataset = HAM10000Dataset(
        df=test_df,
        images_dir=images_dir,
        transform=get_transforms("test", image_size),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),  # Pin memory only works on CUDA
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Get predictions based on mode
    logger.info(f"Running inference ({eval_mode})...")

    if use_ensemble or isinstance(checkpoint_path, list):
        y_true, y_pred, y_prob = get_ensemble_predictions(
            models=models,
            dataloader=test_loader,
            device=device,
            weights=ensemble_weights,
            aggregation=ensemble_aggregation,
            use_tta=use_tta,
            tta_mode=tta_mode,
            tta_aggregation=tta_aggregation,
            use_clahe_tta=use_clahe_tta,
            clahe_clip_limit=clahe_clip_limit,
            clahe_grid_size=clahe_grid_size,
        )
    elif use_tta:
        y_true, y_pred, y_prob = get_predictions_with_tta(
            model=model,
            dataloader=test_loader,
            device=device,
            tta_mode=tta_mode,
            aggregation=tta_aggregation,
            use_clahe_tta=use_clahe_tta,
            clahe_clip_limit=clahe_clip_limit,
            clahe_grid_size=clahe_grid_size,
        )
    else:
        y_true, y_pred, y_prob = get_predictions(model, test_loader, device)

    # Class names in order
    class_names = [IDX_TO_LABEL[i] for i in range(len(CLASS_LABELS))]

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)

    # Compute calibration metrics
    calibration_metrics = plot_calibration_curve(
        y_true, y_prob, output_dir / "calibration_curve.png"
    )
    metrics["calibration"] = calibration_metrics

    # Generate plots
    logger.info("Generating plots...")

    # Confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
    plot_confusion_matrix(
        cm, class_names, output_dir / "confusion_matrix_raw.png", normalize=False
    )

    # ROC curves
    plot_roc_curves(y_true, y_prob, class_names, output_dir / "roc_curves.png")

    # Per-class metrics
    plot_per_class_metrics(metrics, class_names, output_dir / "per_class_metrics.png")

    # Save metrics to JSON
    metrics["evaluation_mode"] = eval_mode
    if use_tta:
        metrics["tta_config"] = {
            "mode": tta_mode,
            "aggregation": tta_aggregation,
            "use_clahe_tta": bool(use_clahe_tta),
            "clahe_clip_limit": float(clahe_clip_limit),
            "clahe_grid_size": int(clahe_grid_size),
        }
    if use_ensemble or isinstance(checkpoint_path, list):
        metrics["ensemble_config"] = {
            "num_models": len(models),
            "aggregation": ensemble_aggregation,
            "weights": (
                ensemble_weights.tolist()
                if isinstance(ensemble_weights, np.ndarray)
                else ensemble_weights
            ),
        }

    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Mode:               {eval_mode}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:       {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:     {metrics['macro_f1']:.4f}")
    if metrics["roc_auc_macro"]:
        print(f"ROC-AUC (Macro):    {metrics['roc_auc_macro']:.4f}")
    print(
        f"ECE (Calibration):  {metrics['calibration']['expected_calibration_error']:.4f}"
    )
    print("=" * 60)

    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    for class_name in class_names:
        m = metrics["per_class_metrics"][class_name]
        print(
            f"{class_name:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1_score']:>10.4f} {m['support']:>10d}"
        )
    print("-" * 60)

    return metrics


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate skin lesion classifier with optional TTA and Ensemble"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Optional config file to source evaluation defaults (e.g., CLAHE-TTA)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to model checkpoint(s). Multiple checkpoints enable ensemble.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="Path to test split CSV file",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Path to images directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--use-tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use test-time augmentation",
    )
    parser.add_argument(
        "--tta-mode",
        choices=["light", "medium", "full"],
        default=None,
        help="TTA complexity (light: 4 augs, medium: 8 augs, full: all)",
    )
    parser.add_argument(
        "--tta-aggregation",
        choices=["mean", "geometric_mean", "max"],
        default=None,
        help="How to aggregate TTA predictions",
    )
    parser.add_argument(
        "--use-clahe-tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add CLAHE-processed branch during TTA (requires OpenCV)",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=None,
        help="CLAHE clip limit used when --use-clahe-tta is enabled",
    )
    parser.add_argument(
        "--clahe-grid-size",
        type=int,
        default=None,
        help="CLAHE tile grid size used when --use-clahe-tta is enabled",
    )
    parser.add_argument(
        "--ensemble-weights",
        type=float,
        nargs="+",
        help="Optional weights for ensemble models (must match number of checkpoints)",
    )
    parser.add_argument(
        "--ensemble-aggregation",
        choices=["mean", "weighted_mean", "geometric_mean"],
        default="weighted_mean",
        help="How to aggregate ensemble predictions",
    )
    args = parser.parse_args()

    config_defaults: Dict[str, Any] = {}
    if args.config is not None and args.config.exists():
        with open(args.config, "r") as f:
            config_defaults = yaml.safe_load(f) or {}

    tta_defaults = config_defaults.get("evaluation", {}).get("tta", {})

    use_tta = (
        bool(args.use_tta)
        if args.use_tta is not None
        else bool(tta_defaults.get("use_tta", False))
    )
    tta_mode = (
        str(args.tta_mode)
        if args.tta_mode is not None
        else str(tta_defaults.get("mode", "medium"))
    )
    tta_aggregation = (
        str(args.tta_aggregation)
        if args.tta_aggregation is not None
        else str(tta_defaults.get("aggregation", "mean"))
    )

    use_clahe_tta = (
        bool(args.use_clahe_tta)
        if args.use_clahe_tta is not None
        else bool(tta_defaults.get("use_clahe_tta", False))
    )
    clahe_clip_limit = (
        float(args.clahe_clip_limit)
        if args.clahe_clip_limit is not None
        else float(tta_defaults.get("clahe_clip_limit", 2.0))
    )
    clahe_grid_size = (
        int(args.clahe_grid_size)
        if args.clahe_grid_size is not None
        else int(tta_defaults.get("clahe_grid_size", 8))
    )

    # Determine if using ensemble
    use_ensemble = len(args.checkpoint) > 1
    checkpoint_path = args.checkpoint if use_ensemble else args.checkpoint[0]

    evaluate(
        checkpoint_path=checkpoint_path,
        test_csv=args.test_csv,
        images_dir=args.images_dir,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_tta=use_tta,
        tta_mode=tta_mode,
        tta_aggregation=tta_aggregation,
        use_clahe_tta=use_clahe_tta,
        clahe_clip_limit=clahe_clip_limit,
        clahe_grid_size=clahe_grid_size,
        use_ensemble=use_ensemble,
        ensemble_weights=args.ensemble_weights,
        ensemble_aggregation=args.ensemble_aggregation,
    )


if __name__ == "__main__":
    main()
