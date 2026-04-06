"""
HAM10000 Dataset Module for Skin Lesion Classification.

This module provides dataset classes and utilities for loading, preprocessing,
and augmenting the HAM10000 dermoscopic image dataset. It implements:
- Lesion-aware stratified splitting to prevent data leakage
- Class-balanced sampling for handling imbalanced data
- Reproducible augmentation pipelines
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.data.metadata_encoder import MetadataEncoder

# HAM10000 class labels and their full names
CLASS_LABELS = {
    "akiec": "Actinic keratoses / Intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

# Numeric mapping for labels
LABEL_TO_IDX = {label: idx for idx, label in enumerate(sorted(CLASS_LABELS.keys()))}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

# ImageNet normalization statistics (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000 skin lesion images.

    Attributes:
        df: DataFrame containing image_id, label, and optional lesion_id
        images_dir: Path to directory containing images
        transform: Optional transforms to apply to images
        target_transform: Optional transforms to apply to labels
        use_metadata: Whether to return metadata along with images
        metadata_columns: List of metadata columns to include
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path | str,
        masks_dir: Path | str | None = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_metadata: bool = False,
        metadata_columns: Optional[list[str]] = None,
        metadata_encoder: Optional[MetadataEncoder] = None,
        strict_labels: bool = True,
        use_segmentation_roi_crop: bool = False,
        segmentation_mask_threshold: int = 10,
        segmentation_crop_margin: float = 0.1,
        segmentation_required: bool = False,
        mask_filename_suffixes: Optional[list[str]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            df: DataFrame with columns 'image_id' and 'label' (and optionally 'lesion_id')
            images_dir: Directory containing the image files
            transform: Transformations to apply to images
            target_transform: Transformations to apply to labels
            use_metadata: Whether to load and return metadata (age, sex, localization)
            metadata_columns: List of metadata columns to include (default: ['age', 'sex', 'localization'])
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir is not None else None
        self.transform = transform
        self.target_transform = target_transform
        self.use_metadata = use_metadata
        self.metadata_encoder = metadata_encoder
        self.strict_labels = strict_labels
        self.use_segmentation_roi_crop = bool(use_segmentation_roi_crop)
        self.segmentation_mask_threshold = int(segmentation_mask_threshold)
        self.segmentation_crop_margin = max(float(segmentation_crop_margin), 0.0)
        self.segmentation_required = bool(segmentation_required)
        self.mask_filename_suffixes = (
            list(mask_filename_suffixes)
            if mask_filename_suffixes is not None
            else ["", "_segmentation", "_mask"]
        )
        self.mask_search_dirs: list[Path] = []
        if self.masks_dir is not None:
            self.mask_search_dirs.append(self.masks_dir)
            for nested_name in ("images", "masks"):
                nested_dir = self.masks_dir / nested_name
                if nested_dir.is_dir():
                    self.mask_search_dirs.append(nested_dir)

        if self.use_segmentation_roi_crop and self.masks_dir is None:
            raise ValueError(
                "use_segmentation_roi_crop=True requires masks_dir to be provided"
            )

        # Default metadata columns from HAM10000 dataset
        if metadata_columns is None:
            self.metadata_columns = ['age', 'sex', 'localization']
        else:
            self.metadata_columns = metadata_columns

        # Validate metadata columns if requested
        if self.use_metadata:
            self._validate_metadata_columns()

        # Validate that all images exist
        self._validate_images()

        # Optionally require masks for all samples
        if self.use_segmentation_roi_crop and self.segmentation_required:
            self._validate_masks()

    def _validate_metadata_columns(self) -> None:
        """Validate that requested metadata columns exist in the DataFrame."""
        missing_cols = [col for col in self.metadata_columns if col not in self.df.columns]
        if missing_cols:
            available_cols = list(self.df.columns)
            raise ValueError(
                f"Requested metadata columns {missing_cols} not found in dataset. "
                f"Available columns: {available_cols}"
            )

    def _validate_images(self) -> None:
        """Validate that all referenced images exist on disk."""
        missing = []
        for img_id in self.df["image_id"].values:
            img_path = self._get_image_path(img_id)
            if not img_path.exists():
                missing.append(img_id)

        if missing and len(missing) <= 10:
            raise FileNotFoundError(f"Missing images: {missing}")
        elif missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} images. First 10: {missing[:10]}"
            )

    def _get_image_path(self, image_id: str) -> Path:
        """Get the full path to an image file."""
        # Try common extensions
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            path = self.images_dir / f"{image_id}{ext}"
            if path.exists():
                return path
        # Default to jpg if not found (will fail in validation)
        return self.images_dir / f"{image_id}.jpg"

    def _get_mask_path(self, image_id: str) -> Optional[Path]:
        """Get the full path to a mask file, if available."""
        if not self.mask_search_dirs:
            return None

        for base_dir in self.mask_search_dirs:
            for suffix in self.mask_filename_suffixes:
                for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                    candidate = base_dir / f"{image_id}{suffix}{ext}"
                    if candidate.exists():
                        return candidate
        return None

    def _validate_masks(self) -> None:
        """Validate that all referenced masks exist on disk when required."""
        missing = []
        for img_id in self.df["image_id"].values:
            if self._get_mask_path(img_id) is None:
                missing.append(img_id)

        if missing and len(missing) <= 10:
            raise FileNotFoundError(
                "Missing segmentation masks: "
                f"{missing}. "
                "Set data.segmentation.required=false to allow fallback to uncropped images."
            )
        elif missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} segmentation masks. First 10: {missing[:10]}. "
                "Set data.segmentation.required=false to allow fallback to uncropped images."
            )

    def _crop_with_segmentation_mask(self, image: Image.Image, mask_path: Path) -> Image.Image:
        """Crop image to lesion bounding box derived from segmentation mask."""
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask, dtype=np.uint8)
        foreground = mask_np > self.segmentation_mask_threshold
        if not np.any(foreground):
            return image

        ys, xs = np.where(foreground)
        y_min_mask, y_max_mask = int(ys.min()), int(ys.max())
        x_min_mask, x_max_mask = int(xs.min()), int(xs.max())

        image_w, image_h = image.size
        mask_h, mask_w = mask_np.shape

        if mask_w <= 0 or mask_h <= 0:
            return image

        scale_x = image_w / mask_w
        scale_y = image_h / mask_h

        x_min = int(np.floor(x_min_mask * scale_x))
        y_min = int(np.floor(y_min_mask * scale_y))
        x_max = int(np.ceil((x_max_mask + 1) * scale_x)) - 1
        y_max = int(np.ceil((y_max_mask + 1) * scale_y)) - 1

        box_w = max(x_max - x_min + 1, 1)
        box_h = max(y_max - y_min + 1, 1)
        margin_px = int(round(max(box_w, box_h) * self.segmentation_crop_margin))

        left = max(x_min - margin_px, 0)
        top = max(y_min - margin_px, 0)
        right = min(x_max + margin_px + 1, image_w)
        bottom = min(y_max + margin_px + 1, image_h)

        if left >= right or top >= bottom:
            return image
        return image.crop((left, top, right, bottom))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, dict] | Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            If use_metadata=False: Tuple of (image_tensor, label_index)
            If use_metadata=True: Tuple of (image_tensor, label_index, metadata_dict)
        """
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = row["label"]

        # Load image
        img_path = self._get_image_path(image_id)
        image = Image.open(img_path).convert("RGB")

        # Optionally crop around lesion ROI derived from segmentation mask.
        if self.use_segmentation_roi_crop:
            mask_path = self._get_mask_path(image_id)
            if mask_path is None:
                if self.segmentation_required:
                    raise FileNotFoundError(
                        f"Missing segmentation mask for image_id={image_id!r}. "
                        "Set data.segmentation.required=false to allow fallback to uncropped images."
                    )
            else:
                image = self._crop_with_segmentation_mask(image, mask_path)

        # Convert label to index
        label_idx = LABEL_TO_IDX.get(label)
        if label_idx is None:
            if self.strict_labels:
                raise KeyError(
                    f"Unknown label {label!r} for image_id={image_id!r}. "
                    f"Expected one of {sorted(LABEL_TO_IDX.keys())}."
                )
            # Sentinel used for unlabeled rows (e.g., external test sets).
            label_idx = -1

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label_idx = self.target_transform(label_idx)

        # Return with or without metadata
        if self.use_metadata:
            metadata = {}
            for col in self.metadata_columns:
                value = row[col]
                # Handle NaN/None values
                if pd.isna(value):
                    metadata[col] = None
                else:
                    metadata[col] = value
            if self.metadata_encoder is not None:
                metadata_tensor = self.metadata_encoder.encode_metadata_dict(metadata)
                return image, label_idx, metadata_tensor
            return image, label_idx, metadata
        else:
            return image, label_idx

    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in the dataset."""
        return self.df["label"].value_counts().to_dict()

    def get_class_weights(self, power: float = 1.0) -> torch.Tensor:
        """
        Compute class weights inversely proportional to class frequencies.
        Useful for weighted loss functions to handle class imbalance.
        """
        class_counts = self.df["label"].value_counts()
        total = len(self.df)
        weights = []
        for label in sorted(LABEL_TO_IDX.keys()):
            count = class_counts.get(label, 1)
            base_weight = total / (len(LABEL_TO_IDX) * count)
            weights.append(base_weight ** max(power, 0.0))
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self, power: float = 1.0) -> torch.Tensor:
        """
        Get per-sample weights for WeightedRandomSampler.
        Each sample gets the weight of its class.
        """
        class_weights = self.get_class_weights(power=power)
        sample_weights = []
        for label in self.df["label"]:
            idx = LABEL_TO_IDX[label]
            sample_weights.append(class_weights[idx].item())
        return torch.tensor(sample_weights, dtype=torch.float64)


def get_transforms(
    split: Literal["train", "val", "test"],
    image_size: int = 224,
    augmentation_strength: Literal[
        "light", "medium", "heavy", "domain", "randaugment"
    ] = "medium",
) -> transforms.Compose:
    """
    Get image transforms for different dataset splits.

    Args:
        split: Dataset split (train, val, or test)
        image_size: Target image size
        augmentation_strength: Strength of augmentation for training

    Returns:
        Composed transforms
    """
    # Common preprocessing for all splits
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        # Training transforms with augmentation
        post_transforms = []
        if augmentation_strength == "light":
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ]
        elif augmentation_strength == "medium":
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
            ]
        elif augmentation_strength == "heavy":
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ]
            post_transforms = [
                transforms.RandomErasing(
                    p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"
                )
            ]
        elif augmentation_strength == "domain":
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=12
                ),
                transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
                transforms.RandomGrayscale(p=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ]
            post_transforms = [
                transforms.RandomErasing(
                    p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"
                )
            ]
        elif augmentation_strength == "randaugment":
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
            ]
            post_transforms = [
                transforms.RandomErasing(
                    p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"
                )
            ]
        else:
            raise ValueError(
                "augmentation_strength must be one of: light, medium, heavy, domain, randaugment"
            )

        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                *aug_transforms,
                transforms.ToTensor(),
                *post_transforms,
                normalize,
            ]
        )

    else:  # val or test
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )


def load_and_split_data(
    labels_csv: Path | str,
    images_dir: Path | str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    lesion_aware: bool = True,
    use_stratified_group_kfold: bool = False,
    kfold_n_splits: int = 5,
    kfold_fold_index: int = 0,
    kfold_group_column: str = "lesion_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the HAM10000 metadata and split into train/val/test sets.

    Uses lesion-aware splitting to prevent data leakage (images of the same
    lesion stay in the same split). Falls back to stratified splitting if
    lesion_id is not available.

    Optional StratifiedGroupKFold mode can be enabled for train/validation
    splitting. In this mode, a holdout test split is created first (if
    test_size > 0), then StratifiedGroupKFold is applied to the remaining data.
    val_size is ignored in this mode.

    Args:
        labels_csv: Path to CSV with image_id, label, and optionally lesion_id
        images_dir: Path to directory containing images
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        lesion_aware: Whether to use lesion-aware splitting
        use_stratified_group_kfold: Whether to use StratifiedGroupKFold for
            train/validation splits
        kfold_n_splits: Number of folds for StratifiedGroupKFold
        kfold_fold_index: Which fold to use as validation (0-indexed)
        kfold_group_column: Column name used as group identifier

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Load labels
    df = pd.read_csv(labels_csv)

    # Validate required columns
    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'image_id' and 'label' columns")

    # Validate labels
    invalid_labels = set(df["label"].unique()) - set(CLASS_LABELS.keys())
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}")

    def _sanitize_group_ids(group_series: pd.Series, prefix: str) -> pd.Series:
        """Return group IDs with NaN replaced by unique per-row surrogate IDs."""
        groups = group_series.copy()
        missing_mask = groups.isna()
        if missing_mask.any():
            missing_idx = groups.index[missing_mask]
            groups.loc[missing_mask] = [f"{prefix}_missing_{idx}" for idx in missing_idx]
        return groups.astype(str)

    if use_stratified_group_kfold:
        if kfold_n_splits < 2:
            raise ValueError("kfold_n_splits must be >= 2 for StratifiedGroupKFold")
        if kfold_fold_index < 0 or kfold_fold_index >= kfold_n_splits:
            raise ValueError(
                f"kfold_fold_index must be in [0, {kfold_n_splits - 1}], got {kfold_fold_index}"
            )
        if kfold_group_column not in df.columns:
            raise ValueError(
                f"Column '{kfold_group_column}' is required for StratifiedGroupKFold"
            )

        # Optional holdout test split first (group-aware)
        split_groups = _sanitize_group_ids(df[kfold_group_column], kfold_group_column)
        if test_size > 0:
            gss_test = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train_val_idx, test_idx = next(
                gss_test.split(df, df["label"], groups=split_groups)
            )
            train_val_df = df.iloc[train_val_idx]
            test_df = df.iloc[test_idx]
        else:
            train_val_df = df
            test_df = df.iloc[0:0].copy()

        sgkf = StratifiedGroupKFold(
            n_splits=kfold_n_splits,
            shuffle=True,
            random_state=random_state,
        )
        split_indices = list(
            sgkf.split(
                train_val_df,
                train_val_df["label"],
                groups=split_groups.iloc[train_val_df.index],
            )
        )
        train_idx, val_idx = split_indices[kfold_fold_index]
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    # Check for lesion_id column for lesion-aware splitting
    has_lesion_id = "lesion_id" in df.columns and lesion_aware

    if has_lesion_id:
        # Lesion-aware splitting using GroupShuffleSplit
        lesion_groups = _sanitize_group_ids(df["lesion_id"], "lesion_id")
        # First split: separate test set
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_val_idx, test_idx = next(
            gss_test.split(df, df["label"], groups=lesion_groups)
        )

        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

        # Second split: separate validation from training
        val_fraction = val_size / (1 - test_size)  # Adjust for remaining data
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=random_state
        )
        train_idx, val_idx = next(
            gss_val.split(
                train_val_df,
                train_val_df["label"],
                groups=lesion_groups.iloc[train_val_df.index],
            )
        )

        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

    else:
        # Stratified splitting without lesion awareness
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=random_state,
        )

        val_fraction = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_fraction,
            stratify=train_val_df["label"],
            random_state=random_state,
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    images_dir: Path | str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    augmentation_strength: Literal[
        "light", "medium", "heavy", "domain", "randaugment"
    ] = "medium",
    use_weighted_sampling: bool = True,
    weighted_sampling_power: float = 1.0,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 2,
    persistent_workers: bool = False,
    use_metadata: bool = False,
    metadata_columns: Optional[list[str]] = None,
    metadata_encoder: Optional[MetadataEncoder] = None,
    masks_dir: Path | str | None = None,
    use_segmentation_roi_crop: bool = False,
    segmentation_mask_threshold: int = 10,
    segmentation_crop_margin: float = 0.1,
    segmentation_required: bool = False,
    mask_filename_suffixes: Optional[list[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Test data DataFrame
        images_dir: Path to image directory
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        image_size: Target image size
        augmentation_strength: Strength of training augmentation
        use_weighted_sampling: Whether to use weighted sampling for class balance
        weighted_sampling_power: Power for inverse-frequency sampler weights (0=no weighting, 1=full)
        pin_memory: Whether to pin memory (faster GPU transfer, only useful for CUDA)
        prefetch_factor: Number of batches to prefetch per worker (None to disable)
        persistent_workers: Keep workers alive between epochs (faster but more memory)
        use_metadata: Whether to return encoded metadata with each sample
        metadata_columns: Metadata columns to read from the CSV DataFrame
        metadata_encoder: Encoder used to convert metadata dicts to tensors
        masks_dir: Path to segmentation masks directory (optional)
        use_segmentation_roi_crop: Whether to crop each image to lesion ROI using mask
        segmentation_mask_threshold: Pixel threshold to binarize masks
        segmentation_crop_margin: Margin around lesion box as fraction of lesion size
        segmentation_required: Whether every sample must have a mask
        mask_filename_suffixes: Candidate suffixes to resolve mask filenames

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = HAM10000Dataset(
        df=train_df,
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=get_transforms("train", image_size, augmentation_strength),
        use_metadata=use_metadata,
        metadata_columns=metadata_columns,
        metadata_encoder=metadata_encoder,
        use_segmentation_roi_crop=use_segmentation_roi_crop,
        segmentation_mask_threshold=segmentation_mask_threshold,
        segmentation_crop_margin=segmentation_crop_margin,
        segmentation_required=segmentation_required,
        mask_filename_suffixes=mask_filename_suffixes,
    )

    val_dataset = HAM10000Dataset(
        df=val_df,
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=get_transforms("val", image_size),
        use_metadata=use_metadata,
        metadata_columns=metadata_columns,
        metadata_encoder=metadata_encoder,
        use_segmentation_roi_crop=use_segmentation_roi_crop,
        segmentation_mask_threshold=segmentation_mask_threshold,
        segmentation_crop_margin=segmentation_crop_margin,
        segmentation_required=segmentation_required,
        mask_filename_suffixes=mask_filename_suffixes,
    )

    test_dataset = HAM10000Dataset(
        df=test_df,
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=get_transforms("test", image_size),
        use_metadata=use_metadata,
        metadata_columns=metadata_columns,
        metadata_encoder=metadata_encoder,
        use_segmentation_roi_crop=use_segmentation_roi_crop,
        segmentation_mask_threshold=segmentation_mask_threshold,
        segmentation_crop_margin=segmentation_crop_margin,
        segmentation_required=segmentation_required,
        mask_filename_suffixes=mask_filename_suffixes,
    )

    # Create samplers
    train_sampler = None
    train_shuffle = True

    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights(power=weighted_sampling_power)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False  # Sampler handles shuffling

    # Create DataLoaders with optimized settings
    # Common kwargs for all loaders
    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    # Add prefetch_factor and persistent_workers only if num_workers > 0
    if num_workers > 0:
        if prefetch_factor is not None:
            common_kwargs["prefetch_factor"] = prefetch_factor
        if persistent_workers:
            common_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,  # Drop incomplete batches for training stability
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )

    return train_loader, val_loader, test_loader


def get_class_weights_for_loss(
    train_df: pd.DataFrame,
    power: float = 1.0,
) -> torch.Tensor:
    """
    Compute class weights for use in weighted loss functions.

    Args:
        train_df: Training DataFrame with 'label' column

    Returns:
        Tensor of class weights
    """
    class_counts = train_df["label"].value_counts()
    total = len(train_df)
    weights = []

    for label in sorted(LABEL_TO_IDX.keys()):
        count = class_counts.get(label, 1)
        # Inverse frequency weighting with optional power scaling
        base_weight = total / (len(LABEL_TO_IDX) * count)
        weights.append(base_weight ** max(power, 0.0))

    # Normalize weights
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)

    return weights


# Utility functions for inference
def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Get transforms for inference (same as validation/test)."""
    return get_transforms("test", image_size)


def preprocess_image(
    image: Image.Image | np.ndarray | str | Path,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Preprocess a single image for inference.

    Args:
        image: PIL Image, numpy array, or path to image file
        image_size: Target image size

    Returns:
        Preprocessed image tensor with batch dimension (1, C, H, W)
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")

    # Apply transforms
    transform = get_inference_transform(image_size)
    tensor = transform(image)

    # Add batch dimension
    return tensor.unsqueeze(0)
