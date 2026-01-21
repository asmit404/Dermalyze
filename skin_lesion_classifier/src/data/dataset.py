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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

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
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path | str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            df: DataFrame with columns 'image_id' and 'label' (and optionally 'lesion_id')
            images_dir: Directory containing the image files
            transform: Transformations to apply to images
            target_transform: Transformations to apply to labels
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Validate that all images exist
        self._validate_images()
    
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
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label_index)
        """
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = row["label"]
        
        # Load image
        img_path = self._get_image_path(image_id)
        image = Image.open(img_path).convert("RGB")
        
        # Convert label to index
        label_idx = LABEL_TO_IDX[label]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label_idx = self.target_transform(label_idx)
        
        return image, label_idx
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in the dataset."""
        return self.df["label"].value_counts().to_dict()
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights inversely proportional to class frequencies.
        Useful for weighted loss functions to handle class imbalance.
        """
        class_counts = self.df["label"].value_counts()
        total = len(self.df)
        weights = []
        for label in sorted(LABEL_TO_IDX.keys()):
            count = class_counts.get(label, 1)
            weights.append(total / (len(LABEL_TO_IDX) * count))
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Get per-sample weights for WeightedRandomSampler.
        Each sample gets the weight of its class.
        """
        class_weights = self.get_class_weights()
        sample_weights = []
        for label in self.df["label"]:
            idx = LABEL_TO_IDX[label]
            sample_weights.append(class_weights[idx].item())
        return torch.tensor(sample_weights, dtype=torch.float64)


def get_transforms(
    split: Literal["train", "val", "test"],
    image_size: int = 224,
    augmentation_strength: Literal["light", "medium", "heavy"] = "medium",
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
        else:  # heavy
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
        
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            *aug_transforms,
            transforms.ToTensor(),
            normalize,
        ])
    
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def load_and_split_data(
    labels_csv: Path | str,
    images_dir: Path | str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    lesion_aware: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the HAM10000 metadata and split into train/val/test sets.
    
    Uses lesion-aware splitting to prevent data leakage (images of the same
    lesion stay in the same split). Falls back to stratified splitting if
    lesion_id is not available.
    
    Args:
        labels_csv: Path to CSV with image_id, label, and optionally lesion_id
        images_dir: Path to directory containing images
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        lesion_aware: Whether to use lesion-aware splitting
        
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
    
    # Check for lesion_id column for lesion-aware splitting
    has_lesion_id = "lesion_id" in df.columns and lesion_aware
    
    if has_lesion_id:
        # Lesion-aware splitting using GroupShuffleSplit
        # First split: separate test set
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_val_idx, test_idx = next(
            gss_test.split(df, df["label"], groups=df["lesion_id"])
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
                groups=train_val_df["lesion_id"],
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
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    images_dir: Path | str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    augmentation_strength: Literal["light", "medium", "heavy"] = "medium",
    use_weighted_sampling: bool = True,
    pin_memory: bool = True,
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
        pin_memory: Whether to pin memory (faster GPU transfer)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = HAM10000Dataset(
        df=train_df,
        images_dir=images_dir,
        transform=get_transforms("train", image_size, augmentation_strength),
    )
    
    val_dataset = HAM10000Dataset(
        df=val_df,
        images_dir=images_dir,
        transform=get_transforms("val", image_size),
    )
    
    test_dataset = HAM10000Dataset(
        df=test_df,
        images_dir=images_dir,
        transform=get_transforms("test", image_size),
    )
    
    # Create samplers
    train_sampler = None
    train_shuffle = True
    
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False  # Sampler handles shuffling
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for training stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader


def get_class_weights_for_loss(train_df: pd.DataFrame) -> torch.Tensor:
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
        # Inverse frequency weighting
        weights.append(total / (len(LABEL_TO_IDX) * count))
    
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
