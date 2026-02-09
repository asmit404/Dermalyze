"""
Feature Caching Module for Skin Lesion Classification.

Pre-extracts backbone features and caches them to disk, enabling
significantly faster training when the backbone is frozen. Instead of
running every image through the full EfficientNet backbone each epoch,
cached features are loaded directly and only the classification head
is trained.

Usage:
    - Set `training.feature_cache.enabled: true` in config.yaml
    - The backbone must be frozen (`model.freeze_backbone: true`)
    - Cache is automatically invalidated when model/data config changes
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from .dataset import LABEL_TO_IDX

logger = logging.getLogger(__name__)


class CachedFeatureDataset(Dataset):
    """
    Dataset that serves pre-extracted backbone features.

    Instead of loading and transforming images, this dataset returns
    pre-computed feature vectors directly, bypassing the backbone entirely.

    Attributes:
        features: Tensor of shape (N, feature_dim) with cached backbone outputs
        labels: Tensor of shape (N,) with integer class labels
        label_names: List of original string labels for weight computation
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        label_names: Optional[List[str]] = None,
    ):
        """
        Args:
            features: Pre-extracted feature vectors (N, feature_dim)
            labels: Integer class labels (N,)
            label_names: Original string labels (for computing sample weights)
        """
        assert len(features) == len(labels), (
            f"Feature/label count mismatch: {len(features)} vs {len(labels)}"
        )
        self.features = features
        self.labels = labels
        self.label_names = label_names

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.features[idx], self.labels[idx].item()

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights."""
        num_classes = len(LABEL_TO_IDX)
        counts = torch.bincount(self.labels, minlength=num_classes).float()
        counts = counts.clamp(min=1)
        total = self.labels.size(0)
        weights = total / (num_classes * counts)
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return class_weights[self.labels].double()


def compute_cache_key(
    model_config: Dict[str, Any],
    image_size: int,
    split_name: str,
    data_hash: str,
) -> str:
    """
    Compute a deterministic hash key for cache validation.

    The cache is invalidated whenever the backbone architecture, pretrained
    weights, image size, or underlying data changes.

    Args:
        model_config: Model configuration dict
        image_size: Input image resolution
        split_name: Dataset split (train/val/test)
        data_hash: Hash of the image IDs in this split

    Returns:
        Hex digest string identifying this cache configuration
    """
    key_parts = {
        "model_name": model_config.get("name", "efficientnet_v2"),
        "pretrained": model_config.get("pretrained", True),
        "image_size": image_size,
        "split": split_name,
        "data_hash": data_hash,
    }
    key_str = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute a hash of image IDs for cache validation."""
    ids = sorted(df["image_id"].tolist())
    return hashlib.sha256(",".join(ids).encode()).hexdigest()[:16]


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    desc: str = "Extracting features",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract backbone features for all samples in a DataLoader.

    Args:
        backbone: The feature extractor (backbone with classifier removed)
        dataloader: DataLoader yielding (images, labels) batches
        device: Device to run extraction on
        desc: Progress bar description

    Returns:
        Tuple of (features, labels) tensors on CPU
    """
    backbone.eval()
    all_features = []
    all_labels = []
    non_blocking = device.type == "cuda"

    for images, labels in tqdm(dataloader, desc=desc, leave=False):
        images = images.to(device, non_blocking=non_blocking)
        features = backbone(images)
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


class FeatureCacheManager:
    """
    Manages backbone feature extraction and on-disk caching.

    Features are stored as .pt files alongside metadata JSON for
    cache validation. When the configuration changes, the cache is
    automatically rebuilt.

    Directory structure:
        cache_dir/
            {split}_{cache_key}.pt      - serialized dict with features/labels
            {split}_{cache_key}.json    - metadata for validation
    """

    def __init__(
        self,
        cache_dir: Path | str,
        model_config: Dict[str, Any],
        image_size: int,
    ):
        """
        Args:
            cache_dir: Directory to store cached features
            model_config: Model configuration for cache key computation
            image_size: Image resolution used for feature extraction
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_config = model_config
        self.image_size = image_size

    def _cache_paths(
        self, split: str, data_hash: str
    ) -> Tuple[Path, Path]:
        """Get paths for cache data and metadata files."""
        key = compute_cache_key(
            self.model_config, self.image_size, split, data_hash
        )
        data_path = self.cache_dir / f"{split}_{key}.pt"
        meta_path = self.cache_dir / f"{split}_{key}.json"
        return data_path, meta_path

    def is_cached(self, split: str, df: pd.DataFrame) -> bool:
        """Check if valid cache exists for a given split."""
        data_hash = hash_dataframe(df)
        data_path, meta_path = self._cache_paths(split, data_hash)
        return data_path.exists() and meta_path.exists()

    def save_cache(
        self,
        split: str,
        df: pd.DataFrame,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Path:
        """
        Save extracted features to disk.

        Args:
            split: Dataset split name
            df: DataFrame for this split (used for cache key and metadata)
            features: Feature tensor (N, feature_dim)
            labels: Label tensor (N,)

        Returns:
            Path to saved cache file
        """
        data_hash = hash_dataframe(df)
        data_path, meta_path = self._cache_paths(split, data_hash)

        # Save features and labels
        cache_data = {
            "features": features,
            "labels": labels,
            "image_ids": df["image_id"].tolist(),
            "label_names": df["label"].tolist(),
        }
        torch.save(cache_data, data_path)

        # Save metadata
        metadata = {
            "split": split,
            "num_samples": len(df),
            "feature_dim": features.shape[1],
            "data_hash": data_hash,
            "cache_key": compute_cache_key(
                self.model_config, self.image_size, split, data_hash
            ),
            "model_name": self.model_config.get("name", "efficientnet_v2"),
            "image_size": self.image_size,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        size_mb = data_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Cached {split} features: {len(df)} samples, "
            f"dim={features.shape[1]}, size={size_mb:.1f}MB → {data_path}"
        )
        return data_path

    def load_cache(
        self, split: str, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Load cached features from disk.

        Args:
            split: Dataset split name
            df: DataFrame for this split (for cache key lookup)

        Returns:
            Tuple of (features, labels, label_names)

        Raises:
            FileNotFoundError: If cache does not exist
        """
        data_hash = hash_dataframe(df)
        data_path, _ = self._cache_paths(split, data_hash)

        if not data_path.exists():
            raise FileNotFoundError(
                f"No cache found for {split} split at {data_path}"
            )

        cache_data = torch.load(data_path, map_location="cpu", weights_only=False)
        logger.info(
            f"Loaded cached {split} features: "
            f"{cache_data['features'].shape[0]} samples, "
            f"dim={cache_data['features'].shape[1]}"
        )
        return (
            cache_data["features"],
            cache_data["labels"],
            cache_data.get("label_names", []),
        )

    def get_or_extract(
        self,
        split: str,
        df: pd.DataFrame,
        backbone: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        force_rebuild: bool = False,
    ) -> CachedFeatureDataset:
        """
        Load cached features or extract and cache them if needed.

        This is the primary entry point. It checks for a valid cache,
        extracts features if necessary, and returns a CachedFeatureDataset.

        Args:
            split: Dataset split name (train/val/test)
            df: DataFrame for this split
            backbone: Model backbone for feature extraction
            dataloader: DataLoader with image transforms applied
            device: Compute device
            force_rebuild: Force re-extraction even if cache exists

        Returns:
            CachedFeatureDataset ready for training/evaluation
        """
        if not force_rebuild and self.is_cached(split, df):
            logger.info(f"Using cached features for {split} split")
            features, labels, label_names = self.load_cache(split, df)
        else:
            if force_rebuild:
                logger.info(f"Force rebuilding {split} feature cache...")
            else:
                logger.info(f"No valid cache for {split} split, extracting features...")

            features, labels = extract_features(
                backbone, dataloader, device, desc=f"Caching {split} features"
            )
            label_names = df["label"].tolist()
            self.save_cache(split, df, features, labels)

        return CachedFeatureDataset(
            features=features,
            labels=labels,
            label_names=label_names,
        )

    def clear_cache(self, split: Optional[str] = None) -> int:
        """
        Remove cached feature files.

        Args:
            split: If specified, only remove caches for this split.
                   If None, remove all caches.

        Returns:
            Number of files removed
        """
        removed = 0
        pattern = f"{split}_*" if split else "*"
        for path in self.cache_dir.glob(pattern):
            if path.suffix in (".pt", ".json"):
                path.unlink()
                removed += 1

        logger.info(
            f"Cleared {removed} cache files"
            + (f" for {split} split" if split else "")
        )
        return removed


def create_cached_dataloaders(
    train_dataset: CachedFeatureDataset,
    val_dataset: CachedFeatureDataset,
    batch_size: int = 32,
    num_workers: int = 2,
    use_weighted_sampling: bool = True,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for cached feature datasets.

    These are lightweight loaders since the data is already in tensor form
    and much smaller than raw images.

    Args:
        train_dataset: Cached training features
        val_dataset: Cached validation features
        batch_size: Batch size (can be larger since features are small)
        num_workers: Number of workers (typically 0-2 since I/O is minimal)
        use_weighted_sampling: Whether to use class-balanced sampling
        pin_memory: Whether to pin memory for CUDA transfers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Weighted sampling for class balance
    train_sampler = None
    train_shuffle = True

    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for val (no gradients)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
