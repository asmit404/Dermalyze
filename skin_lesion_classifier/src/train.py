"""
Training Module for Skin Lesion Classification.

This script provides a complete training pipeline for the EfficientNet-V2
based skin lesion classifier, including:
- Configuration management
- Data loading and augmentation
- Model training with mixed precision
- Learning rate scheduling
- Checkpoint saving and resuming
- Experiment logging
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import (
    load_and_split_data,
    create_dataloaders,
    get_class_weights_for_loss,
    CLASS_LABELS,
    IDX_TO_LABEL,
)
from src.models.efficientnet import create_model, get_loss_function


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class Mixup:
    """Mixup data augmentation.
    
    Reference: https://arxiv.org/abs/1710.09412
    """
    
    def __init__(self, alpha: float = 1.0, num_classes: int = 7):
        """
        Args:
            alpha: Beta distribution parameter for mixup
            num_classes: Number of classes for one-hot encoding
        """
        self.alpha = alpha
        self.num_classes = num_classes
    
    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch.
        
        Returns:
            mixed_images: Mixed images
            targets_a: First set of targets
            targets_b: Second set of targets  
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        targets_a = targets
        targets_b = targets[index]
        
        return mixed_images, targets_a, targets_b, lam


class CutMix:
    """CutMix data augmentation.
    
    Reference: https://arxiv.org/abs/1905.04899
    """
    
    def __init__(self, alpha: float = 1.0, num_classes: int = 7):
        """
        Args:
            alpha: Beta distribution parameter for cutmix
            num_classes: Number of classes for one-hot encoding
        """
        self.alpha = alpha
        self.num_classes = num_classes
    
    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply cutmix to a batch.
        
        Returns:
            mixed_images: Mixed images with cut patches
            targets_a: First set of targets
            targets_b: Second set of targets
            lam: Mixing coefficient based on box area
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        # Get image dimensions
        _, _, h, w = images.size()
        
        # Calculate box coordinates
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random center point
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        
        targets_a = targets
        targets_b = targets[index]
        
        return mixed_images, targets_a, targets_b, lam


class ModelEMA:
    """Exponential Moving Average of model parameters.
    
    Maintains a moving average of model weights for better generalization.
    Reference: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Model to track
            decay: Decay rate for EMA (typically 0.999 - 0.9999)
            device: Device to store EMA model on
        """
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        
        if self.device is not None:
            self.module.to(device=device)
        
        # Freeze EMA parameters
        for param in self.module.parameters():
            param.requires_grad_(False)
    
    def _update(self, model: nn.Module, update_fn):
        """Update EMA parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.module.state_dict().values(),
                model.state_dict().values(),
            ):
                if ema_param.dtype.is_floating_point:
                    update_fn(ema_param, model_param)
    
    def update(self, model: nn.Module):
        """Update EMA parameters with current model."""
        self._update(
            model,
            update_fn=lambda e, m: e.copy_(
                self.decay * e + (1.0 - self.decay) * m
            ),
        )
    
    def set(self, model: nn.Module):
        """Set EMA parameters to current model parameters."""
        self._update(
            model,
            update_fn=lambda e, m: e.copy_(m),
        )
    
    def state_dict(self):
        """Return EMA model state dict."""
        return self.module.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load EMA model state dict."""
        self.module.load_state_dict(state_dict)


def mixup_criterion(
    criterion: nn.Module,
    outputs: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Calculate loss for mixup/cutmix."""
    return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)


def set_seed(seed: int, fast_mode: bool = False) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        fast_mode: If True, disables deterministic mode for faster training
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if not fast_mode:
        # For deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster training mode
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def load_config(config_path: Path | str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: "min" for loss, "max" for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class MetricTracker:
    """Track and compute training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.loss_sum = 0.0
        self.correct = 0
        self.total = 0
        self.all_preds = []
        self.all_targets = []
    
    def update(
        self,
        loss: float,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ):
        """Update metrics with batch results."""
        batch_size = targets.size(0)
        self.loss_sum += loss * batch_size
        self.correct += (preds == targets).sum().item()
        self.total += batch_size
        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        return {
            "loss": self.loss_sum / max(self.total, 1),
            "accuracy": self.correct / max(self.total, 1),
        }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    use_amp: bool = True,
    mixup: Optional[Mixup] = None,
    cutmix: Optional[CutMix] = None,
    mixup_prob: float = 0.5,
    model_ema: Optional[ModelEMA] = None,
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision training
        scheduler: Learning rate scheduler (if using OneCycleLR)
        epoch: Current epoch number
        use_amp: Whether to use automatic mixed precision
        mixup: Mixup augmentation instance
        cutmix: CutMix augmentation instance
        mixup_prob: Probability of applying mixup vs cutmix
        model_ema: ModelEMA instance for exponential moving average
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    metrics = MetricTracker()
    
    # Use non_blocking only for CUDA
    non_blocking = device.type == "cuda"
    
    # Determine if we should use mixup/cutmix
    use_augmentation = mixup is not None or cutmix is not None
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        
        # Apply Mixup or CutMix
        if use_augmentation:
            if mixup is not None and cutmix is not None:
                # Randomly choose between mixup and cutmix
                if np.random.rand() < mixup_prob:
                    images, targets_a, targets_b, lam = mixup(images, targets)
                else:
                    images, targets_a, targets_b, lam = cutmix(images, targets)
            elif mixup is not None:
                images, targets_a, targets_b, lam = mixup(images, targets)
            else:
                images, targets_a, targets_b, lam = cutmix(images, targets)
            use_mixed_loss = True
        else:
            use_mixed_loss = False
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision (only for CUDA)
        if use_amp and device.type == "cuda":
            with autocast(device_type="cuda"):
                outputs = model(images)
                if use_mixed_loss:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training (MPS and CPU)
            outputs = model(images)
            if use_mixed_loss:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Update ModelEMA
        if model_ema is not None:
            model_ema.update(model)
        
        # Update learning rate if using OneCycleLR
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        # Update metrics (use original targets for metric computation)
        preds = torch.argmax(outputs, dim=1)
        if use_mixed_loss:
            # For mixed samples, use the primary target for accuracy
            metrics.update(loss.item(), preds, targets_a)
        else:
            metrics.update(loss.item(), preds, targets)
        
        # Update progress bar
        current_metrics = metrics.compute()
        pbar.set_postfix({
            "loss": f"{current_metrics['loss']:.4f}",
            "acc": f"{current_metrics['accuracy']:.4f}",
        })
    
    return metrics.compute()


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    metrics = MetricTracker()
    
    # Use non_blocking only for CUDA
    non_blocking = device.type == "cuda"
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for images, targets in pbar:
        images = images.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        preds = torch.argmax(outputs, dim=1)
        metrics.update(loss.item(), preds, targets)
        
        current_metrics = metrics.compute()
        pbar.set_postfix({
            "loss": f"{current_metrics['loss']:.4f}",
            "acc": f"{current_metrics['accuracy']:.4f}",
        })
    
    return metrics.compute()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    output_dir: Path,
    is_best: bool = False,
    model_ema: Optional[ModelEMA] = None,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config,
    }
    
    # Add EMA state if available
    if model_ema is not None:
        checkpoint["model_ema_state_dict"] = model_ema.state_dict()
    
    # Save latest checkpoint
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model with val_loss: {metrics['val_loss']:.4f}")
    
    # Save epoch checkpoint (optional, controlled by config)
    save_epoch_checkpoints = config.get("output", {}).get("save_epoch_checkpoints", False)
    if save_epoch_checkpoints:
        epoch_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    model_ema: Optional[ModelEMA] = None,
) -> Tuple[int, Dict[str, float]]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if model_ema is not None:
        if "model_ema_state_dict" in checkpoint:
            model_ema.load_state_dict(checkpoint["model_ema_state_dict"])
        else:
            model_ema.set(model)
    
    return checkpoint["epoch"], checkpoint.get("metrics", {})


def train(
    config: Dict[str, Any],
    output_dir: Path,
    resume_from: Optional[Path] = None,
) -> None:
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save outputs
        resume_from: Path to checkpoint to resume from
    """
    # Set random seed
    train_config = config.get("training", {})
    seed = train_config.get("seed", 42)
    fast_mode = train_config.get("fast_mode", True)
    set_seed(seed, fast_mode=fast_mode)
    logger.info(f"Random seed: {seed} (fast_mode={fast_mode})")
    
    # Get device
    device = get_device()
    
    # Get data paths from config
    data_config = config.get("data", {})
    labels_csv = Path(data_config.get("labels_csv", "data/HAM10000/labels.csv"))
    images_dir = Path(data_config.get("images_dir", "data/HAM10000/images"))
    
    # Load and split data
    logger.info("Loading and splitting data...")
    split_seed = data_config.get("split_seed", seed)  # Allow separate split seed
    train_df, val_df, test_df = load_and_split_data(
        labels_csv=labels_csv,
        images_dir=images_dir,
        val_size=data_config.get("val_size", 0.15),
        test_size=data_config.get("test_size", 0.15),
        random_state=split_seed,
        lesion_aware=data_config.get("lesion_aware", True),
    )
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Save split information
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)
    
    # Extract training config values
    batch_size = train_config.get("batch_size", 32)
    num_workers = train_config.get("num_workers", 4)
    
    # Two-stage training support
    use_two_stage = "stage1_epochs" in train_config and "stage2_epochs" in train_config
    if use_two_stage:
        stage1_epochs = train_config.get("stage1_epochs", 5)
        stage2_epochs = train_config.get("stage2_epochs", 25)
        epochs = stage1_epochs + stage2_epochs
        stage1_lr = train_config.get("stage1_lr", 1e-3)
        stage2_lr = train_config.get("stage2_lr", 1e-4)
        stage1_weight_decay = train_config.get("stage1_weight_decay", 0.01)
        stage2_weight_decay = train_config.get("stage2_weight_decay", 0.02)
        lr = stage1_lr  # Start with stage 1 lr
        weight_decay = stage1_weight_decay
        logger.info(f"Two-stage training enabled:")
        logger.info(f"  • Stage 1 (Warm-up): {stage1_epochs} epochs, lr={stage1_lr}, wd={stage1_weight_decay}")
        logger.info(f"  • Stage 2 (Fine-tuning): {stage2_epochs} epochs, lr={stage2_lr}, wd={stage2_weight_decay}")
        logger.info(f"  • Total: {epochs} epochs")
    else:
        epochs = train_config.get("epochs", 30)
        lr = train_config.get("lr", 1e-4)
        weight_decay = train_config.get("weight_decay", 0.01)
    
    use_amp = train_config.get("use_amp", True) and device.type == "cuda"
    
    # Create DataLoaders
    logger.info("Creating DataLoaders...")
    prefetch_factor = train_config.get("prefetch_factor", 2)
    persistent_workers = train_config.get("persistent_workers", True) and num_workers > 0
    # pin_memory only helps with CUDA, not MPS or CPU
    pin_memory = device.type == "cuda"
    
    train_loader, val_loader, _ = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        images_dir=images_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=config.get("model", {}).get("image_size", 224),
        augmentation_strength=train_config.get("augmentation", "medium"),
        use_weighted_sampling=train_config.get("use_weighted_sampling", True),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    
    # Calculate class weights for loss function
    class_weights = get_class_weights_for_loss(train_df)
    class_weights = class_weights.to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Create model
    model_config = config.get("model", {})
    head_type = model_config.get("head_type", "simple")
    logger.info("Creating model...")
    logger.info(f"Using EfficientNet-V2 Small with {head_type.upper()} classification head")
    model = create_model(
        num_classes=model_config.get("num_classes", 7),
        pretrained=model_config.get("pretrained", True),
        dropout_rate=model_config.get("dropout_rate", 0.3),
        freeze_backbone=model_config.get("freeze_backbone", False),
        head_type=head_type,
    )
    model = model.to(device)
    base_model = model
    
    # Enable torch.compile for PyTorch 2.0+ (skip for MPS due to potential backward pass issues)
    use_compile = train_config.get("use_torch_compile", True)
    if use_compile and hasattr(torch, "compile"):
        if device.type == "mps":
            logger.warning(
                "torch.compile is enabled but running on MPS. "
                "MPS may have issues with backward passes in compiled mode. "
                "Set use_torch_compile: false in config if you encounter errors."
            )
        try:
            logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model, mode="default")
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
    
    logger.info(f"Model parameters: {model.get_total_params():,}")
    logger.info(f"Trainable parameters: {model.get_trainable_params():,}")
    
    # Create loss function
    loss_config = config.get("loss", {})
    
    # Handle alpha parameter - use manual values if provided, otherwise use computed class weights
    focal_alpha = None
    if "alpha" in loss_config and loss_config["alpha"] is not None:
        focal_alpha = torch.tensor(loss_config["alpha"], dtype=torch.float32)
        logger.info(f"Using manual alpha weights: {focal_alpha.tolist()}")
    else:
        focal_alpha = class_weights
        if focal_alpha is not None:
            logger.info(f"Using computed class weights: {focal_alpha.tolist()}")
    
    criterion = get_loss_function(
        loss_type=loss_config.get("type", "focal"),
        class_weights=focal_alpha if loss_config.get("type") != "focal" else None,
        label_smoothing=loss_config.get("label_smoothing", 0.1),
        focal_gamma=loss_config.get("gamma", loss_config.get("focal_gamma", 2.0)),
        focal_alpha=focal_alpha if loss_config.get("type") == "focal" else None,
    )
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Create learning rate scheduler
    scheduler_config = train_config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "cosine")
    
    if scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=scheduler_config.get("warmup_pct", 0.1),
            anneal_strategy="cos",
        )
    else:  # cosine
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get("T_0", 10),
            T_mult=scheduler_config.get("T_mult", 2),
            eta_min=scheduler_config.get("eta_min", 1e-6),
        )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if use_amp else None
    
    # Initialize Mixup and CutMix
    mixup_config = train_config.get("mixup", {})
    cutmix_config = train_config.get("cutmix", {})
    
    mixup = None
    cutmix = None
    mixup_prob = 0.5
    
    if mixup_config.get("enabled", False):
        mixup_alpha = mixup_config.get("alpha", 1.0)
        mixup = Mixup(alpha=mixup_alpha, num_classes=model_config.get("num_classes", 7))
        logger.info(f"Mixup enabled with alpha={mixup_alpha}")
    
    if cutmix_config.get("enabled", False):
        cutmix_alpha = cutmix_config.get("alpha", 1.0)
        cutmix = CutMix(alpha=cutmix_alpha, num_classes=model_config.get("num_classes", 7))
        logger.info(f"CutMix enabled with alpha={cutmix_alpha}")
    
    if mixup is not None and cutmix is not None:
        mixup_prob = train_config.get("mixup_prob", 0.5)
        logger.info(f"Using both Mixup and CutMix with mixup_prob={mixup_prob}")
    
    # Initialize ModelEMA
    model_ema = None
    ema_config = train_config.get("ema", {})
    use_ema_for_eval = ema_config.get("use_for_eval", ema_config.get("validate_ema", True))
    save_ema_best = ema_config.get("save_best", True)
    
    if ema_config.get("enabled", False):
        ema_decay = ema_config.get("decay", 0.9999)
        ema_source = base_model if model is not base_model else model
        model_ema = ModelEMA(ema_source, decay=ema_decay, device=device)
        model_ema.set(model)
        logger.info(f"ModelEMA enabled with decay={ema_decay}, use_for_eval={use_ema_for_eval}, save_best={save_ema_best}")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=train_config.get("early_stopping_patience", 15),
        min_delta=0.001,
        mode="min",
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    has_saved_best = False
    
    if resume_from is not None and resume_from.exists():
        logger.info(f"Resuming from checkpoint: {resume_from}")
        start_epoch, prev_metrics = load_checkpoint(
            resume_from, model, optimizer, scheduler, model_ema
        )
        start_epoch += 1
        
        # Check if best checkpoint exists and load its metrics
        best_checkpoint_path = output_dir / "checkpoint_best.pt"
        if best_checkpoint_path.exists():
            logger.info("Found existing best checkpoint - loading its metrics")
            best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
            best_metrics = best_checkpoint.get("metrics", {})
            best_val_loss = best_metrics.get("val_loss", float("inf"))
            has_saved_best = True
            logger.info(f"Best validation loss from existing checkpoint: {best_val_loss:.4f}")
        else:
            # Use metrics from resumed checkpoint as baseline
            best_val_loss = prev_metrics.get("val_loss", float("inf"))
            has_saved_best = False
            logger.warning("No existing best checkpoint found - will create new one")
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # Two-stage training: transition to stage 2
        if use_two_stage and epoch == stage1_epochs and epoch > start_epoch:
            logger.info("\n" + "="*70)
            logger.info(f"STAGE TRANSITION: Warm-up Complete → Starting Fine-tuning")
            logger.info(f"Completed {stage1_epochs} warm-up epochs, starting {stage2_epochs} fine-tuning epochs")
            logger.info(f"Updating LR: {stage1_lr} -> {stage2_lr}")
            logger.info(f"Updating Weight Decay: {stage1_weight_decay} -> {stage2_weight_decay}")
            logger.info("="*70 + "\n")
            
            # Recreate optimizer with stage 2 parameters
            optimizer = AdamW(
                model.parameters(),
                lr=stage2_lr,
                weight_decay=stage2_weight_decay,
                betas=(0.9, 0.999),
            )
            
            # Recreate scheduler for stage 2
            if scheduler_type == "onecycle":
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=stage2_lr,
                    epochs=stage2_epochs,
                    steps_per_epoch=len(train_loader),
                    pct_start=scheduler_config.get("warmup_pct", 0.1),
                    anneal_strategy="cos",
                )
            else:  # cosine
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=scheduler_config.get("T_0", 10),
                    T_mult=scheduler_config.get("T_mult", 2),
                    eta_min=scheduler_config.get("eta_min", 1e-6),
                )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        stage_info = ""
        if use_two_stage:
            current_stage = 1 if epoch < stage1_epochs else 2
            stage_epoch = epoch + 1 if current_stage == 1 else epoch - stage1_epochs + 1
            stage_total = stage1_epochs if current_stage == 1 else stage2_epochs
            stage_name = "Warm-up" if current_stage == 1 else "Fine-tuning"
            stage_info = f" | {stage_name} Stage {current_stage} [{stage_epoch}/{stage_total}]"
        logger.info(f"\nEpoch {epoch + 1}/{epochs}{stage_info} | LR: {current_lr:.2e}")
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            scheduler=scheduler if scheduler_type == "onecycle" else None,
            epoch=epoch + 1,
            use_amp=use_amp,
            mixup=mixup,
            cutmix=cutmix,
            mixup_prob=mixup_prob,
            model_ema=model_ema,
        )
        
        # Validate with EMA model if available and configured, otherwise use regular model
        if model_ema is not None and use_ema_for_eval:
            val_metrics = validate(
                model=model_ema.module,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch + 1,
            )
            logger.info("Validation performed with EMA model")
        else:
            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch + 1,
            )
        
        # Update scheduler (if not OneCycleLR)
        if scheduler_type != "onecycle":
            scheduler.step()
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)
        
        # Check for best model and save checkpoint
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            has_saved_best = True
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics={
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
            },
            config=config,
            output_dir=output_dir,
            is_best=is_best,
            model_ema=model_ema,
        )
        
        # Early stopping check
        if early_stopping(val_metrics["loss"]):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Ensure we have a best checkpoint (use latest if no improvement occurred)
    if not has_saved_best:
        logger.warning("No validation improvement during training. Saving latest as best.")
        import shutil
        shutil.copy(
            output_dir / "checkpoint_latest.pt",
            output_dir / "checkpoint_best.pt"
        )
    
    # Save training history
    import json
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time / 60:.1f} minutes")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train skin lesion classifier"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"run_{timestamp}"
    else:
        output_dir = args.output
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Set up file logging
    file_handler = logging.FileHandler(output_dir / "training.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {config}")
    
    # Start training
    train(config, output_dir, args.resume)


if __name__ == "__main__":
    main()
