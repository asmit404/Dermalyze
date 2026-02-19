"""
Training Module for Skin Lesion Classification.

This script provides a complete training pipeline for the EfficientNet-B0
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


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic operations (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup to a batch."""
    if alpha <= 0:
        return images, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed_images, targets_a, targets_b, lam


def _rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    """Generate random bounding box for cutmix."""
    _, _, height, width = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, width)
    bby2 = min(cy + cut_h // 2, height)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply cutmix to a batch."""
    if alpha <= 0:
        return images, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)

    bbx1, bby1, bbx2, bby2 = _rand_bbox(images.size(), lam)

    # Clone images to avoid in-place modification issues
    mixed_images = images.clone()
    mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda based on mixed area
    area = (bbx2 - bbx1) * (bby2 - bby1)
    lam = 1.0 - area / (images.size(2) * images.size(3))
    targets_a = targets
    targets_b = targets[index]
    return mixed_images, targets_a, targets_b, lam


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
        # Enable TF32 for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
        # MPS-specific optimizations
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = (
            "0.0"  # Better memory management
        )
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


class ModelEMA:
    """Exponential moving average (EMA) of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.detach(), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


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
    freeze_backbone: bool = False,
    ema: Optional["ModelEMA"] = None,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    cutmix_prob: float = 0.5,
    gradient_accumulation_steps: int = 1,
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

    Returns:
        Dictionary of training metrics
    """
    model.train()
    if freeze_backbone and hasattr(model, "backbone"):
        model.backbone.eval()
    metrics = MetricTracker()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    # Ensure model parameters are contiguous for MPS efficiency
    if device.type == "mps":
        for param in model.parameters():
            if param.requires_grad and not param.is_contiguous():
                param.data = param.data.contiguous()

    for batch_idx, (images, targets) in enumerate(pbar):
        # Ensure tensors are contiguous before transfer for MPS
        if not images.is_contiguous():
            images = images.contiguous()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        use_mix = False
        if (mixup_alpha > 0 or cutmix_alpha > 0) and np.random.rand() < mixup_prob:
            if cutmix_alpha > 0 and (
                mixup_alpha <= 0 or np.random.rand() < cutmix_prob
            ):
                images, targets_a, targets_b, lam = cutmix_data(
                    images, targets, cutmix_alpha
                )
            else:
                images, targets_a, targets_b, lam = mixup_data(
                    images, targets, mixup_alpha
                )
            use_mix = True
        else:
            targets_a = targets_b = targets
            lam = 1.0

        # Gradient accumulation: only zero grads at the start of accumulation
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision
        if use_amp and device.type == "cuda":
            with autocast(device_type="cuda"):
                outputs = model(images)
                if use_mix:
                    loss = lam * criterion(outputs, targets_a) + (
                        1.0 - lam
                    ) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
        else:
            outputs = model(images)
            if use_mix:
                loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(
                    outputs, targets_b
                )
            else:
                loss = criterion(outputs, targets)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

                # MPS synchronization for accurate timing
                if device.type == "mps":
                    torch.mps.synchronize()

        # Update learning rate if using OneCycleLR (only after optimizer step)
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(
            train_loader
        ):
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()

        # Update metrics (scale loss back up for reporting)
        preds = torch.argmax(outputs, dim=1)
        metric_targets = targets_a if use_mix else targets
        metrics.update(loss.item() * gradient_accumulation_steps, preds, metric_targets)

        # Update progress bar
        current_metrics = metrics.compute()
        pbar.set_postfix(
            {
                "loss": f"{current_metrics['loss']:.4f}",
                "acc": f"{current_metrics['accuracy']:.4f}",
            }
        )

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

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

    for images, targets in pbar:
        # Ensure contiguous tensors for MPS
        if not images.is_contiguous():
            images = images.contiguous()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        preds = torch.argmax(outputs, dim=1)
        metrics.update(loss.item(), preds, targets)

        current_metrics = metrics.compute()
        pbar.set_postfix(
            {
                "loss": f"{current_metrics['loss']:.4f}",
                "acc": f"{current_metrics['accuracy']:.4f}",
            }
        )

    # MPS synchronization before returning metrics
    if device.type == "mps":
        torch.mps.synchronize()

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
    ema: Optional[ModelEMA] = None,
    save_ema_for_best: bool = False,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config,
        "ema_state_dict": ema.shadow if ema is not None else None,
        "ema_decay": ema.decay if ema is not None else None,
    }

    # Save latest checkpoint
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, checkpoint_path)

    def _export_torchscript(best_model_state_dict: Dict[str, torch.Tensor]) -> None:
        """Export TorchScript model for inference portability."""
        try:
            model_config = config.get("model", {})
            export_model = create_model(
                num_classes=int(model_config.get("num_classes", 7)),
                pretrained=False,
                dropout_rate=float(model_config.get("dropout_rate", 0.3)),
                freeze_backbone=False,
                use_gradient_checkpointing=False,
            )
            cpu_state_dict = {
                name: tensor.detach().cpu()
                for name, tensor in best_model_state_dict.items()
            }
            export_model.load_state_dict(cpu_state_dict)
            export_model.eval()

            scripted_model = torch.jit.script(export_model)
            torchscript_path = output_dir / "checkpoint_best_torchscript.pt"
            scripted_model.save(str(torchscript_path))
            logger.info(f"Saved TorchScript model to: {torchscript_path}")
        except Exception as exc:
            logger.warning(f"TorchScript export skipped: {exc}")

    # Save best checkpoint
    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        if ema is not None and save_ema_for_best:
            ema_state = model.state_dict()
            for name, value in ema.shadow.items():
                if name in ema_state:
                    ema_state[name] = value.detach().clone()
            best_checkpoint = dict(checkpoint)
            best_checkpoint["model_state_dict"] = ema_state
            torch.save(best_checkpoint, best_path)
            _export_torchscript(best_checkpoint["model_state_dict"])
        else:
            torch.save(checkpoint, best_path)
            _export_torchscript(checkpoint["model_state_dict"])
        logger.info(f"Saved best model with val_loss: {metrics['val_loss']:.4f}")

    # Save epoch checkpoint (optional, controlled by config)
    save_epoch_checkpoints = config.get("output", {}).get(
        "save_epoch_checkpoints", False
    )
    if save_epoch_checkpoints:
        epoch_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Tuple[int, Dict[str, float]]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

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
    # Set random seed (training)
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Get device
    device = get_device()

    # Get data paths from config
    data_config = config.get("data", {})
    split_seed = data_config.get("split_seed", seed)
    if split_seed != seed:
        logger.info(f"Split seed: {split_seed}")
    use_stratified_group_kfold = bool(
        data_config.get("use_stratified_group_kfold", False)
    )
    kfold_config = data_config.get("kfold", {})
    kfold_n_splits = int(kfold_config.get("n_splits", 5) or 5)
    kfold_fold_index = int(kfold_config.get("fold_index", 0) or 0)
    kfold_group_column = str(kfold_config.get("group_column", "lesion_id"))

    if use_stratified_group_kfold:
        logger.info(
            "Using StratifiedGroupKFold (n_splits=%s, fold_index=%s, group_column=%s)",
            kfold_n_splits,
            kfold_fold_index,
            kfold_group_column,
        )

    labels_csv = Path(data_config.get("labels_csv", "data/HAM10000/labels.csv"))
    images_dir = Path(data_config.get("images_dir", "data/HAM10000/images"))

    # Load and split data
    logger.info("Loading and splitting data...")
    train_df, val_df, test_df = load_and_split_data(
        labels_csv=labels_csv,
        images_dir=images_dir,
        val_size=data_config.get("val_size", 0.15),
        test_size=data_config.get("test_size", 0.15),
        random_state=split_seed,
        lesion_aware=data_config.get("lesion_aware", True),
        use_stratified_group_kfold=use_stratified_group_kfold,
        kfold_n_splits=kfold_n_splits,
        kfold_fold_index=kfold_fold_index,
        kfold_group_column=kfold_group_column,
    )

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    # Save split information
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    # Get training config
    train_config = config.get("training", {})
    batch_size = train_config.get("batch_size", 32)
    num_workers = train_config.get("num_workers", 4)
    epochs = train_config.get("epochs", 30)
    lr = train_config.get("lr", 1e-4)
    weight_decay = train_config.get("weight_decay", 0.01)
    use_amp = train_config.get("use_amp", True) and device.type == "cuda"
    mixup_alpha = float(train_config.get("mixup_alpha", 0.0) or 0.0)
    cutmix_alpha = float(train_config.get("cutmix_alpha", 0.0) or 0.0)
    mixup_prob = float(train_config.get("mixup_prob", 0.0) or 0.0)
    cutmix_prob = float(train_config.get("cutmix_prob", 0.5) or 0.5)
    weighted_sampling_power = float(
        train_config.get("sampling_weight_power", 1.0) or 1.0
    )
    gradient_accumulation_steps = int(
        train_config.get("gradient_accumulation_steps", 1) or 1
    )
    if use_amp:
        logger.info("Using Automatic Mixed Precision (AMP)")
    if gradient_accumulation_steps > 1:
        logger.info(f"Using gradient accumulation: {gradient_accumulation_steps} steps")

    # Optional two-stage fine-tuning (head warmup -> full fine-tune)
    stage1_epochs = int(train_config.get("stage1_epochs", 0) or 0)
    stage2_epochs_cfg = train_config.get("stage2_epochs", None)
    stage_epoch_mode = str(train_config.get("stage_epoch_mode", "fit_total")).lower()
    if stage1_epochs > 0:
        if stage_epoch_mode == "explicit":
            if stage2_epochs_cfg is None:
                stage2_epochs = max(0, epochs - stage1_epochs)
            else:
                stage2_epochs = int(stage2_epochs_cfg)
            total_epochs = stage1_epochs + stage2_epochs
        else:
            stage2_epochs = max(0, epochs - stage1_epochs)
            total_epochs = epochs
    else:
        stage2_epochs = epochs
        total_epochs = epochs

    if stage_epoch_mode != "explicit" and stage2_epochs_cfg is not None:
        configured_total = stage1_epochs + int(stage2_epochs_cfg)
        if configured_total != epochs:
            logger.warning(
                "Ignoring stage2_epochs=%s to respect epochs=%s (stage_epoch_mode=fit_total). "
                "Set training.stage_epoch_mode=explicit to use stage1+stage2 exactly.",
                stage2_epochs_cfg,
                epochs,
            )

    stage1_lr = train_config.get("stage1_lr", lr)
    stage2_lr = train_config.get("stage2_lr", lr)
    stage1_weight_decay = train_config.get("stage1_weight_decay", weight_decay)
    stage2_weight_decay = train_config.get("stage2_weight_decay", weight_decay)

    # Exponential Moving Average (EMA) settings
    ema_config = train_config.get("ema", {})
    ema_enabled = bool(ema_config.get("enabled", False))
    ema_decay = float(ema_config.get("decay", 0.999))
    ema_use_for_eval = bool(ema_config.get("use_for_eval", True))
    ema_save_best = bool(ema_config.get("save_best", True))

    # Create DataLoaders
    logger.info("Creating DataLoaders...")
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
        weighted_sampling_power=weighted_sampling_power,
        pin_memory=(device.type == "cuda"),  # Pin memory only works on CUDA
        prefetch_factor=train_config.get("prefetch_factor", 2),
        persistent_workers=train_config.get("persistent_workers", True),
    )

    # Loss configuration and class weights
    loss_config = config.get("loss", {})

    # Calculate class weights for loss function
    class_weight_power = float(loss_config.get("class_weight_power", 1.0) or 1.0)
    class_weights = get_class_weights_for_loss(train_df, power=class_weight_power)
    class_weights = class_weights.to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")
    logger.info(
        "Balance controls | sampling_weight_power=%.3f, class_weight_power=%.3f",
        weighted_sampling_power,
        class_weight_power,
    )

    # Create model
    model_config = config.get("model", {})
    use_gradient_checkpointing = train_config.get("use_gradient_checkpointing", False)
    logger.info("Creating model (EfficientNet-B0)...")
    model = create_model(
        num_classes=model_config.get("num_classes", 7),
        pretrained=model_config.get("pretrained", True),
        dropout_rate=model_config.get("dropout_rate", 0.3),
        freeze_backbone=model_config.get("freeze_backbone", False),
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    if use_gradient_checkpointing:
        logger.info("Gradient checkpointing enabled (reduces memory usage)")

    model = model.to(device)

    ema = ModelEMA(model, decay=ema_decay) if ema_enabled else None
    if ema_enabled:
        logger.info(f"EMA enabled (decay={ema_decay})")

    logger.info(f"Model parameters: {model.get_total_params():,}")
    logger.info(f"Trainable parameters: {model.get_trainable_params():,}")

    # Create loss function

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

    # Create learning rate scheduler config
    scheduler_config = train_config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "cosine")

    def build_scheduler(optimizer: torch.optim.Optimizer, stage_epochs: int):
        if stage_epochs <= 0:
            return None
        if scheduler_type == "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]["lr"],
                epochs=stage_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=scheduler_config.get("warmup_pct", 0.1),
                anneal_strategy="cos",
            )
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get("T_0", 10),
            T_mult=scheduler_config.get("T_mult", 2),
            eta_min=scheduler_config.get("eta_min", 1e-6),
        )

    def build_optimizer_and_scheduler(
        stage_lr: float,
        stage_weight_decay: float,
        stage_epochs: int,
        only_classifier: bool,
    ):
        params = (
            model.classifier.parameters() if only_classifier else model.parameters()
        )
        optimizer = AdamW(
            params,
            lr=stage_lr,
            weight_decay=stage_weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = build_scheduler(optimizer, stage_epochs)
        return optimizer, scheduler

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if use_amp else None

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
        checkpoint = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        prev_metrics = checkpoint.get("metrics", {})
        if stage1_epochs > 0:
            logger.warning(
                "Staged training resume: optimizer/scheduler state will not be restored."
            )

        # Check if best checkpoint exists and load its metrics
        best_checkpoint_path = output_dir / "checkpoint_best.pt"
        if best_checkpoint_path.exists():
            logger.info("Found existing best checkpoint - loading its metrics")
            best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
            best_metrics = best_checkpoint.get("metrics", {})
            best_val_loss = best_metrics.get("val_loss", float("inf"))
            has_saved_best = True
            logger.info(
                f"Best validation loss from existing checkpoint: {best_val_loss:.4f}"
            )
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

    def run_stage(
        stage_name: str,
        stage_start_epoch: int,
        stage_epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        freeze_backbone: bool,
        enable_early_stopping: bool,
    ) -> bool:
        nonlocal best_val_loss, has_saved_best
        for epoch in range(stage_start_epoch, stage_start_epoch + stage_epochs):
            epoch_start = time.time()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"\n{stage_name} Epoch {epoch + 1}/{total_epochs} | LR: {current_lr:.2e}"
            )

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
                freeze_backbone=freeze_backbone,
                ema=ema,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                mixup_prob=mixup_prob,
                cutmix_prob=cutmix_prob,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

            if ema is not None and ema_use_for_eval:
                ema.apply_to(model)
                val_metrics = validate(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    device=device,
                    epoch=epoch + 1,
                )
                ema.restore(model)
            else:
                val_metrics = validate(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    device=device,
                    epoch=epoch + 1,
                )

            if scheduler is not None and scheduler_type != "onecycle":
                scheduler.step()

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["lr"].append(current_lr)

            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                has_saved_best = True

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
                ema=ema,
                save_ema_for_best=ema_save_best,
            )

            if enable_early_stopping and early_stopping(val_metrics["loss"]):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                return True
        return False

    stopped = False
    current_epoch = start_epoch

    # Stage 1: head warmup (optional)
    if stage1_epochs > 0 and current_epoch < stage1_epochs:
        if hasattr(model, "_freeze_backbone"):
            model._freeze_backbone()
        else:
            for param in model.backbone.parameters():
                param.requires_grad = False
        stage1_remaining = stage1_epochs - current_epoch
        optimizer, scheduler = build_optimizer_and_scheduler(
            stage1_lr,
            stage1_weight_decay,
            stage1_remaining,
            only_classifier=True,
        )
        stopped = run_stage(
            stage_name="Stage 1 (head warmup)",
            stage_start_epoch=current_epoch,
            stage_epochs=stage1_remaining,
            optimizer=optimizer,
            scheduler=scheduler,
            freeze_backbone=True,
            enable_early_stopping=False,
        )
        current_epoch = stage1_epochs

    # Stage 2: full fine-tune
    if not stopped and stage2_epochs > 0 and current_epoch < total_epochs:
        if hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone()
        else:
            for param in model.backbone.parameters():
                param.requires_grad = True
        stage2_remaining = total_epochs - current_epoch
        optimizer, scheduler = build_optimizer_and_scheduler(
            stage2_lr,
            stage2_weight_decay,
            stage2_remaining,
            only_classifier=False,
        )
        stopped = run_stage(
            stage_name="Stage 2 (fine-tune)",
            stage_start_epoch=current_epoch,
            stage_epochs=stage2_remaining,
            optimizer=optimizer,
            scheduler=scheduler,
            freeze_backbone=False,
            enable_early_stopping=True,
        )

    # Ensure we have a best checkpoint (use latest if no improvement occurred)
    if not has_saved_best:
        logger.warning(
            "No validation improvement during training. Saving latest as best."
        )
        import shutil

        shutil.copy(
            output_dir / "checkpoint_latest.pt", output_dir / "checkpoint_best.pt"
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
    parser = argparse.ArgumentParser(description="Train skin lesion classifier")
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
