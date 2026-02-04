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
    metrics = MetricTracker()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        if use_amp and device.type == "cuda":
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Update learning rate if using OneCycleLR
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        # Update metrics
        preds = torch.argmax(outputs, dim=1)
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
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
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
    # Set random seed
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Get device
    device = get_device()
    
    # Get data paths from config
    data_config = config.get("data", {})
    labels_csv = Path(data_config.get("labels_csv", "data/HAM10000/labels.csv"))
    images_dir = Path(data_config.get("images_dir", "data/HAM10000/images"))
    
    # Load and split data
    logger.info("Loading and splitting data...")
    train_df, val_df, test_df = load_and_split_data(
        labels_csv=labels_csv,
        images_dir=images_dir,
        val_size=data_config.get("val_size", 0.15),
        test_size=data_config.get("test_size", 0.15),
        random_state=seed,
        lesion_aware=data_config.get("lesion_aware", True),
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
    )
    
    # Calculate class weights for loss function
    class_weights = get_class_weights_for_loss(train_df)
    class_weights = class_weights.to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Create model
    model_config = config.get("model", {})
    logger.info("Creating model...")
    model = create_model(
        num_classes=model_config.get("num_classes", 7),
        model_size=model_config.get("size", "small"),
        pretrained=model_config.get("pretrained", True),
        dropout_rate=model_config.get("dropout_rate", 0.3),
        freeze_backbone=model_config.get("freeze_backbone", False),
        head_type=model_config.get("head_type", "simple"),
    )
    model = model.to(device)
    
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
            resume_from, model, optimizer, scheduler
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
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch + 1}/{epochs} | LR: {current_lr:.2e}")
        
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
        )
        
        # Validate
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
