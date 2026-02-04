"""
Feature-Space SMOTEENN Training Script.

This script implements a two-stage training process to handle class imbalance:
1. Feature Extraction: Passes images through a frozen EfficientNet backbone.
2. Rebalancing: Applies SMOTEENN to the extracted feature vectors.
3. Head Training: Trains the classification head on the balanced features.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from imblearn.combine import SMOTEENN
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import (
    load_and_split_data,
    create_dataloaders,
    CLASS_LABELS,
)
from src.models.efficientnet import create_model, get_loss_function
from src.train import (
    set_seed,
    load_config,
    get_device,
    MetricTracker,
    save_checkpoint,
    EarlyStopping  # Reusing EarlyStopping from train.py if available, or redefining it
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from the frozen backbone.
    
    Args:
        model: Full model (backbone + classifier)
        loader: DataLoader
        device: Device
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    model.eval()
    features_list = []
    labels_list = []
    
    logger.info("Extracting features...")
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Extraction"):
            images = images.to(device)
            # Use the backbone directly
            feats = model.backbone(images)
            # Flatten if necessary (EfficientNet backbone usually outputs [N, C] or [N, C, 1, 1])
            if len(feats.shape) > 2:
                feats = feats.flatten(start_dim=1)
                
            features_list.append(feats.cpu().numpy())
            labels_list.append(targets.numpy())
            
    return np.vstack(features_list), np.concatenate(labels_list)


def train_head_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train only the classifier head on features."""
    model.classifier.train()
    metrics = MetricTracker()
    
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass through classifier only
        outputs = model.classifier(features)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(outputs, dim=1)
        metrics.update(loss.item(), preds, targets)
        
    return metrics.compute()


@torch.no_grad()
def validate_head(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the classifier head."""
    model.classifier.eval()
    metrics = MetricTracker()
    
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        
        outputs = model.classifier(features)
        loss = criterion(outputs, targets)
        
        preds = torch.argmax(outputs, dim=1)
        metrics.update(loss.item(), preds, targets)
        
    return metrics.compute()


def main():
    parser = argparse.ArgumentParser(description="Train SMOTEENN model")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for head training")
    args = parser.parse_args()

    # Load Config
    config = load_config(args.config)
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    
    # Setup Output Dir
    if args.output:
        output_dir = args.output
    else:
        from datetime import datetime
        output_dir = Path("outputs") / f"run_smoteenn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    
    # 1. Data Loading
    logger.info("Loading Data...")
    data_config = config.get("data", {})
    labels_csv = Path(data_config.get("labels_csv", "data/HAM10000/labels.csv"))
    images_dir = Path(data_config.get("images_dir", "data/HAM10000/images"))
    
    train_df, val_df, test_df = load_and_split_data(
        labels_csv, images_dir, 
        val_size=data_config.get("val_size", 0.15),
        test_size=data_config.get("test_size", 0.15),
        random_state=seed
    )
    
    # Create standard dataloaders for feature extraction
    train_config = config.get("training", {})
    train_loader, val_loader, _ = create_dataloaders(
        train_df, val_df, test_df, images_dir,
        batch_size=train_config.get("batch_size", 32),
        num_workers=train_config.get("num_workers", 4),
        use_weighted_sampling=False,
        augmentation_strength="light"
    )
    
    # 2. Model Setup
    logger.info("Initializing Model...")
    model_config = config.get("model", {})
    model = create_model(
        num_classes=model_config.get("num_classes", 7),
        model_size=model_config.get("size", "small"),
        pretrained=True,
        freeze_backbone=True
    ).to(device)
    
    # 3. Extract Features
    logger.info("Step 1/3: Extracting Features from Training Set...")
    X_train_feats, y_train = extract_features(model, train_loader, device)
    logger.info(f"Extracted features shape: {X_train_feats.shape}")
    
    logger.info("Extracting Features from Validation Set...")
    X_val_feats, y_val = extract_features(model, val_loader, device)

    # 4. Apply SMOTEENN
    logger.info("Step 2/3: Applying SMOTEENN...")
    logger.info(f"Original Class Distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    smote_enn = SMOTEENN(random_state=seed)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train_feats, y_train)
    
    logger.info(f"Resampled Class Distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    logger.info(f"New Training Size: {X_resampled.shape[0]}")
    
    # Create Feature Datasets
    train_feat_dataset = TensorDataset(
        torch.tensor(X_resampled, dtype=torch.float32),
        torch.tensor(y_resampled, dtype=torch.long)
    )
    val_feat_dataset = TensorDataset(
        torch.tensor(X_val_feats, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=256, shuffle=True)
    val_feat_loader = DataLoader(val_feat_dataset, batch_size=256, shuffle=False)
    
    # 5. Train Head
    logger.info("Step 3/3: Training Classifier Head...")
    
    # Re-initialize classifier weights to be safe
    model._initialize_classifier()
    
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss() # Standard CE is fine now that data is balanced
    
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=10)
    
    for epoch in range(args.epochs):
        train_metrics = train_head_one_epoch(model, train_feat_loader, criterion, optimizer, device)
        val_metrics = validate_head(model, val_feat_loader, criterion, device)
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}"
        )
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_metrics = {
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
            }
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch,
                metrics=save_metrics,
                config=config,
                output_dir=output_dir,
                is_best=True
            )
            
        if early_stopping(val_metrics['loss']):
            logger.info("Early stopping triggered")
            break

    logger.info(f"Training Complete. Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
