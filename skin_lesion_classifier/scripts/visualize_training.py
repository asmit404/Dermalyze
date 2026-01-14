"""
Visualize Training Progress and Detect Overfitting

This script creates comprehensive visualizations of training metrics
to help identify and diagnose overfitting issues.

Usage:
    python scripts/visualize_training.py --run outputs/run_20260113_132017
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")


def plot_training_curves(history, output_path):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Training vs Validation Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    best_epoch = np.argmin(history['val_loss']) + 1
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.axhline(y=min(history['val_loss']), color='purple', linestyle=':', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Training vs Validation Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Overfitting Gap (Generalization Gap)
    ax = axes[1, 0]
    loss_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' for gap in loss_gap]
    ax.bar(epochs, loss_gap, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.05)')
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Severe (0.10)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss - Training Loss', fontsize=12)
    ax.set_title('Overfitting Gap (Red = Severe, Orange = Moderate, Green = Good)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curves to: {output_path}")
    plt.close()


def print_overfitting_analysis(history):
    """Print detailed overfitting analysis."""
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    train_acc = np.array(history['train_acc'])
    val_acc = np.array(history['val_acc'])
    
    # Find best epoch
    best_epoch = np.argmin(val_loss)
    
    # Calculate gaps
    final_loss_gap = val_loss[-1] - train_loss[-1]
    final_acc_gap = train_acc[-1] - val_acc[-1]
    best_loss_gap = val_loss[best_epoch] - train_loss[best_epoch]
    
    # Trend analysis
    val_loss_increasing = val_loss[-1] > val_loss[best_epoch]
    train_loss_decreasing = train_loss[-1] < train_loss[best_epoch]
    
    print("\n" + "=" * 70)
    print("OVERFITTING ANALYSIS")
    print("=" * 70)
    
    print(f"\n📊 Final Metrics (Epoch {len(train_loss)}):")
    print(f"  Training Loss:     {train_loss[-1]:.4f}")
    print(f"  Validation Loss:   {val_loss[-1]:.4f}")
    print(f"  Training Accuracy: {train_acc[-1]:.4f}")
    print(f"  Validation Accuracy: {val_acc[-1]:.4f}")
    
    print(f"\n🎯 Best Epoch: {best_epoch + 1}")
    print(f"  Best Val Loss:     {val_loss[best_epoch]:.4f}")
    print(f"  Best Val Accuracy: {val_acc[best_epoch]:.4f}")
    
    print(f"\n📈 Generalization Gaps:")
    print(f"  Final Loss Gap:    {final_loss_gap:.4f} ({final_loss_gap*100:.1f}%)")
    print(f"  Final Accuracy Gap: {final_acc_gap:.4f} ({final_acc_gap*100:.1f}%)")
    print(f"  Best Loss Gap:     {best_loss_gap:.4f}")
    
    # Diagnosis
    print(f"\n🔍 Diagnosis:")
    if final_loss_gap < 0.05:
        print("  ✅ GOOD: Minimal overfitting detected")
        severity = "GOOD"
    elif final_loss_gap < 0.10:
        print("  ⚠️  MODERATE: Some overfitting present")
        severity = "MODERATE"
    else:
        print("  🚨 SEVERE: Significant overfitting detected!")
        severity = "SEVERE"
    
    if val_loss_increasing and train_loss_decreasing:
        print("  ⚠️  Validation loss increasing while training loss decreasing")
        print("     → Classic overfitting pattern")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if severity == "SEVERE":
        print("  1. Use config_anti_overfit.yaml for next training")
        print("  2. Increase dropout to 0.5-0.6")
        print("  3. Increase weight decay to 0.05-0.1")
        print("  4. Use heavy augmentation")
        print("  5. Reduce learning rate by 50%")
        print("  6. Implement early stopping (patience=5)")
        print(f"  7. Use model from epoch {best_epoch + 1} instead of latest")
    elif severity == "MODERATE":
        print("  1. Increase dropout to 0.4-0.5")
        print("  2. Increase weight decay to 0.03-0.05")
        print("  3. Switch augmentation from 'medium' to 'heavy'")
        print("  4. Reduce early stopping patience to 7-10")
    else:
        print("  1. Continue training with current settings")
        print("  2. Monitor for future overfitting")
        print("  3. Consider slight increase in model complexity if underfitting")
    
    print("=" * 70 + "\n")
    
    return severity


def main():
    parser = argparse.ArgumentParser(description="Visualize training progress")
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Path to training run directory (e.g., outputs/run_20260113_132017)"
    )
    args = parser.parse_args()
    
    # Load training history
    history_path = args.run / "training_history.json"
    if not history_path.exists():
        print(f"❌ Error: {history_path} not found")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print(f"\n📂 Analyzing run: {args.run.name}")
    
    # Create visualizations
    output_path = args.run / "training_analysis.png"
    plot_training_curves(history, output_path)
    
    # Print analysis
    severity = print_overfitting_analysis(history)
    
    # Save analysis to file
    analysis_path = args.run / "overfitting_analysis.txt"
    with open(analysis_path, 'w') as f:
        f.write(f"Overfitting Severity: {severity}\n")
        f.write(f"See training_analysis.png for visualizations\n")
    
    print(f"✓ Analysis saved to: {args.run}")


if __name__ == "__main__":
    main()
