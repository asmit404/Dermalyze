import argparse
import json
from pathlib import Path


def analyze_fit(history_file: Path):
    with open(history_file) as f:
        history = json.load(f)

    val_loss = history["val_loss"]
    train_loss = history["train_loss"]

    window = min(5, len(train_loss), len(val_loss))
    if window == 0:
        raise ValueError("training_history.json contains no loss values")

    # Check recent epochs (up to last 5)
    recent_gap = [train_loss[i] - val_loss[i] for i in range(-window, 0)]
    avg_gap = sum(recent_gap) / len(recent_gap)

    print(f" Recent Gap (last 5 epochs): {avg_gap:.4f}")

    if avg_gap < -0.02:
        print("  UNDERFITTING - Model too weak, increase capacity")
        return "underfit"
    elif avg_gap < 0.05:
        print(" BALANCED - Good generalization!")
        return "balanced"
    elif avg_gap < 0.15:
        print("  MILD OVERFITTING - Increase regularization slightly")
        return "mild_overfit"
    else:
        print(" SEVERE OVERFITTING - Apply aggressive regularization")
        return "severe_overfit"


def main():
    parser = argparse.ArgumentParser(
        description="Quickly diagnose underfitting/overfitting from training_history.json"
    )
    parser.add_argument(
        "history_file",
        type=Path,
        help="Path to training history JSON (e.g., outputs/run_xxx/training_history.json)",
    )
    args = parser.parse_args()
    analyze_fit(args.history_file)


if __name__ == "__main__":
    main()
