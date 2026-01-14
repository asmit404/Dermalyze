import json
import sys


def analyze_fit(history_file):
    with open(history_file) as f:
        history = json.load(f)

    val_loss = history["val_loss"]
    train_loss = history["train_loss"]

    # Check last 5 epochs
    recent_gap = [train_loss[i] - val_loss[i] for i in range(-5, 0)]
    avg_gap = sum(recent_gap) / len(recent_gap)

    print(f"📊 Recent Gap (last 5 epochs): {avg_gap:.4f}")

    if avg_gap < -0.02:
        print("⚠️  UNDERFITTING - Model too weak, increase capacity")
        return "underfit"
    elif avg_gap < 0.05:
        print("✅ BALANCED - Good generalization!")
        return "balanced"
    elif avg_gap < 0.15:
        print("⚠️  MILD OVERFITTING - Increase regularization slightly")
        return "mild_overfit"
    else:
        print("🚨 SEVERE OVERFITTING - Apply aggressive regularization")
        return "severe_overfit"


if __name__ == "__main__":
    analyze_fit(sys.argv[1])
