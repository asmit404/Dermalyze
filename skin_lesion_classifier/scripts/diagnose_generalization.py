"""
Diagnose underfitting/overfitting from held-out evaluation signals.

This script implements a stronger diagnosis workflow than training-curve-only checks:
1. Run `src/evaluate.py` (or reuse an existing evaluation directory)
2. Read macro/per-class metrics, confusion matrix, and calibration (ECE)
3. Analyze prediction confidence on mistakes
4. Optionally compare train-vs-test gap using checkpoint metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

def _resolve_project_path(project_root: Path, path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value
    return (project_root / path_value).resolve()


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _run_evaluation(
    *,
    project_root: Path,
    config_path: Path,
    checkpoints: List[Path],
    test_csv: Path,
    images_dir: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    use_tta: Optional[bool],
    tta_mode: Optional[str],
) -> None:
    command = [
        sys.executable,
        "src/evaluate.py",
        "--config",
        str(config_path),
        "--checkpoint",
        *[str(p) for p in checkpoints],
        "--test-csv",
        str(test_csv),
        "--images-dir",
        str(images_dir),
        "--output",
        str(output_dir),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
    ]
    if use_tta is not None:
        command.append("--use-tta" if use_tta else "--no-use-tta")
    if tta_mode is not None:
        command.extend(["--tta-mode", tta_mode])

    subprocess.run(command, cwd=str(project_root), check=True)


def _read_prediction_confidence(predictions_path: Path) -> Dict[str, Optional[float]]:
    with predictions_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{predictions_path} has no header")

        prob_columns = [c for c in reader.fieldnames if c.startswith("prob_")]
        if not prob_columns:
            raise ValueError(f"{predictions_path} has no prob_* columns")

        confidences: List[float] = []
        correct_confidences: List[float] = []
        wrong_confidences: List[float] = []
        high_conf_wrong = 0
        high_conf_threshold = 0.80
        labeled_samples = 0

        for row in reader:
            true_idx = int(row["true_idx"])
            if true_idx < 0:
                continue
            labeled_samples += 1

            pred_idx = int(row["pred_idx"])
            confidence = max(float(row[c]) for c in prob_columns)
            confidences.append(confidence)

            if pred_idx == true_idx:
                correct_confidences.append(confidence)
            else:
                wrong_confidences.append(confidence)
                if confidence >= high_conf_threshold:
                    high_conf_wrong += 1

    wrong_total = len(wrong_confidences)
    return {
        "mean_confidence": mean(confidences) if confidences else None,
        "mean_confidence_correct": (
            mean(correct_confidences) if correct_confidences else None
        ),
        "mean_confidence_incorrect": (
            mean(wrong_confidences) if wrong_confidences else None
        ),
        "high_confidence_error_rate": (
            high_conf_wrong / wrong_total if wrong_total > 0 else None
        ),
        "num_labeled_samples": int(labeled_samples),
    }


def _extract_checkpoint_train_metrics(
    checkpoint_path: Optional[Path],
) -> Dict[str, Optional[float]]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {"train_acc": None, "val_acc": None, "val_macro_f1": None}

    try:
        import torch
    except ModuleNotFoundError:
        print(
            "Warning: torch is not installed; skipping checkpoint train/val metric extraction."
        )
        return {"train_acc": None, "val_acc": None, "val_macro_f1": None}

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    metrics = checkpoint.get("metrics", {})

    train_acc = metrics.get("train_acc")
    val_acc = metrics.get("val_acc", metrics.get("accuracy"))
    val_macro_f1 = metrics.get("macro_f1")

    return {
        "train_acc": float(train_acc) if train_acc is not None else None,
        "val_acc": float(val_acc) if val_acc is not None else None,
        "val_macro_f1": float(val_macro_f1) if val_macro_f1 is not None else None,
    }


def _top_confusions(
    confusion_matrix: List[List[int]],
    class_names: List[str],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    pairs: List[Tuple[int, str, str]] = []
    for i, row in enumerate(confusion_matrix):
        for j, count in enumerate(row):
            if i == j:
                continue
            if count > 0:
                pairs.append((int(count), class_names[i], class_names[j]))
    pairs.sort(key=lambda x: x[0], reverse=True)

    return [
        {"count": count, "true_label": true_label, "pred_label": pred_label}
        for count, true_label, pred_label in pairs[:limit]
    ]


def _minority_gap(per_class_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    class_items = []
    for class_name, values in per_class_metrics.items():
        class_items.append(
            {
                "class_name": class_name,
                "support": int(values["support"]),
                "recall": float(values["recall"]),
            }
        )
    class_items.sort(key=lambda x: x["support"])

    k = min(3, len(class_items))
    minority = class_items[:k]
    majority = class_items[-k:]

    minority_recall = mean([x["recall"] for x in minority]) if minority else 0.0
    majority_recall = mean([x["recall"] for x in majority]) if majority else 0.0

    return {
        "minority_classes": [x["class_name"] for x in minority],
        "majority_classes": [x["class_name"] for x in majority],
        "minority_mean_recall": minority_recall,
        "majority_mean_recall": majority_recall,
        "minority_recall_gap": majority_recall - minority_recall,
    }


def _diagnose(
    *,
    accuracy: float,
    macro_recall: float,
    macro_f1: float,
    ece: float,
    minority_recall_gap: float,
    mean_confidence_incorrect: Optional[float],
    train_acc: Optional[float],
) -> Dict[str, Any]:
    underfit_signals: List[str] = []
    overfit_signals: List[str] = []

    if macro_f1 < 0.60:
        underfit_signals.append(f"Low macro F1 ({macro_f1:.4f})")
    if macro_recall < 0.60:
        underfit_signals.append(f"Low macro recall ({macro_recall:.4f})")
    if train_acc is not None and train_acc < 0.75:
        underfit_signals.append(f"Low train accuracy at best checkpoint ({train_acc:.4f})")

    if train_acc is not None and (train_acc - accuracy) >= 0.12:
        overfit_signals.append(
            f"Large train→test accuracy gap ({train_acc - accuracy:.4f})"
        )
    if ece >= 0.10 and mean_confidence_incorrect is not None and mean_confidence_incorrect >= 0.70:
        overfit_signals.append(
            f"Overconfident errors (ECE={ece:.4f}, mean wrong confidence={mean_confidence_incorrect:.4f})"
        )
    if minority_recall_gap >= 0.20:
        overfit_signals.append(
            f"Minority recall gap is large ({minority_recall_gap:.4f})"
        )

    if len(overfit_signals) >= 2 and len(underfit_signals) == 0:
        diagnosis = "overfitting"
    elif len(underfit_signals) >= 2 and len(overfit_signals) == 0:
        diagnosis = "underfitting"
    elif len(overfit_signals) > 0 and len(underfit_signals) > 0:
        diagnosis = "mixed_signals"
    else:
        diagnosis = "balanced_or_mild"

    return {
        "diagnosis": diagnosis,
        "underfitting_signals": underfit_signals,
        "overfitting_signals": overfit_signals,
    }


def _write_text_report(report: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "Generalization Diagnosis",
        "========================",
        f"Diagnosis: {report['diagnosis']['diagnosis']}",
        "",
        "Key Metrics",
        "-----------",
        f"Accuracy:       {_format_float(report['key_metrics']['accuracy'])}",
        f"Macro Recall:   {_format_float(report['key_metrics']['macro_recall'])}",
        f"Macro F1:       {_format_float(report['key_metrics']['macro_f1'])}",
        f"ECE:            {_format_float(report['key_metrics']['expected_calibration_error'])}",
        "",
        "Train vs Test",
        "-------------",
        f"Train Acc (ckpt): {_format_float(report['checkpoint_metrics']['train_acc'])}",
        f"Val Acc (ckpt):   {_format_float(report['checkpoint_metrics']['val_acc'])}",
        "",
        "Class Balance Signal",
        "--------------------",
        f"Minority classes: {', '.join(report['class_balance']['minority_classes'])}",
        f"Majority classes: {', '.join(report['class_balance']['majority_classes'])}",
        (
            "Minority recall gap (majority - minority): "
            f"{report['class_balance']['minority_recall_gap']:.4f}"
        ),
        "",
        "Confidence Signal",
        "-----------------",
        f"Mean confidence (all):      {_format_float(report['confidence']['mean_confidence'])}",
        (
            "Mean confidence (correct):  "
            f"{_format_float(report['confidence']['mean_confidence_correct'])}"
        ),
        (
            "Mean confidence (incorrect): "
            f"{_format_float(report['confidence']['mean_confidence_incorrect'])}"
        ),
        (
            "High-confidence error rate: "
            f"{_format_float(report['confidence']['high_confidence_error_rate'])}"
        ),
        "",
        "Top Confusions",
        "--------------",
    ]

    for entry in report["top_confusions"]:
        lines.append(
            f"{entry['count']} samples: true={entry['true_label']} predicted={entry['pred_label']}"
        )

    lines.append("")
    lines.append("Underfitting Signals")
    lines.append("--------------------")
    if report["diagnosis"]["underfitting_signals"]:
        lines.extend(report["diagnosis"]["underfitting_signals"])
    else:
        lines.append("None")

    lines.append("")
    lines.append("Overfitting Signals")
    lines.append("-------------------")
    if report["diagnosis"]["overfitting_signals"]:
        lines.extend(report["diagnosis"]["overfitting_signals"])
    else:
        lines.append("None")
    lines.append("")

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose underfitting/overfitting using held-out evaluation "
            "(macro/per-class metrics, confusion matrix, calibration, confidence)."
        )
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help=(
            "Training run directory (e.g., outputs/run_xxx). "
            "If provided, defaults checkpoint=test artifacts from this folder."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        default=None,
        help="Checkpoint path(s). Defaults to <run>/checkpoint_best.pt when --run is set.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Test split CSV. Defaults to <run>/test_split.csv when --run is set.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Images directory used for evaluation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Config file passed to evaluate.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Evaluation output directory (default: <run>/generalization_eval)",
    )
    parser.add_argument(
        "--reuse-evaluation",
        action="store_true",
        help="Reuse existing evaluation_metrics.json and predictions.csv from --output",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluate.py",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Data loader workers for evaluate.py",
    )
    parser.add_argument(
        "--use-tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override TTA flag passed to evaluate.py",
    )
    parser.add_argument(
        "--tta-mode",
        choices=["light", "medium", "full"],
        default=None,
        help="Optional TTA mode passed to evaluate.py",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    run_dir: Optional[Path] = None
    if args.run is not None:
        run_dir = _resolve_project_path(project_root, args.run)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoints = (
        [_resolve_project_path(project_root, p) for p in args.checkpoint]
        if args.checkpoint
        else ([run_dir / "checkpoint_best.pt"] if run_dir else None)
    )
    primary_checkpoint = checkpoints[0] if checkpoints else None

    test_csv = (
        _resolve_project_path(project_root, args.test_csv)
        if args.test_csv is not None
        else (run_dir / "test_split.csv" if run_dir else None)
    )
    images_dir = (
        _resolve_project_path(project_root, args.images_dir)
        if args.images_dir is not None
        else None
    )
    config_path = _resolve_project_path(project_root, args.config)

    output_dir = (
        _resolve_project_path(project_root, args.output)
        if args.output is not None
        else (
            (run_dir / "generalization_eval")
            if run_dir is not None
            else (project_root / "evaluation_results_generalization")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "evaluation_metrics.json"
    predictions_path = output_dir / "predictions.csv"

    if not args.reuse_evaluation:
        if not checkpoints:
            raise ValueError("Provide --checkpoint or --run (to infer checkpoint_best.pt).")
        if test_csv is None:
            raise ValueError("Provide --test-csv or --run (to infer test_split.csv).")
        if images_dir is None:
            raise ValueError("Provide --images-dir (required for evaluation).")

        _run_evaluation(
            project_root=project_root,
            config_path=config_path,
            checkpoints=checkpoints,
            test_csv=test_csv,
            images_dir=images_dir,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_tta=args.use_tta,
            tta_mode=args.tta_mode,
        )
    else:
        if not metrics_path.exists() or not predictions_path.exists():
            raise FileNotFoundError(
                "--reuse-evaluation was set, but required files are missing: "
                f"{metrics_path} and/or {predictions_path}"
            )

    with metrics_path.open("r") as f:
        metrics = json.load(f)

    if "per_class_metrics" not in metrics:
        raise ValueError(
            "evaluation_metrics.json has no per_class_metrics. "
            "Ensure evaluate.py ran with labeled test data."
        )

    class_names = list(metrics["per_class_metrics"].keys())
    class_balance = _minority_gap(metrics["per_class_metrics"])
    top_confusions = _top_confusions(metrics["confusion_matrix"], class_names)
    confidence_stats = _read_prediction_confidence(predictions_path)
    checkpoint_metrics = _extract_checkpoint_train_metrics(primary_checkpoint)

    diagnosis = _diagnose(
        accuracy=float(metrics["accuracy"]),
        macro_recall=float(metrics["macro_recall"]),
        macro_f1=float(metrics["macro_f1"]),
        ece=float(metrics["calibration"]["expected_calibration_error"]),
        minority_recall_gap=float(class_balance["minority_recall_gap"]),
        mean_confidence_incorrect=confidence_stats["mean_confidence_incorrect"],
        train_acc=checkpoint_metrics["train_acc"],
    )

    report = {
        "diagnosis": diagnosis,
        "key_metrics": {
            "accuracy": float(metrics["accuracy"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "expected_calibration_error": float(
                metrics["calibration"]["expected_calibration_error"]
            ),
        },
        "checkpoint_metrics": checkpoint_metrics,
        "class_balance": class_balance,
        "confidence": confidence_stats,
        "top_confusions": top_confusions,
        "artifacts": {
            "evaluation_metrics_json": str(metrics_path),
            "predictions_csv": str(predictions_path),
            "report_json": str(output_dir / "generalization_diagnosis.json"),
            "report_txt": str(output_dir / "generalization_diagnosis.txt"),
        },
    }

    report_json_path = output_dir / "generalization_diagnosis.json"
    with report_json_path.open("w") as f:
        json.dump(report, f, indent=2)

    report_txt_path = output_dir / "generalization_diagnosis.txt"
    _write_text_report(report, report_txt_path)

    print("\n" + "=" * 68)
    print("GENERALIZATION DIAGNOSIS")
    print("=" * 68)
    print(f"Diagnosis:          {report['diagnosis']['diagnosis']}")
    print(f"Accuracy:           {report['key_metrics']['accuracy']:.4f}")
    print(f"Macro Recall:       {report['key_metrics']['macro_recall']:.4f}")
    print(f"Macro F1:           {report['key_metrics']['macro_f1']:.4f}")
    print(
        "ECE (Calibration):  "
        f"{report['key_metrics']['expected_calibration_error']:.4f}"
    )
    print(f"Report JSON:        {report_json_path}")
    print(f"Report Text:        {report_txt_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
