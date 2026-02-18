#!/usr/bin/env python3
"""
Run a 5-fold StratifiedGroupKFold sweep and aggregate metrics.

This script:
1) Forces fold indices 0..4 using config.data.kfold.fold_index
2) Trains one model per fold
3) Evaluates each fold model on its test split
4) Optionally evaluates each fold with TTA
5) Aggregates standard (and optional TTA) metrics into summary JSON
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config at {path}")
    return data


def write_yaml(path: Path, content: Dict[str, Any]) -> None:
    with open(path, "w") as file:
        yaml.dump(content, file, default_flow_style=False)


def create_fold_config(base_config_path: Path, fold_index: int, output_path: Path) -> Path:
    config = load_yaml(base_config_path)

    data_cfg = config.setdefault("data", {})
    data_cfg["use_stratified_group_kfold"] = True
    kfold_cfg = data_cfg.setdefault("kfold", {})
    kfold_cfg["n_splits"] = 5
    kfold_cfg["fold_index"] = fold_index
    kfold_cfg.setdefault("group_column", "lesion_id")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
    return output_path


def run_command(command: List[str], cwd: Path) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True, capture_output=False)


def build_command_plan(
    fold_config_paths: Dict[int, Path],
    output_root: Path,
    images_dir: Path,
    run_tta: bool,
    tta_mode: str,
    tta_aggregation: str,
) -> List[str]:
    commands: List[str] = []
    for fold_index in range(5):
        fold_dir = output_root / f"fold_{fold_index}"
        fold_config = fold_config_paths[fold_index]
        commands.append(
            f"{sys.executable} src/train.py --config {fold_config} --output {fold_dir}"
        )
        commands.append(
            f"{sys.executable} src/evaluate.py --checkpoint {fold_dir / 'checkpoint_best.pt'} --test-csv {fold_dir / 'test_split.csv'} --images-dir {images_dir} --output {fold_dir / 'evaluation_results'}"
        )
        if run_tta:
            commands.append(
                f"{sys.executable} src/evaluate.py --checkpoint {fold_dir / 'checkpoint_best.pt'} --test-csv {fold_dir / 'test_split.csv'} --images-dir {images_dir} --output {fold_dir / 'evaluation_results_tta'} --use-tta --tta-mode {tta_mode} --tta-aggregation {tta_aggregation}"
            )
    return commands


def aggregate_numeric_metrics(metrics_by_fold: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "roc_auc_macro",
    ]

    agg: Dict[str, Any] = {}
    for key in keys:
        values = [float(entry[key]) for entry in metrics_by_fold if key in entry]
        if not values:
            continue
        agg[key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    ece_values = []
    for entry in metrics_by_fold:
        calibration = entry.get("calibration", {})
        if isinstance(calibration, dict) and "expected_calibration_error" in calibration:
            ece_values.append(float(calibration["expected_calibration_error"]))

    if ece_values:
        agg["expected_calibration_error"] = {
            "mean": mean(ece_values),
            "std": pstdev(ece_values) if len(ece_values) > 1 else 0.0,
            "min": min(ece_values),
            "max": max(ece_values),
        }

    return agg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 5-fold stratified-group sweep and aggregate fold metrics"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Base config path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root directory (default: outputs/kfold_sweep_TIMESTAMP)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root where src/train.py and src/evaluate.py exist",
    )
    parser.add_argument(
        "--run-tta",
        action="store_true",
        help="Also run TTA evaluation per fold and aggregate separately",
    )
    parser.add_argument(
        "--tta-mode",
        choices=["light", "medium", "full"],
        default="medium",
        help="TTA mode to use when --run-tta is enabled",
    )
    parser.add_argument(
        "--tta-aggregation",
        choices=["mean", "geometric_mean", "max"],
        default="mean",
        help="TTA aggregation method for evaluate.py",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    base_config_path = (project_root / args.config).resolve() if not args.config.is_absolute() else args.config
    if not base_config_path.exists():
        raise FileNotFoundError(f"Config not found: {base_config_path}")

    base_config = load_yaml(base_config_path)
    data_cfg = base_config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("Invalid config.data section")

    images_dir_value = data_cfg.get("images_dir", "data/HAM10000/images")
    images_dir = (project_root / images_dir_value).resolve()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = project_root / "outputs" / f"kfold_sweep_{timestamp}"
    else:
        output_root = args.output if args.output.is_absolute() else project_root / args.output

    output_root.mkdir(parents=True, exist_ok=True)

    fold_config_dir = output_root / "fold_configs"
    fold_config_paths: Dict[int, Path] = {}
    for fold_index in range(5):
        fold_config_paths[fold_index] = create_fold_config(
            base_config_path=base_config_path,
            fold_index=fold_index,
            output_path=fold_config_dir / f"fold_{fold_index}.yaml",
        )

    command_plan = build_command_plan(
        fold_config_paths=fold_config_paths,
        output_root=output_root,
        images_dir=images_dir,
        run_tta=args.run_tta,
        tta_mode=args.tta_mode,
        tta_aggregation=args.tta_aggregation,
    )

    plan_path = output_root / "kfold_command_plan.sh"
    plan_lines = ["#!/bin/bash", "set -euo pipefail", "", "# 5-fold sweep command plan (fold indices 0..4)"]
    plan_lines.extend(command_plan)
    plan_path.write_text("\n".join(plan_lines) + "\n")
    plan_path.chmod(0o755)

    print("=" * 70)
    print("Running 5-fold sweep (fold indices 0..4)")
    print("=" * 70)
    print(f"Base config: {base_config_path}")
    print(f"Output root: {output_root}")
    print(f"Command plan: {plan_path}")
    print()

    fold_results: List[Dict[str, Any]] = []
    fold_results_tta: List[Dict[str, Any]] = []

    for fold_index in range(5):
        fold_dir = output_root / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 70)
        print(f"Fold {fold_index}/4")
        print("-" * 70)

        train_command = [
            sys.executable,
            "src/train.py",
            "--config",
            str(fold_config_paths[fold_index]),
            "--output",
            str(fold_dir),
        ]
        run_command(train_command, cwd=project_root)

        checkpoint_path = fold_dir / "checkpoint_best.pt"
        test_csv_path = fold_dir / "test_split.csv"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        if not test_csv_path.exists():
            raise FileNotFoundError(f"Missing test split: {test_csv_path}")

        eval_output_dir = fold_dir / "evaluation_results"
        eval_command = [
            sys.executable,
            "src/evaluate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--test-csv",
            str(test_csv_path),
            "--images-dir",
            str(images_dir),
            "--output",
            str(eval_output_dir),
        ]
        run_command(eval_command, cwd=project_root)

        metrics_path = eval_output_dir / "evaluation_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing fold metrics: {metrics_path}")

        with open(metrics_path, "r") as file:
            metrics = json.load(file)

        fold_results.append(
            {
                "fold_index": fold_index,
                "fold_config": str(fold_config_paths[fold_index]),
                "output_dir": str(fold_dir),
                "metrics_path": str(metrics_path),
                "metrics": metrics,
            }
        )

        if args.run_tta:
            eval_tta_output_dir = fold_dir / "evaluation_results_tta"
            eval_tta_command = [
                sys.executable,
                "src/evaluate.py",
                "--checkpoint",
                str(checkpoint_path),
                "--test-csv",
                str(test_csv_path),
                "--images-dir",
                str(images_dir),
                "--output",
                str(eval_tta_output_dir),
                "--use-tta",
                "--tta-mode",
                args.tta_mode,
                "--tta-aggregation",
                args.tta_aggregation,
            ]
            run_command(eval_tta_command, cwd=project_root)

            metrics_tta_path = eval_tta_output_dir / "evaluation_metrics.json"
            if not metrics_tta_path.exists():
                raise FileNotFoundError(f"Missing fold TTA metrics: {metrics_tta_path}")

            with open(metrics_tta_path, "r") as file:
                metrics_tta = json.load(file)

            fold_results_tta.append(
                {
                    "fold_index": fold_index,
                    "fold_config": str(fold_config_paths[fold_index]),
                    "output_dir": str(fold_dir),
                    "metrics_path": str(metrics_tta_path),
                    "metrics": metrics_tta,
                }
            )

    metrics_only = [entry["metrics"] for entry in fold_results]
    aggregated = aggregate_numeric_metrics(metrics_only)

    best_by_macro_f1 = max(
        fold_results,
        key=lambda entry: float(entry["metrics"].get("macro_f1", float("-inf"))),
    )
    best_by_accuracy = max(
        fold_results,
        key=lambda entry: float(entry["metrics"].get("accuracy", float("-inf"))),
    )

    aggregated_tta: Dict[str, Any] | None = None
    best_tta_by_macro_f1: Dict[str, Any] | None = None
    best_tta_by_accuracy: Dict[str, Any] | None = None
    if fold_results_tta:
        metrics_tta_only = [entry["metrics"] for entry in fold_results_tta]
        aggregated_tta = aggregate_numeric_metrics(metrics_tta_only)
        best_tta_by_macro_f1 = max(
            fold_results_tta,
            key=lambda entry: float(entry["metrics"].get("macro_f1", float("-inf"))),
        )
        best_tta_by_accuracy = max(
            fold_results_tta,
            key=lambda entry: float(entry["metrics"].get("accuracy", float("-inf"))),
        )

    summary = {
        "sweep_type": "stratified_group_kfold",
        "fold_indices": [0, 1, 2, 3, 4],
        "base_config": str(base_config_path),
        "output_root": str(output_root),
        "command_plan_path": str(plan_path),
        "num_folds": len(fold_results),
        "aggregate_metrics": aggregated,
        "best_fold": {
            "by_macro_f1": {
                "fold_index": best_by_macro_f1["fold_index"],
                "macro_f1": float(best_by_macro_f1["metrics"].get("macro_f1", 0.0)),
            },
            "by_accuracy": {
                "fold_index": best_by_accuracy["fold_index"],
                "accuracy": float(best_by_accuracy["metrics"].get("accuracy", 0.0)),
            },
        },
        "folds": fold_results,
    }

    if fold_results_tta and aggregated_tta is not None and best_tta_by_macro_f1 is not None and best_tta_by_accuracy is not None:
        summary["tta"] = {
            "enabled": True,
            "tta_mode": args.tta_mode,
            "tta_aggregation": args.tta_aggregation,
            "num_folds": len(fold_results_tta),
            "aggregate_metrics": aggregated_tta,
            "best_fold": {
                "by_macro_f1": {
                    "fold_index": best_tta_by_macro_f1["fold_index"],
                    "macro_f1": float(best_tta_by_macro_f1["metrics"].get("macro_f1", 0.0)),
                },
                "by_accuracy": {
                    "fold_index": best_tta_by_accuracy["fold_index"],
                    "accuracy": float(best_tta_by_accuracy["metrics"].get("accuracy", 0.0)),
                },
            },
            "folds": fold_results_tta,
        }
    else:
        summary["tta"] = {
            "enabled": False,
        }

    summary_path = output_root / "kfold_summary.json"
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)

    print()
    print("=" * 70)
    print("5-fold sweep complete")
    print("=" * 70)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
