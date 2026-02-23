#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for the skin lesion classifier.

This script is intentionally wired to the existing training entrypoint:
    python src/train.py --config <trial_config> --output <trial_output>

For each trial it:
1) Loads a base YAML config
2) Samples hyperparameters from a HAM10000-focused search space
3) Writes a per-trial config
4) Runs training as a subprocess
5) Reads outputs/<study>/trial_xxxxx/training_history.json
6) Returns objective metric to Optuna
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import optuna
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config at {path}")
    return data


def write_yaml(path: Path, content: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        yaml.dump(content, file, default_flow_style=False)


def ensure_existing_data_paths(
    config: Dict[str, Any], project_root: Path
) -> Dict[str, Any]:
    """
    Ensure configured data paths exist, with safe fallback to raw HAM10000 paths.

    This helps when config points to balanced_19k files that were not generated yet.
    """
    data_cfg = config.setdefault("data", {})

    labels_csv = Path(str(data_cfg.get("labels_csv", "data/HAM10000/labels.csv")))
    images_dir = Path(str(data_cfg.get("images_dir", "data/HAM10000/images")))

    labels_abs = labels_csv if labels_csv.is_absolute() else project_root / labels_csv
    images_abs = images_dir if images_dir.is_absolute() else project_root / images_dir

    fallback_labels = project_root / "data/HAM10000/labels.csv"
    fallback_images = project_root / "data/HAM10000/images"

    if not labels_abs.exists() and fallback_labels.exists():
        print(
            f"[tune_optuna] labels_csv not found at {labels_csv}. "
            f"Falling back to {fallback_labels.relative_to(project_root)}"
        )
        data_cfg["labels_csv"] = str(fallback_labels.relative_to(project_root))

    if not images_abs.exists() and fallback_images.exists():
        print(
            f"[tune_optuna] images_dir not found at {images_dir}. "
            f"Falling back to {fallback_images.relative_to(project_root)}"
        )
        data_cfg["images_dir"] = str(fallback_images.relative_to(project_root))

    final_labels = Path(str(data_cfg.get("labels_csv")))
    final_images = Path(str(data_cfg.get("images_dir")))
    final_labels_abs = (
        final_labels if final_labels.is_absolute() else project_root / final_labels
    )
    final_images_abs = (
        final_images if final_images.is_absolute() else project_root / final_images
    )

    if not final_labels_abs.exists():
        raise FileNotFoundError(
            "labels_csv does not exist: "
            f"{final_labels_abs}. Provide a valid path in config or run prepare_data.py first."
        )
    if not final_images_abs.exists():
        raise FileNotFoundError(
            "images_dir does not exist: "
            f"{final_images_abs}. Provide a valid path in config."
        )

    return config


def _ensure_sections(
    config: Dict[str, Any],
) -> Tuple[
    Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]
]:
    model_cfg = config.setdefault("model", {})
    train_cfg = config.setdefault("training", {})
    loss_cfg = config.setdefault("loss", {})
    scheduler_cfg = train_cfg.setdefault("scheduler", {})
    ema_cfg = train_cfg.setdefault("ema", {})
    return model_cfg, train_cfg, loss_cfg, scheduler_cfg, ema_cfg


def apply_ham10000_search_space(
    config: Dict[str, Any], trial: optuna.Trial
) -> Dict[str, Any]:
    model_cfg, train_cfg, loss_cfg, scheduler_cfg, ema_cfg = _ensure_sections(config)

    train_cfg["epochs"] = trial.suggest_int("epochs", 18, 36)
    train_cfg["batch_size"] = trial.suggest_categorical(
        "batch_size", [8, 12, 16, 24, 32]
    )
    train_cfg["lr"] = trial.suggest_float("lr", 1e-5, 8e-4, log=True)
    train_cfg["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-4, 8e-2, log=True
    )

    stage1_epochs = trial.suggest_int("stage1_epochs", 0, 8)
    train_cfg["stage1_epochs"] = stage1_epochs
    train_cfg["stage_epoch_mode"] = "fit_total"
    train_cfg["stage2_epochs"] = max(0, int(train_cfg["epochs"]) - stage1_epochs)
    train_cfg["stage1_lr"] = trial.suggest_float("stage1_lr", 5e-5, 2e-3, log=True)
    train_cfg["stage2_lr"] = train_cfg["lr"]
    train_cfg["stage1_weight_decay"] = trial.suggest_float(
        "stage1_weight_decay", 5e-5, 5e-2, log=True
    )
    train_cfg["stage2_weight_decay"] = train_cfg["weight_decay"]

    model_cfg["dropout_rate"] = trial.suggest_float("dropout_rate", 0.2, 0.6)

    train_cfg["augmentation"] = trial.suggest_categorical(
        "augmentation", ["light", "medium", "domain", "randaugment"]
    )

    train_cfg["sampling_weight_power"] = trial.suggest_float(
        "sampling_weight_power", 0.2, 0.8
    )
    loss_cfg["class_weight_power"] = trial.suggest_float("class_weight_power", 0.2, 0.8)

    loss_cfg["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.08)
    loss_cfg["type"] = trial.suggest_categorical(
        "loss_type", ["focal", "cross_entropy", "label_smoothing"]
    )

    train_cfg["mixup_alpha"] = trial.suggest_float("mixup_alpha", 0.0, 0.4)
    train_cfg["cutmix_alpha"] = trial.suggest_float("cutmix_alpha", 0.0, 0.4)
    train_cfg["mixup_prob"] = trial.suggest_float("mixup_prob", 0.1, 0.7)
    train_cfg["cutmix_prob"] = trial.suggest_float("cutmix_prob", 0.1, 0.9)

    train_cfg["early_stopping_patience"] = trial.suggest_int(
        "early_stopping_patience", 5, 12
    )

    scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "onecycle"])
    scheduler_cfg["type"] = scheduler_type
    if scheduler_type == "cosine":
        scheduler_cfg["T_0"] = trial.suggest_int("cosine_T0", 8, 24)
        scheduler_cfg["T_mult"] = trial.suggest_int("cosine_Tmult", 1, 2)
        scheduler_cfg["eta_min"] = trial.suggest_float(
            "cosine_eta_min", 1e-7, 2e-5, log=True
        )
        scheduler_cfg.pop("warmup_pct", None)
    else:
        scheduler_cfg["warmup_pct"] = trial.suggest_float(
            "onecycle_warmup_pct", 0.05, 0.3
        )
        scheduler_cfg.pop("T_0", None)
        scheduler_cfg.pop("T_mult", None)
        scheduler_cfg.pop("eta_min", None)

    ema_enabled = trial.suggest_categorical("ema_enabled", [True, False])
    ema_cfg["enabled"] = ema_enabled
    if ema_enabled:
        ema_cfg["decay"] = trial.suggest_float("ema_decay", 0.995, 0.9999)
        ema_cfg["use_for_eval"] = True
        ema_cfg["save_best"] = True

    return config


def resolve_objective(eval_metrics: Dict[str, Any], objective_metric: str) -> float:
    def _get_metric(name: str) -> float:
        value = eval_metrics.get(name)
        if value is None:
            raise ValueError(f"evaluation_metrics.json missing {name}")
        return float(value)

    if objective_metric == "macro_recall":
        return _get_metric("macro_recall")

    if objective_metric == "macro_f1":
        return _get_metric("macro_f1")

    if objective_metric == "weighted_recall":
        return _get_metric("weighted_recall")

    if objective_metric == "weighted_f1":
        return _get_metric("weighted_f1")

    if objective_metric == "macro_recall_f1_mean":
        macro_recall = _get_metric("macro_recall")
        macro_f1 = _get_metric("macro_f1")
        return (macro_recall + macro_f1) / 2.0

    raise ValueError(f"Unsupported objective metric: {objective_metric}")


def run_trial_training(
    project_root: Path,
    trial_config_path: Path,
    trial_output_dir: Path,
    verbose_subprocess: bool,
) -> None:
    command = [
        sys.executable,
        "src/train.py",
        "--config",
        str(trial_config_path),
        "--output",
        str(trial_output_dir),
    ]

    if verbose_subprocess:
        print("$", shlex.join(command))
        subprocess.run(command, cwd=project_root, check=True)
        return

    log_path = trial_output_dir / "trial_train.log"
    trial_output_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log_file:
        log_file.write(f"\n$ {shlex.join(command)}\n")
        log_file.flush()
        subprocess.run(
            command,
            cwd=project_root,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )


def run_trial_evaluation(
    project_root: Path,
    trial_output_dir: Path,
    trial_config: Dict[str, Any],
    verbose_subprocess: bool,
) -> Dict[str, Any]:
    checkpoint_path = trial_output_dir / "checkpoint_best.pt"
    val_csv_path = trial_output_dir / "val_split.csv"
    eval_output_dir = trial_output_dir / "optuna_eval"
    eval_metrics_path = eval_output_dir / "evaluation_metrics.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Missing validation split: {val_csv_path}")

    images_dir = Path(
        trial_config.get("data", {}).get("images_dir", "data/HAM10000/images")
    )

    command = [
        sys.executable,
        "src/evaluate.py",
        "--checkpoint",
        str(checkpoint_path),
        "--test-csv",
        str(val_csv_path),
        "--images-dir",
        str(images_dir),
        "--output",
        str(eval_output_dir),
    ]

    if verbose_subprocess:
        print("$", shlex.join(command))
        subprocess.run(command, cwd=project_root, check=True)
    else:
        log_path = trial_output_dir / "trial_eval.log"
        with open(log_path, "a") as log_file:
            log_file.write(f"\n$ {shlex.join(command)}\n")
            log_file.flush()
            subprocess.run(
                command,
                cwd=project_root,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

    if not eval_metrics_path.exists():
        raise FileNotFoundError(f"Missing evaluation metrics: {eval_metrics_path}")

    with open(eval_metrics_path, "r") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for HAM10000 skin lesion classifier"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Base training config path",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root where src/train.py exists",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for study artifacts (default: outputs/optuna_<timestamp>)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="ham10000_optuna",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URL, e.g. sqlite:///outputs/optuna/study.db. "
            "If omitted, in-memory storage is used for this run."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of trials",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Stop study after this many seconds",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=42,
        help="Seed for Optuna sampler reproducibility",
    )
    parser.add_argument(
        "--objective-metric",
        choices=[
            "macro_recall",
            "macro_f1",
            "weighted_recall",
            "weighted_f1",
            "macro_recall_f1_mean",
        ],
        default="macro_recall_f1_mean",
        help=(
            "Metric derived from evaluation metrics of each trial checkpoint. "
            "Default optimizes the mean of macro recall and macro F1."
        ),
    )
    parser.add_argument(
        "--verbose-subprocess",
        action="store_true",
        help=(
            "Stream train.py/evaluate.py logs live to console "
            "instead of writing only per-trial log files"
        ),
    )
    parser.add_argument(
        "--show-progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show Optuna trial-level progress bar",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    base_config_path = args.config.resolve()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = project_root / "outputs" / f"optuna_{timestamp}"
    else:
        output_root = args.output.resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    trial_configs_dir = output_root / "trial_configs"
    trial_configs_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_yaml(base_config_path)
    base_config = ensure_existing_data_paths(base_config, project_root)

    direction = "maximize"
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    print(f"Study name: {study.study_name}")
    print(f"Output root: {output_root}")
    print(f"Objective: {args.objective_metric} ({direction})")
    print(f"Trials: {args.n_trials}")

    def objective(trial: optuna.Trial) -> float:
        trial_output_dir = output_root / f"trial_{trial.number:05d}"
        trial_config_path = trial_configs_dir / f"trial_{trial.number:05d}.yaml"

        trial_config = json.loads(json.dumps(base_config))
        trial_config = apply_ham10000_search_space(trial_config, trial)
        write_yaml(trial_config_path, trial_config)

        run_trial_training(
            project_root=project_root,
            trial_config_path=trial_config_path,
            trial_output_dir=trial_output_dir,
            verbose_subprocess=args.verbose_subprocess,
        )

        eval_metrics = run_trial_evaluation(
            project_root=project_root,
            trial_output_dir=trial_output_dir,
            trial_config=trial_config,
            verbose_subprocess=args.verbose_subprocess,
        )

        score = resolve_objective(eval_metrics, args.objective_metric)

        for key in [
            "accuracy",
            "macro_recall",
            "macro_f1",
            "weighted_recall",
            "weighted_f1",
            "macro_precision",
            "weighted_precision",
        ]:
            if key in eval_metrics:
                trial.set_user_attr(key, float(eval_metrics[key]))
        trial.set_user_attr("trial_output_dir", str(trial_output_dir))
        trial.set_user_attr("trial_config", str(trial_config_path))
        trial.set_user_attr(
            "trial_eval_metrics",
            str(trial_output_dir / "optuna_eval" / "evaluation_metrics.json"),
        )

        return score

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=bool(args.show_progress_bar),
        catch=(
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
            RuntimeError,
        ),
    )

    try:
        best = study.best_trial
    except ValueError as exc:
        raise RuntimeError(
            "No successful Optuna trials were completed. "
            "Check trial logs under output_root/trial_*/trial_train.log"
        ) from exc
    print("\n=== Best Trial ===")
    print(f"Trial #{best.number}")
    print(f"Objective value: {best.value}")
    print("Parameters:")
    for key, value in sorted(best.params.items()):
        print(f"  {key}: {value}")

    summary = {
        "study_name": study.study_name,
        "objective_metric": args.objective_metric,
        "direction": direction,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len(study.trials),
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "output_root": str(output_root),
        "base_config": str(base_config_path),
    }
    with open(output_root / "optuna_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    print(f"\nSaved summary: {output_root / 'optuna_summary.json'}")
    print(
        "Tip: retrain best trial for full budget if tuning used shorter epochs. "
        "Use trial config in trial_configs/ as a starting point."
    )


if __name__ == "__main__":
    main()
