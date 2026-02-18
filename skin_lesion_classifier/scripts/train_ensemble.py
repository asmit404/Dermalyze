#!/usr/bin/env python3
"""
Train an Ensemble of 3 Models for Skin Lesion Classification.

This script trains 3 models with different random seeds to create a diverse
ensemble. Each model is trained independently with the same architecture
but different initialization and data shuffling.

Usage:
    python scripts/train_ensemble.py [--output OUTPUT_DIR] [--seeds SEED1 SEED2 SEED3]

Examples:
    python scripts/train_ensemble.py
    python scripts/train_ensemble.py --output outputs/my_ensemble
    python scripts/train_ensemble.py --seeds 42 7 123
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import yaml


def create_temp_config(base_config_path: Path, seed: int, fold_index: int) -> Path:
    """Create a temporary config file with modified seed and fold index."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify seed
    config['training']['seed'] = seed

    # Ensure k-fold config exists and set fold index for this model
    data_cfg = config.setdefault('data', {})
    data_cfg['use_stratified_group_kfold'] = True
    kfold_cfg = data_cfg.setdefault('kfold', {})
    kfold_cfg['fold_index'] = fold_index
    
    # Write to temporary file
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False, prefix=f'config_seed{seed}_'
    )
    yaml.dump(config, temp_file, default_flow_style=False)
    temp_file.close()
    
    return Path(temp_file.name)


def train_model(
    config_path: Path,
    output_dir: Path,
    model_name: str,
    seed: int,
    fold_index: int,
    model_idx: int,
    total_models: int
) -> bool:
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"Training Model {model_idx + 1}/{total_models}: {model_name}")
    print(f"{'='*60}")
    print(f"Seed: {seed}")
    print(f"Fold index: {fold_index}")
    print(f"Output: {output_dir}")
    print()
    
    # Create temporary config with modified seed/fold
    temp_config = create_temp_config(config_path, seed, fold_index)
    
    try:
        # Train the model
        print("Starting training...")
        result = subprocess.run(
            [
                sys.executable, 'src/train.py',
                '--config', str(temp_config),
                '--output', str(output_dir)
            ],
            check=True,
            capture_output=False
        )
        
        # Check if training was successful
        checkpoint = output_dir / 'checkpoint_best.pt'
        if checkpoint.exists():
            print(f"✓ Model {model_name} trained successfully!")
            print(f"  Best checkpoint: {checkpoint}")
            return True
        else:
            print(f"✗ Model {model_name} training failed! No checkpoint found.")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Model {model_name} training failed with error!")
        print(f"  Error: {e}")
        return False
    finally:
        # Clean up temp config
        temp_config.unlink(missing_ok=True)


def create_evaluation_script(base_dir: Path, model_dirs: List[Path]) -> Path:
    """Create a bash script for ensemble evaluation."""
    script_path = base_dir / 'evaluate_ensemble.sh'
    
    checkpoints = ' \\\n        '.join([
        f'"{d / "checkpoint_best.pt"}"' for d in model_dirs
    ])
    
    script_content = f"""#!/bin/bash
# Automatically generated ensemble evaluation script

python src/evaluate.py \\
    --checkpoint \\
        {checkpoints} \\
    --test-csv "{model_dirs[0] / 'test_split.csv'}" \\
    --images-dir data/HAM10000/images \\
    --output "{base_dir / 'evaluation_results'}" \\
    "$@"
"""
    
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    
    return script_path


def create_inference_script(base_dir: Path, model_dirs: List[Path]) -> Path:
    """Create a bash script for ensemble inference."""
    script_path = base_dir / 'predict_ensemble.sh'
    
    checkpoints = ' \\\n        '.join([
        f'"{d / "checkpoint_best.pt"}"' for d in model_dirs
    ])
    
    script_content = f"""#!/bin/bash
# Automatically generated ensemble inference script

if [ -z "$1" ]; then
    echo "Usage: $0 <image_path>"
    exit 1
fi

IMAGE_PATH="$1"

python src/inference.py \\
    --ensemble \\
    --checkpoint \\
        {checkpoints} \\
    --image "${{IMAGE_PATH}}"
"""
    
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description='Train an ensemble of 3 models for skin lesion classification'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Base output directory for ensemble (default: outputs/ensemble_TIMESTAMP)'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs=3,
        default=[7, 13, 21],
        help='Three random seeds for ensemble diversity (default: 7 13 21)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.yaml'),
        help='Base config file (default: config.yaml)'
    )
    parser.add_argument(
        '--start-fold',
        type=int,
        default=0,
        help='Starting fold index; models use consecutive folds (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = Path(f'outputs/ensemble_{timestamp}')
    
    # Model names
    model_names = ['model_1', 'model_2', 'model_3']

    # Read k-fold setup from config (required for fold-diverse ensemble)
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    data_cfg = base_config.get('data', {}) if isinstance(base_config, dict) else {}
    kfold_cfg = data_cfg.get('kfold', {}) if isinstance(data_cfg, dict) else {}
    n_splits = int(kfold_cfg.get('n_splits', 5) or 5)

    if n_splits < len(model_names):
        print(
            f"✗ data.kfold.n_splits ({n_splits}) must be >= number of models ({len(model_names)}) "
            "for unique fold assignment."
        )
        sys.exit(1)

    if args.start_fold < 0 or args.start_fold >= n_splits:
        print(f"✗ --start-fold must be in [0, {n_splits - 1}], got {args.start_fold}")
        sys.exit(1)

    fold_indices = [(args.start_fold + i) % n_splits for i in range(len(model_names))]
    if len(set(fold_indices)) != len(fold_indices):
        print("✗ Computed fold indices are not unique. Increase n_splits or adjust start fold.")
        sys.exit(1)
    
    print("="*60)
    print("Training Ensemble of 3 Models")
    print("="*60)
    print()
    print(f"Base output directory: {args.output}")
    print(f"Random seeds: {args.seeds}")
    print(f"Fold indices: {fold_indices} (n_splits={n_splits})")
    print()
    
    # Create base output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Save ensemble metadata
    metadata = {
        'ensemble_name': args.output.name,
        'created_at': datetime.now().isoformat(),
        'num_models': 3,
        'seeds': args.seeds,
        'fold_indices': fold_indices,
        'kfold_n_splits': n_splits,
        'models': [
            {
                'name': name,
                'seed': seed,
                'fold_index': fold_index,
                'output_dir': str(args.output / name)
            }
            for name, seed, fold_index in zip(model_names, args.seeds, fold_indices)
        ]
    }
    
    metadata_file = args.output / 'ensemble_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created ensemble metadata: {metadata_file}")
    print()
    
    # Train each model
    model_dirs = []
    for i, (name, seed, fold_index) in enumerate(zip(model_names, args.seeds, fold_indices)):
        output_dir = args.output / name
        model_dirs.append(output_dir)
        
        success = train_model(
            config_path=args.config,
            output_dir=output_dir,
            model_name=name,
            seed=seed,
            fold_index=fold_index,
            model_idx=i,
            total_models=3
        )
        
        if not success:
            print(f"\n✗ Training failed for model {name}")
            sys.exit(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("Ensemble Training Complete!")
    print(f"{'='*60}")
    print()
    print("✓ All 3 models trained successfully!")
    print()
    print(f"Ensemble directory: {args.output}")
    print()
    print("Trained models:")
    for i, (name, seed, fold_index, model_dir) in enumerate(zip(model_names, args.seeds, fold_indices, model_dirs)):
        checkpoint = model_dir / 'checkpoint_best.pt'
        print(f"  {i + 1}. {name} (seed={seed}, fold_index={fold_index})")
        print(f"     {checkpoint}")
    print()
    
    # Create convenience scripts
    eval_script = create_evaluation_script(args.output, model_dirs)
    print(f"✓ Created evaluation script: {eval_script}")
    
    inference_script = create_inference_script(args.output, model_dirs)
    print(f"✓ Created inference script: {inference_script}")
    print()
    
    # Usage instructions
    print("To evaluate the ensemble:")
    print(f"  bash {eval_script}")
    print()
    print("To evaluate with TTA:")
    print(f"  bash {eval_script} --use-tta --tta-mode medium")
    print()
    print("To predict on an image:")
    print(f"  bash {inference_script} path/to/image.jpg")
    print()
    
    print(f"{'='*60}")
    print("Ensemble Ready!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
