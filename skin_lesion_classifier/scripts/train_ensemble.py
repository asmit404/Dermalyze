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


def create_temp_config(base_config_path: Path, seed: int) -> Path:
    """Create a temporary config file with modified seed."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify seed
    config['training']['seed'] = seed
    
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
    model_idx: int,
    total_models: int
) -> bool:
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"Training Model {model_idx + 1}/{total_models}: {model_name}")
    print(f"{'='*60}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}")
    print()
    
    # Create temporary config with modified seed
    temp_config = create_temp_config(config_path, seed)
    
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
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = Path(f'outputs/ensemble_{timestamp}')
    
    # Model names
    model_names = ['model_1', 'model_2', 'model_3']
    
    print("="*60)
    print("Training Ensemble of 3 Models")
    print("="*60)
    print()
    print(f"Base output directory: {args.output}")
    print(f"Random seeds: {args.seeds}")
    print()
    
    # Create base output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Save ensemble metadata
    metadata = {
        'ensemble_name': args.output.name,
        'created_at': datetime.now().isoformat(),
        'num_models': 3,
        'seeds': args.seeds,
        'models': [
            {
                'name': name,
                'seed': seed,
                'output_dir': str(args.output / name)
            }
            for name, seed in zip(model_names, args.seeds)
        ]
    }
    
    metadata_file = args.output / 'ensemble_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created ensemble metadata: {metadata_file}")
    print()
    
    # Train each model
    model_dirs = []
    for i, (name, seed) in enumerate(zip(model_names, args.seeds)):
        output_dir = args.output / name
        model_dirs.append(output_dir)
        
        success = train_model(
            config_path=args.config,
            output_dir=output_dir,
            model_name=name,
            seed=seed,
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
    for i, (name, seed, model_dir) in enumerate(zip(model_names, args.seeds, model_dirs)):
        checkpoint = model_dir / 'checkpoint_best.pt'
        print(f"  {i + 1}. {name} (seed={seed})")
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
