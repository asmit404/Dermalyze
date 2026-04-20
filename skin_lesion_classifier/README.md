# Dermalyze Training Pipeline

Training, evaluation, and experimentation pipeline for 7-class skin lesion classification.

>  DISCLAIMER: Educational/research purposes only. Not for medical diagnosis.

## Scope

This module contains:

- data preparation (`src/prepare_data.py`)
- model training (`src/train.py`)
- evaluation and reporting (`src/evaluate.py`)
- local inference utilities (`src/inference.py`)
- sweep/tuning scripts (`scripts/`)

Deployment-facing API lives separately in [`../inference_service/README.md`](../inference_service/README.md).

## Supported Backbones

Training/evaluation support these backbone keys:

- EfficientNet: `efficientnet_b0` ... `efficientnet_b7`
- EfficientNetV2: `efficientnetv2_s`, `efficientnetv2_m`, `efficientnetv2_l`
- ConvNeXt: `convnext_tiny`
- ResNeSt: `resnest_101`
- SE-ResNeXt: `seresnext_101`

Aliases (for example `efficientnet`, `convnext`, `resnest-101`) are resolved in `src/train.py`.

## Current Default Configuration (config.yaml)

- `model.backbone: efficientnetv2_s` (set this to a supported key such as `resnest_101` before training)
- `model.image_size: 224`
- `data.use_metadata: true`
- `data.segmentation.enabled: true`
- `data.segmentation.required: true`
- `data.segmentation.masks_dir: data/HAM10000_Segmentations`
- `training.epochs: 40` with two-stage schedule (`stage1_epochs: 5`, `stage2_epochs: 35`)
- `training.use_weighted_sampling: true`
- `loss.type: focal`
- `loss.class_weight_power: 0.0`
- `evaluation.tta.use_tta: false`, `mode: full`, `aggregation: mean`, `use_clahe_tta: false`

See `config.md` for full field reference.

## Dataset Layout

Expected directory structure:

```text
data/
  HAM10000_Training/
    images/
    ground_truth.csv
    metadata.csv
  HAM10000_Segmentations/
    images/
      ISIC_XXXX_segmentation.png
  HAM10000_Val/
    images/
    ground_truth.csv
  HAM10000_Test/
    images/
    ground_truth.csv
```

`ground_truth.csv` must include one-hot class columns: `MEL,NV,BCC,AKIEC,BKL,DF,VASC`.

Segmentation directory is optional unless `data.segmentation.enabled: true`.
Mask lookup supports both `data.segmentation.masks_dir` and nested folders like `data.segmentation.masks_dir/images`.

## Setup

macOS/Linux:

```bash
cd skin_lesion_classifier
bash scripts/install_pytorch.sh
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
cd skin_lesion_classifier
.\scripts\install_pytorch.ps1
pip install -r requirements.txt
```

Optional channel override:

```bash
TORCH_CHANNEL=cu128 bash scripts/install_pytorch.sh
```

## Data Preparation

Training split with metadata:

```bash
python src/prepare_data.py --data-dir data/HAM10000_Training --include-metadata
```

Validation/test splits:

```bash
python src/prepare_data.py --data-dir data/HAM10000_Val
python src/prepare_data.py --data-dir data/HAM10000_Test
```

Useful flags:

- `--output <path>` custom output CSV
- `--skip-validation` skip per-image validation

Output files:

- with metadata: `labels_with_metadata.csv`
- without metadata: `labels.csv`

## Training

Basic run:

```bash
python src/train.py --config config.yaml
```

Enable segmentation ROI crop in `config.yaml`:

```yaml
data:
  segmentation:
    enabled: true
    masks_dir: data/HAM10000_Segmentations
    required: true
    mask_threshold: 10
    crop_margin: 0.10
    filename_suffixes: ["", "_segmentation", "_mask"]
```

Resume:

```bash
python src/train.py --config config.yaml --resume outputs/run_xxx/checkpoint_latest.pt
```

Custom output directory:

```bash
python src/train.py --config config.yaml --output outputs/my_run
```

Generated artifacts in each run directory:

- `checkpoint_latest.pt`
- `checkpoint_best.pt`
- `train_split.csv`, `val_split.csv`, `test_split.csv`
- `training_history.json`
- `training.log`
- copied `config.yaml`

## Evaluation

Single checkpoint:

```bash
python src/evaluate.py \
  --checkpoint outputs/run_xxx/checkpoint_best.pt \
  --test-csv outputs/run_xxx/test_split.csv \
  --images-dir data/HAM10000_Training/images
```

Ensemble:

```bash
python src/evaluate.py \
  --checkpoint model1.pt model2.pt model3.pt \
  --test-csv outputs/run_xxx/test_split.csv \
  --images-dir data/HAM10000_Training/images \
  --ensemble-aggregation weighted_mean
```

TTA override via CLI:

```bash
python src/evaluate.py \
  --checkpoint outputs/run_xxx/checkpoint_best.pt \
  --test-csv outputs/run_xxx/test_split.csv \
  --images-dir data/HAM10000_Training/images \
  --use-tta --tta-mode full --tta-aggregation mean --use-clahe-tta
```

Evaluation with segmentation ROI crop via CLI:

```bash
python src/evaluate.py \
  --checkpoint outputs/run_xxx/checkpoint_best.pt \
  --test-csv outputs/run_xxx/test_split.csv \
  --images-dir data/HAM10000_Training/images \
  --masks-dir data/HAM10000_Segmentations \
  --use-segmentation-roi-crop \
  --segmentation-required \
  --segmentation-mask-threshold 10 \
  --segmentation-crop-margin 0.10
```

Evaluation output directory (default: `evaluation_results`) includes:

- `evaluation_metrics.json`
- `predictions.csv`
- `confusion_matrix.png`
- `confusion_matrix_raw.png`
- `one_vs_rest_confusion_counts.png`
- `roc_curves.png`
- `per_class_metrics.png`
- `calibration_curve.png`

Note: `evaluate.py` supports unlabeled rows (for example `label=unknown`). Those rows are kept in `predictions.csv`, while aggregate metrics/plots are computed only from labeled rows.

## Local Inference Utility

Single-model CLI:

```bash
python src/inference.py --checkpoint outputs/run_xxx/checkpoint_best.pt --image path/to/image.jpg
```

Single-model with TTA:

```bash
python src/inference.py --checkpoint outputs/run_xxx/checkpoint_best.pt --image path/to/image.jpg --use-tta --tta-mode medium
```

Ensemble CLI:

```bash
python src/inference.py --ensemble model1.pt model2.pt model3.pt --image path/to/image.jpg
```

## Sweeps and Tuning

Optuna tuning:

```bash
python scripts/tune_optuna.py --config config.yaml --n-trials 40 --objective-metric macro_recall_f1_mean
```

Verbose subprocess logs during tuning:

```bash
python scripts/tune_optuna.py --config config.yaml --n-trials 5 --verbose-subprocess
```

Ensemble training helper:

```bash
python scripts/train_ensemble.py --config config.yaml
```

## Analysis Utilities

```bash
python scripts/visualize_training.py --run outputs/run_xxx
python scripts/check_fit.py outputs/run_xxx/training_history.json
python scripts/diagnose_generalization.py --run outputs/run_xxx --images-dir data/HAM10000_Training/images
python scripts/benchmark.py --config config.yaml --num-batches 20
```

`diagnose_generalization.py` writes a held-out generalization diagnosis report to
`outputs/run_xxx/generalization_eval/` (JSON + text), using evaluation metrics,
confusion patterns, calibration, and confidence behavior.

## Metadata and Imbalance Notes

- `data.use_metadata: true` enables multi-input fusion with metadata encoder.
- Required metadata roles are inferred from configured columns (age, sex, localization/anatom site).
- MixUp/CutMix are supported with metadata; metadata features are mixed using the same lambda as images.
- Class balancing is typically controlled via weighted sampling. Current default uses:
  - `training.use_weighted_sampling: true`
  - `loss.class_weight_power: 0.0`

## Export to Inference Service

After training:

```bash
cp outputs/run_xxx/checkpoint_best.pt ../inference_service/models/checkpoint_best.pt
```

The inference service also supports legacy fallback path `../inference_service/model/checkpoint_best.pt`.

## Related Modules

- Inference API: [`../inference_service/README.md`](../inference_service/README.md)
- Frontend app: [`../frontend/README.md`](../frontend/README.md)
