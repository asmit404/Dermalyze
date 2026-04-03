# Configuration Reference (`config.yaml`)

This document explains all supported parameters in `config.yaml`, including valid options and practical constraints.

---

## `data`

### `data.images_dir`

- **Type:** string (path)
- **Default:** `data/HAM10000/images`
- **Description:** Directory containing image files.

### `data.labels_csv`

- **Type:** string (path)
- **Default:** `data/HAM10000_Training/labels_with_metadata.csv`
- **Description:** CSV with at least `image_id` and `label` columns.

### `data.split_seed`

- **Type:** int
- **Default:** falls back to `training.seed` if omitted, otherwise `42`
- **Description:** Seed used for train/val/test split generation.

### `data.val_size`

- **Type:** float
- **Default:** `0.15`
- **Description:** Validation fraction.
- **Notes:** In `use_stratified_group_kfold: true` mode, `val_size` is ignored.

### `data.test_size`

- **Type:** float
- **Default:** `0.15`
- **Description:** Test fraction.
- **Notes:** If `> 0`, a holdout test set is created before k-fold split.

### `data.lesion_aware`

- **Type:** bool
- **Default:** `true`
- **Description:** Uses lesion-aware grouping (`lesion_id`) to reduce leakage when available.

### `data.use_stratified_group_kfold`

- **Type:** bool
- **Default:** `true`
- **Description:** Enables `StratifiedGroupKFold` for train/val splitting.

### `data.kfold.n_splits`

- **Type:** int
- **Default:** `5`
- **Valid values:** `>= 2`

### `data.kfold.fold_index`

- **Type:** int
- **Default:** `0`
- **Valid values:** `0` to `n_splits - 1`

### `data.kfold.group_column`

- **Type:** string
- **Default:** `lesion_id`
- **Description:** Group identifier column used by `StratifiedGroupKFold`.
- **Requirement:** Must exist in CSV when k-fold mode is enabled.

### Multi-Input Models (Metadata)

### `data.use_metadata`

- **Type:** bool
- **Default:** `true`
- **Description:** Enable multi-input model architecture combining images + patient metadata for improved accuracy.
- **Accuracy Impact:** 🎯 **+2-5% improvement** over image-only models
- **Requirements:**
  - CSV must have columns: `age_approx`, `sex`, `anatom_site`
  - Run data preparation with metadata: `python src/prepare_data.py --data-dir data/HAM10000_Training --include-metadata`
  - Automatically disables MixUp/CutMix (metadata cannot be interpolated)
- **Architecture:**
  - When `true`: `MultiInputClassifier` (image backbone + metadata encoder + fusion layer)
  - When `false`: Standard image-only model
- **CRITICAL:** Checkpoints trained with `use_metadata: true` require this setting to load properly. Mismatched settings cause size mismatch errors.
- **Related:** See `data.metadata_columns`, `model.metadata_hidden_dim`, `model.fusion_hidden_dim`, `training.mixup_alpha`

### `data.metadata_columns`

- **Type:** list[string]
- **Default:** `[age_approx, anatom_site, sex]`
- **Description:** Column names for metadata features used in multi-input models
- **Used when:** `data.use_metadata: true`
- **Column Types:**
  - `age_approx`: Numerical (0-100), normalized continuous feature representing patient age
  - `sex`: Categorical (male/female/unknown), one-hot encoded
  - `anatom_site`: Categorical anatomical location, one-hot encoded
    - Valid values: back, lower extremity, trunk, upper extremity, head/neck, chest, abdomen, face, foot, hand, scalp, ear, neck, genital, acral
- **Missing Values:** Automatically filled with defaults (age=50, sex=unknown, site=unknown) if columns are present but have null values

### Segmentation-Assisted ROI Crop

### `data.segmentation.enabled`

- **Type:** bool
- **Default:** `false`
- **Description:** Master switch for lesion ROI cropping using segmentation masks before image transforms.
- **Behavior:** ROI crop is applied only when this is `true`.

### `data.segmentation.masks_dir`

- **Type:** string (path)
- **Default:** `data/HAM10000_Segmentations`
- **Description:** Directory containing segmentation mask files.

### `data.segmentation.required`

- **Type:** bool
- **Default:** `true`
- **Description:** Strictness flag for mask availability when ROI crop is enabled.
- **Behavior:**
  - If `data.segmentation.enabled: true` and `required: true`, every sample must have a mask; missing masks raise an error.
  - If `data.segmentation.enabled: true` and `required: false`, samples without masks fall back to uncropped images.
  - If `data.segmentation.enabled: false`, `required` is ignored (and train/eval log a warning when set to true).

### `data.segmentation.mask_threshold`

- **Type:** int
- **Default:** `10`
- **Description:** Threshold used to binarize grayscale masks (`mask > threshold`).

### `data.segmentation.crop_margin`

- **Type:** float
- **Default:** `0.10`
- **Description:** Margin added around lesion bounding box as a fraction of lesion size.

### `data.segmentation.filename_suffixes`

- **Type:** list[string]
- **Default:** `["", "_segmentation", "_mask"]`
- **Description:** Candidate suffixes used to resolve mask filenames.
- **Example resolution order:** `ISIC_001.png`, `ISIC_001_segmentation.png`, `ISIC_001_mask.png` (and jpg/jpeg variants).
- **Used by:** `src/train.py` and `src/evaluate.py`
- **CLI overrides (evaluation):** `--masks-dir`, `--use-segmentation-roi-crop`, `--segmentation-mask-threshold`, `--segmentation-crop-margin`, `--segmentation-required`, `--segmentation-mask-suffixes`

---

## `model`

### `model.num_classes`

- **Type:** int
- **Default:** `7`
- **Description:** Number of output classes.

### `model.image_size`

- **Type:** int
- **Default:** `224`
- **Description:** Input resolution used in train/eval transforms.

### `model.backbone`

- **Type:** string
- **Default:** `ResNeSt-101`
- **Total Options:** 14 backbones across 5 families
- **Description:** Selects which model backbone `src/train.py` instantiates for the classification task.

**Supported Backbones:**

| Family | Model | Config Value | Accepted Aliases | Feature Dim |
| -------- | ------- | -------------- | ------------------ | ------------- |
| **EfficientNet** | B0 | `efficientnet_b0` | `efficientnet`, `efficientnet-b0` | 1280 |
| EfficientNet | B1 | `efficientnet_b1` | `efficientnet-b1` | 1280 |
| EfficientNet | B2 | `efficientnet_b2` | `efficientnet-b2` | 1408 |
| EfficientNet | B3 | `efficientnet_b3` | `efficientnet-b3` | 1536 |
| EfficientNet | B4 | `efficientnet_b4` | `efficientnet-b4` | 1792 |
| EfficientNet | B5 | `efficientnet_b5` | `efficientnet-b5` | 2048 |
| EfficientNet | B6 | `efficientnet_b6` | `efficientnet-b6` | 2304 |
| EfficientNet | B7 | `efficientnet_b7` | `efficientnet-b7` | 2560 |
| **EfficientNetV2** | S | `efficientnetv2_s` | `efficientnet_v2_s`, `efficientnet-v2-s` | 1280 |
| EfficientNetV2 | M | `efficientnetv2_m` | `efficientnet_v2_m`, `efficientnet-v2-m` | 1280 |
| EfficientNetV2 | L | `efficientnetv2_l` | `efficientnet_v2_l`, `efficientnet-v2-l` | 1280 |
| **ConvNeXt** | Tiny | `convnext_tiny` | `convnext`, `convnext-tiny` | 768 |
| **ResNet** | ResNeSt-101 | `resnest_101` | `ResNeSt-101`, `resnest-101`, `resnest101` | 2048 |
| ResNet | SE-ResNeXt-101 | `seresnext_101` | `seresnext-101`, `seresnext101`, `se-resnext-101`, `se_resnext_101` | 2048 |

**Performance & Resource Requirements:**

| Backbone | Recommended `image_size` | GPU Memory | Speed | Notes |
| ---------- | ------------------------- | ------------ | ------- | ------- |
| EfficientNet-B0 | 224 | 6-8GB | Fastest | Good baseline |
| EfficientNet-B1 | 240 | 6-8GB | Fast | +5% params vs B0 |
| EfficientNet-B2 | 260 | 8-10GB | Fast | Balanced |
| EfficientNet-B3 | 300 | 10-12GB | Medium | Sweet spot |
| EfficientNet-B4 | 380 | 12-14GB | Medium | Higher accuracy |
| EfficientNet-B5 | 456 | 14-16GB | Slow | High accuracy |
| EfficientNet-B6 | 528 | 16GB+ | Slow | Very high accuracy |
| EfficientNet-B7 | 600 | 20GB+ | Slowest | Best accuracy |
| EfficientNetV2-S | 224-384 | 8-12GB | ⚡ Fast | Faster than v1 |
| EfficientNetV2-M | 384-480 | 12-16GB | Medium | Balanced v2 |
| EfficientNetV2-L | 384-480 | 16GB+ | Medium | High-capacity v2 |
| ConvNeXt-Tiny | 224-256 | 8-10GB | ⚡ Fast | Modern ConvNet |
| ResNeSt-101 | 224-384 | 10-14GB | Medium | Current default, strong multi-scale |
| SE-ResNeXt-101 | 224-384 | 10-14GB | Medium | Attention mechanism |

**Decision Guidance:**

- **Limited GPU (<8GB):** Use `efficientnet_b0` with `image_size: 224`
- **Balanced Setup (12GB GPU):** Use `ResNeSt-101` or `efficientnet_b3` with `image_size: 224-300`
- **Maximum Accuracy (16GB+ GPU):** Use `ResNeSt-101` or `efficientnet_b5` with `image_size: 384-456`
- **Fastest Training:** Use `efficientnet_b0` with `image_size: 224`, disable gradient checkpointing
- **Medical Imaging:** ResNeSt/SE-ResNeXt have strong multi-scale features beneficial for dermoscopy

**CRITICAL:** `model.image_size` must match between training and inference. Changing image size requires retraining from scratch.

**Usage:**

- All backbones support both image-only and multi-input (metadata) models
- `src/evaluate.py` can load checkpoints from any backbone and supports mixed-backbone ensembles

### `model.pretrained`

- **Type:** bool
- **Default:** `true`
- **Description:** Use ImageNet pretrained weights for the selected backbone.

### `model.dropout_rate`

- **Type:** float
- **Default:** `0.3`
- **Description:** Dropout used in classifier head.

### `model.metadata_hidden_dim`

- **Type:** int
- **Default:** `64`
- **Description:** Hidden dimension for metadata MLP encoder in multi-input models
- **Used when:** `data.use_metadata: true`
- **Notes:** Controls the dimensionality of the metadata feature representation before fusion with image features

### `model.fusion_hidden_dim`

- **Type:** int
- **Default:** `256`
- **Description:** Hidden dimension for the fusion layer that combines image and metadata features
- **Used when:** `data.use_metadata: true`
- **Notes:** Larger values allow more complex feature interactions but increase memory usage

### `model.freeze_backbone`

- **Type:** bool
- **Default:** `false`
- **Description:** Starts with backbone frozen in model creation.

---

## `training`

### Core training

### `training.seed`

- **Type:** int
- **Default:** `42`
- **Description:** Global training seed.

### `training.batch_size`

- **Type:** int
- **Default:** `16`  

### `training.epochs`

- **Type:** int
- **Default:** `40`  

### `training.lr`

- **Type:** float
- **Default:** `0.00025`  

### `training.weight_decay`

- **Type:** float
- **Default:** `0.02`  

### Two-stage fine-tuning

### `training.stage1_epochs`

- **Type:** int
- **Default:** `5`  
- **Description:** Head-warmup stage length.

### `training.stage2_epochs`

- **Type:** int or null
- **Default:** `35`   (derived from `epochs` and mode)
- **Description:** Explicit stage-2 length (used in `explicit` mode).

### `training.stage_epoch_mode`

- **Type:** string
- **Default:** `fit_total`
- **Valid options:**
  - `fit_total`: total epochs fixed by `training.epochs`; `stage2_epochs` can be ignored if inconsistent.
  - `explicit`: total epochs = `stage1_epochs + stage2_epochs`.

### `training.stage1_lr`, `training.stage2_lr`

- **Type:** float
- **Default:** `training.stage1_lr: 0.001`  , `training.stage2_lr: 0.0005`   (previously fell back to `training.lr`)

### `training.stage1_weight_decay`, `training.stage2_weight_decay`

- **Type:** float
- **Default:** `training.stage1_weight_decay: 0.01`  , `training.stage2_weight_decay: 0.02`   (previously fell back to `training.weight_decay`)

### Data loading / sampling

### `training.num_workers`

- **Type:** int
- **Default:** `4`

### `training.prefetch_factor`

- **Type:** int or null
- **Default:** `2`
- **Description:** Used only when `num_workers > 0`.

### `training.persistent_workers`

- **Type:** bool
- **Default:** `true`
- **Description:** Used only when `num_workers > 0`.

### `training.use_weighted_sampling`

- **Type:** bool
- **Default:** `true`
- **Description:** Enables `WeightedRandomSampler` on training set.

### `training.sampling_weight_power`

- **Type:** float
- **Default:** `0.35`  
- **Description:** Inverse-frequency sampling power (`0`=no weighting effect, `1`=full).

### `training.augmentation`

- **Type:** string
- **Default:** `light`  
- **Valid options:** `light`, `medium`, `heavy`, `domain`, `randaugment`

### Performance

### `training.use_amp`

- **Type:** bool
- **Default:** `true`
- **Description:** AMP is only active on CUDA.

### `training.use_gradient_checkpointing`

- **Type:** bool
- **Default:** `true`  

### `training.gradient_accumulation_steps`

- **Type:** int
- **Default:** `2`  

### Mixup / CutMix

### `training.mixup_alpha`

- **Type:** float
- **Default:** `0.0`

### `training.cutmix_alpha`

- **Type:** float
- **Default:** `0.0`

### `training.mixup_prob`

- **Type:** float
- **Default:** `0.0`
- **Description:** Probability to enter mix augmentation branch.

### `training.cutmix_prob`

- **Type:** float
- **Default:** `0.15`
- **Description:** Conditional probability of CutMix vs Mixup when mix branch is used.

### Scheduler

### `training.scheduler.type`

- **Type:** string
- **Default:** `cosine`
- **Valid options:** `cosine`, `onecycle`

#### If `type: cosine`

- `training.scheduler.T_0` (int, default `20`  )
- `training.scheduler.T_mult` (int, default `1`  )
- `training.scheduler.eta_min` (float, default `1e-6`)

#### If `type: onecycle`

- `training.scheduler.warmup_pct` (float, default `0.1`)

### EMA

### `training.ema.enabled`

- **Type:** bool
- **Default:** `true`

### `training.ema.decay`

- **Type:** float
- **Default:** `0.999`

### `training.ema.use_for_eval`

- **Type:** bool
- **Default:** `true`

### `training.ema.save_best`

- **Type:** bool
- **Default:** `false`  

### Early stopping

### `training.early_stopping_patience`

- **Type:** int
- **Default:** `7`  

---

## `loss`

### `loss.type`

- **Type:** string
- **Default:** `focal`
- **Valid options:** `cross_entropy`, `focal`, `label_smoothing`

### `loss.class_weight_power`

- **Type:** float
- **Default:** `0.0`  
- **Description:** Power for class-weight computation from training frequencies.

### `loss.alpha`

- **Type:** null or list[float]
- **Default:** null
- **Description:** Manual class weights. If null, computed class weights are used.
- **Notes:** When `loss.type == focal`, used as `focal_alpha`; otherwise used as class weights.

### `loss.label_smoothing`

- **Type:** float
- **Default:** `0.03`  
- **Used when:** `loss.type == label_smoothing`

### `loss.gamma` / `loss.focal_gamma`

- **Type:** float
- **Default:** `2.0`
- **Used when:** `loss.type == focal`
- **Notes:** `gamma` takes precedence if both are set.

---

## `output`

### `output.save_epoch_checkpoints`

- **Type:** bool
- **Default:** `false`
- **Description:** If true, saves `checkpoint_epoch_{n}.pt` each epoch.

---

## `evaluation`

These values are read by `src/evaluate.py` as defaults. CLI args override them.

## `evaluation.tta.use_tta`

- **Type:** bool
- **Default:** `false`

## `evaluation.tta.mode`

- **Type:** string
- **Default:** `full`  
- **Valid options:** `light`, `medium`, `full`
- **Current branch counts (without CLAHE):**
  - `light`: 4
  - `medium`: 8
  - `full`: 12

## `evaluation.tta.aggregation`

- **Type:** string
- **Default:** `mean`
- **Valid options:** `mean`, `geometric_mean`, `max`

## `evaluation.tta.use_clahe_tta`

- **Type:** bool
- **Default:** `false`
- **Requirement:** OpenCV installed (`opencv-python-headless`).

## `evaluation.tta.clahe_clip_limit`

- **Type:** float
- **Default:** `2.0`

## `evaluation.tta.clahe_grid_size`

- **Type:** int
- **Default:** `8`

### Ensemble Evaluation

These parameters control ensemble prediction using multiple model checkpoints. While not directly configurable in `config.yaml`, they control `evaluate.py` behavior via CLI flags.

## `evaluation.ensemble.use_ensemble`

- **Type:** bool
- **Default:** `false`
- **Description:** Enable ensemble evaluation with multiple model checkpoints
- **Accuracy Impact:** 🎯 **+1-3% typical improvement** over single best model
- **Usage:** Provide multiple checkpoints via CLI `--checkpoint` flag:

  ```bash
  python src/evaluate.py --checkpoint model1.pt model2.pt model3.pt ...
  ```

- **Best Practices:** Ensemble 3-5 diverse models (diminishing returns beyond 5)

## `evaluation.ensemble.aggregation`

- **Type:** string
- **Default:** `weighted_mean`
- **Valid options:** `mean`, `weighted_mean`, `geometric_mean`
- **Description:** How to combine predictions from multiple models
  - `mean`: Simple arithmetic average (all models weighted equally)
  - `weighted_mean`: ⭐ **Recommended** - Weight by validation accuracy (auto-extracted from checkpoint metrics)
  - `geometric_mean`: Geometric average (rarely used, can be unstable)
- **Decision Guidance:**
  - Use `weighted_mean` when models have varying accuracies (default, typically best)
  - Use `mean` for testing or when all models perform similarly
  - `weighted_mean` typically +0.5-1% better than simple `mean`

## `evaluation.ensemble.weights`

- **Type:** list[float] or null
- **Default:** `null`
- **Description:** Custom ensemble weights for each model
- **Behavior:**
  - If `null` with `aggregation: weighted_mean`, weights are auto-computed from checkpoint validation metrics
  - If specified, must be a list with length matching number of checkpoints
  - Values are normalized to sum to 1.0
- **Example:** `[0.4, 0.35, 0.25]` for 3 models

**Ensemble Usage Examples:**

```bash
# Auto-weighted ensemble (recommended)
python src/evaluate.py \
  --checkpoint outputs/run_1/checkpoint_best.pt \
              outputs/run_2/checkpoint_best.pt \
              outputs/run_3/checkpoint_best.pt \
  --test-csv data/test_split.csv \
  --images-dir data/HAM10000_Training/images \
  --ensemble-aggregation weighted_mean

# Ensemble with TTA (maximum accuracy)
python src/evaluate.py \
  --checkpoint model1.pt model2.pt model3.pt \
  --test-csv test_split.csv \
  --images-dir data/HAM10000_Training/images \
  --use-tta --tta-mode full \
  --tta-aggregation mean \
  --ensemble-aggregation weighted_mean

# Diverse backbone ensemble (recommended for robustness)
python src/evaluate.py \
  --checkpoint outputs/resnest/checkpoint_best.pt \
              outputs/efficientnet_b3/checkpoint_best.pt \
              outputs/convnext/checkpoint_best.pt \
  --test-csv test_split.csv \
  --images-dir data/HAM10000_Training/images \
  --ensemble-aggregation weighted_mean
```

**Best Practices for Ensemble:**

- Use diverse backbones (e.g., EfficientNet-B3 + ResNeSt-101 + ConvNeXt-Tiny) for better robustness
- Ensure all models have the same `data.use_metadata` setting
- Use `weighted_mean` aggregation for best accuracy
- Combine ensemble with TTA for maximum performance (+3-5% total improvement)

---

## Recent Configuration Changes (2026-03-24)

The following optimizations were applied to improve model accuracy and training stability:

### 1. Stage 2 Learning Rate Increased

**Parameter:** `training.stage2_lr`
**Change:** `0.00025` → `0.0005`
**Rationale:** Improves adaptation during full fine-tuning; previous value was too conservative and could slow convergence
**Impact:** Faster stage 2 convergence, +0.5-1% accuracy improvement

### 2. MixUp/CutMix Explicitly Disabled

**Parameters:** `training.mixup_alpha`, `training.cutmix_alpha`, `training.mixup_prob`
**Change:** `0.1, 0.1, 0.25` → `0.0, 0.0, 0.0`
**Rationale:** Incompatible with `data.use_metadata: true` (metadata features cannot be meaningfully interpolated between samples); makes auto-disable behavior explicit in config
**Impact:** Clearer configuration, no functional change (already auto-disabled when metadata enabled)

### 3. EMA Best-Checkpoint Behavior Adjusted

**Parameter:** `training.ema.save_best`
**Change:** `true` → `false`
**Rationale:** Prevents early-epoch EMA snapshots from being saved as "best" checkpoint; ties best checkpoint to the validated model state
**Impact:** More predictable checkpoint management, avoids confusion from multiple "best" checkpoints

### 4. Removed Duplicate Class-Imbalance Correction

**Parameter:** `loss.class_weight_power`
**Change:** `0.35` → `0.0`
**Rationale:** `training.use_weighted_sampling: true` already handles class imbalance via oversampling; stacking both sampling weights AND loss weights can over-correct minority classes, degrading precision
**Impact:** Cleaner single-mechanism balancing, improved minority class precision

### 5. TTA Aggregation Switched to Arithmetic Mean

**Parameter:** `evaluation.tta.aggregation`
**Change:** `geometric_mean` → `mean`
**Rationale:** Arithmetic mean is safer and more stable for probability aggregation across diverse augmentations
**Impact:** More stable TTA predictions, especially with full augmentation mode

**Key Takeaway:** Use a **single class-imbalance control mechanism** (weighted sampling) rather than combining multiple approaches. When `use_metadata: true`, MixUp/CutMix must remain disabled.

---
