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
- **Default:** `data/HAM10000/labels.csv`
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
- **Default:** `false`
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

### `model.pretrained`
- **Type:** bool
- **Default:** `true`
- **Description:** Use ImageNet pretrained EfficientNet-B0 backbone.

### `model.dropout_rate`
- **Type:** float
- **Default:** `0.3`
- **Description:** Dropout used in classifier head.

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
- **Default:** `32`

### `training.epochs`
- **Type:** int
- **Default:** `30`

### `training.lr`
- **Type:** float
- **Default:** `1e-4`

### `training.weight_decay`
- **Type:** float
- **Default:** `0.01`

### Two-stage fine-tuning

### `training.stage1_epochs`
- **Type:** int
- **Default:** `0`
- **Description:** Head-warmup stage length.

### `training.stage2_epochs`
- **Type:** int or null
- **Default:** derived from `epochs` and mode
- **Description:** Explicit stage-2 length (used in `explicit` mode).

### `training.stage_epoch_mode`
- **Type:** string
- **Default:** `fit_total`
- **Valid options:**
  - `fit_total`: total epochs fixed by `training.epochs`; `stage2_epochs` can be ignored if inconsistent.
  - `explicit`: total epochs = `stage1_epochs + stage2_epochs`.

### `training.stage1_lr`, `training.stage2_lr`
- **Type:** float
- **Default:** fallback to `training.lr`

### `training.stage1_weight_decay`, `training.stage2_weight_decay`
- **Type:** float
- **Default:** fallback to `training.weight_decay`

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
- **Default:** `1.0`
- **Description:** Inverse-frequency sampling power (`0`=no weighting effect, `1`=full).

### `training.augmentation`
- **Type:** string
- **Default:** `medium`
- **Valid options:** `light`, `medium`, `heavy`, `domain`, `randaugment`

### Performance

### `training.use_amp`
- **Type:** bool
- **Default:** `true`
- **Description:** AMP is only active on CUDA.

### `training.use_gradient_checkpointing`
- **Type:** bool
- **Default:** `false`

### `training.gradient_accumulation_steps`
- **Type:** int
- **Default:** `1`

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
- **Default:** `0.5`
- **Description:** Conditional probability of CutMix vs Mixup when mix branch is used.

### Scheduler

### `training.scheduler.type`
- **Type:** string
- **Default:** `cosine`
- **Valid options:** `cosine`, `onecycle`

#### If `type: cosine`
- `training.scheduler.T_0` (int, default `10`)
- `training.scheduler.T_mult` (int, default `2`)
- `training.scheduler.eta_min` (float, default `1e-6`)

#### If `type: onecycle`
- `training.scheduler.warmup_pct` (float, default `0.1`)

### EMA

### `training.ema.enabled`
- **Type:** bool
- **Default:** `false`

### `training.ema.decay`
- **Type:** float
- **Default:** `0.999`

### `training.ema.use_for_eval`
- **Type:** bool
- **Default:** `true`

### `training.ema.save_best`
- **Type:** bool
- **Default:** `true`

### Early stopping

### `training.early_stopping_patience`
- **Type:** int
- **Default:** `15`

---

## `loss`

### `loss.type`
- **Type:** string
- **Default:** `focal`
- **Valid options:** `cross_entropy`, `focal`, `label_smoothing`

### `loss.class_weight_power`
- **Type:** float
- **Default:** `1.0`
- **Description:** Power for class-weight computation from training frequencies.

### `loss.alpha`
- **Type:** null or list[float]
- **Default:** null
- **Description:** Manual class weights. If null, computed class weights are used.
- **Notes:** When `loss.type == focal`, used as `focal_alpha`; otherwise used as class weights.

### `loss.label_smoothing`
- **Type:** float
- **Default:** `0.1`
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
- **Default:** `medium`
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

---
