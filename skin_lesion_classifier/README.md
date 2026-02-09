# Dermalyze: Educational Skin Lesion Classification System

An educational skin lesion image classification system using EfficientNet-V2 and modern deep learning practices. This system classifies dermoscopic images into seven clinically relevant skin lesion categories from the HAM10000 dataset.

> ⚠️ **DISCLAIMER**: This system is for **educational and research purposes only**. It does not provide medical diagnosis or clinical decision-making. Always consult a qualified healthcare professional for medical advice.

## Overview

This project demonstrates how to build a modern, maintainable, and reproducible deep learning system for multi-class skin lesion classification. Key features include:

- **EfficientNet-V2 Small backbone** with transfer learning
- **Multiple classification heads**:
  - Simple: Multi-layer perceptron with BatchNorm and Dropout
  - ACRNN: Attention-based Convolutional Recurrent Neural Network
- **Advanced data augmentation**:
  - Mixup: Probabilistic blending of images and labels
  - CutMix: Spatial patch mixing with area-proportional labels
  - Domain-specific augmentations for dermatoscopy
- **Model optimization**:
  - ModelEMA (Exponential Moving Average) for better generalization
  - Two-stage training with separate learning rates
- **Robust inference strategies**:
  - **Test-Time Augmentation (TTA)**: Multiple augmented predictions with aggregation
  - **Ensemble Prediction**: Multi-model voting with uncertainty quantification
- **Lesion-aware data splitting** to prevent data leakage via `lesion_id` grouping
- **Class-imbalanced learning**:
  - Multiple loss functions (Cross-Entropy, Focal Loss, Label Smoothing)
  - Weighted random sampling for training
  - Class-weighted loss computation
- **Comprehensive evaluation** with per-class metrics, ROC curves, confusion matrix, and calibration analysis
- **Mixed precision training (AMP)** with automatic device selection (CUDA/MPS/CPU)
- **Advanced training features**:
  - Cosine annealing with warm restarts
  - Early stopping with configurable patience
  - Checkpoint saving and resuming
  - JSON-based training history logging
- **Flexible YAML configuration** system for reproducible experiments
- **Modular architecture** for easy extension and deployment
- **Utility scripts** for training visualization and fit analysis

## Supported Classes

The model classifies images into seven categories from the HAM10000 dataset:

| Label | Description |
|-------|-------------|
| akiec | Actinic keratoses and intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

## Project Structure

```
skin_lesion_classifier/
├── config.yaml              # Training configuration (YAML)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/
│   └── HAM10000/
│       ├── images/         # Dermoscopic images (JPG)
│       ├── HAM10000_metadata.csv  # Original metadata
│       └── labels.csv      # Processed labels (created by prepare_data.py)
├── outputs/                # Training outputs (auto-created)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── checkpoint_best.pt    # Best model weights
│       ├── checkpoint_latest.pt  # Latest checkpoint
│       ├── config.yaml           # Config snapshot
│       ├── training_history.json # Training metrics
│       ├── train_split.csv       # Training split
│       ├── val_split.csv         # Validation split
│       └── test_split.csv        # Test split
├── scripts/
│   ├── visualize_training.py     # Visualize training curves
│   ├── check_fit.py              # Analyze overfitting/underfitting
│   └── benchmark.py              # Performance profiling tool
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py      # Dataset, transforms, data loading
│   ├── models/
│   │   ├── __init__.py
│   │   ├── efficientnet.py # EfficientNet-V2 classifier
│   │   └── acrnn.py        # ACRNN classification head
│   ├── train.py            # Training script
│   ├── evaluate.py         # Comprehensive evaluation
│   ├── inference.py        # Inference API
│   └── prepare_data.py     # Data preparation & validation
└── notebooks/              # Jupyter notebooks (optional)
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd skin_lesion_classifier
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Performance Benchmarking

Measure your actual training performance:

```bash
python scripts/benchmark.py --num-batches 20
```

This comprehensive benchmark reports:
- Data loading speed
- Forward pass throughput
- Backward pass timing
- Training step breakdown
- Memory usage (allocated/reserved)
- Estimated epoch time

If you need exact reproducibility for research, set `fast_mode: false` in config (will be ~30% slower).

## Dataset Setup

1. **Download the HAM10000 dataset** from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

2. **Organize the data**:
```
data/HAM10000/
├── images/
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   └── ...
└── HAM10000_metadata.csv
```

3. **Prepare the dataset**:
```bash
python src/prepare_data.py --data-dir data/HAM10000
```

This will create `labels.csv` with the required format and validate the images.

### Basic Training

```bash
python src/train.py --config config.yaml
```

This will:
- Load configuration from `config.yaml`
- Create timestamped output directory in `outputs/`
- Split data (train/val/test) with lesion-aware stratification
- Train the model with automatic checkpointing
- Save best and latest checkpoints
- Log training history to `training_history.json`

### Training with Custom Output Directory

```bash
python src/train.py --config config.yaml --output outputs/experiment_1
```

### Resume Training from Checkpoint

```bash
python src/train.py --config config.yaml --resume outputs/run_xxx/checkpoint_latest.pt
```

### Training Output Files

After training, you'll find:
- `checkpoint_best.pt` - Model with best validation loss
- `checkpoint_latest.pt` - Most recent checkpoint (for resuming)
- `training_history.json` - Epoch-by-epoch metrics (loss, accuracy, learning rate)
- `config.yaml` - Snapshot of training configuration
- `train_split.csv`, `val_split.csv`, `test_split.csv` - Data splits for reproducibility

### Configuration Options

Key settings in `config.yaml`:

```yaml
model:
  name: efficientnet_v2    # EfficientNet-V2 Small
  pretrained: true         # Use ImageNet pretrained weights
  head_type: simple        # Options: simple, acrnn
  dropout_rate: 0.35       # Dropout for regularization
  freeze_backbone: false   # Freeze backbone during training

training:
  batch_size: 32           # Reduce if out of memory
  epochs: 40               # Maximum epochs
  lr: 0.0003               # Learning rate
  weight_decay: 0.02       # L2 regularization
  num_workers: 4           
  use_amp: false           # Disabled for MPS (no benefit)
  use_torch_compile: false # Disabled for MPS (backward pass issues)
  fast_mode: true          # Speed over determinism (~40% faster)
  use_weighted_sampling: true  # Balance class distribution
  early_stopping_patience: 6   # Stop if no improvement
  prefetch_factor: 2       # Prefetch batches for pipeline
  persistent_workers: true # Keep workers alive between epochs
  
  scheduler:
    type: cosine           # Cosine annealing with warm restarts
    T_0: 20                # Restart every 20 epochs
    eta_min: 0.00001       # Minimum learning rate
    warmup_pct: 0.05       # 5% warmup

loss:
  type: cross_entropy      # Options: cross_entropy, focal, label_smoothing
  label_smoothing: 0.05    # For label_smoothing loss

data:
  val_size: 0.15           # Validation split proportion
  test_size: 0.15          # Test split proportion
  lesion_aware:true        # Group by lesion_id to prevent leakage
```

### Analyzing Training Progress

Visualize training curves and detect overfitting:

```bash
# Generate comprehensive training visualizations
python scripts/visualize_training.py --run outputs/run_20260205_120000

# Quick fit analysis (overfitting/underfitting detector)
python scripts/check_fit.py outputs/run_20260205_120000/training_history.json

# Benchmark training performance (especially useful for optimization)
python scripts/benchmark.py --num-batches 50
```

### Training an Ensemble

An **ensemble** combines multiple independently trained models to improve accuracy and robustness. Train multiple models with diverse configurations:

#### Automated Ensemble Training (Recommended)

Use the provided scripts to automatically train 3 models with different random seeds:

**Python Script (Cross-platform):**
```bash
# Train ensemble with default seeds (7, 13, 21)
python scripts/train_ensemble.py

# Train ensemble with custom output directory
python scripts/train_ensemble.py --output outputs/my_ensemble

# Train ensemble with custom seeds
python scripts/train_ensemble.py --seeds 10 20 30
```

**Bash Script (Unix/macOS/Linux):**
```bash
# Train ensemble with auto-generated output directory
bash scripts/train_ensemble.sh

# Train ensemble with custom output directory
bash scripts/train_ensemble.sh outputs/my_ensemble
```

Both scripts will:
- Train 3 models with different random seeds for diversity
- Save all models in organized subdirectories
- Create evaluation and inference convenience scripts
- Generate ensemble metadata (JSON)

After training completes, the scripts create:
- `ensemble_metadata.json` - Ensemble configuration details
- `evaluate_ensemble.sh` - Ready-to-use evaluation script
- `predict_ensemble.sh` - Ready-to-use inference script

**Example output structure:**
```
outputs/ensemble_20260209_152530/
├── ensemble_metadata.json
├── evaluate_ensemble.sh
├── predict_ensemble.sh
├── model_1/
│   ├── checkpoint_best.pt
│   ├── config.yaml
│   └── ...
├── model_2/
│   ├── checkpoint_best.pt
│   └── ...
└── model_3/
    ├── checkpoint_best.pt
    └── ...
```

#### Manual Ensemble Training

Alternatively, train models manually with different configurations:

#### Strategy 1: Different Random Seeds

Train multiple models with the same architecture but different seeds:

```bash
# Model 1 (seed=7)
python src/train.py --config config.yaml --output outputs/ensemble_model_1

# Model 2 (seed=13) - edit config.yaml: training.seed = 13
python src/train.py --config config.yaml --output outputs/ensemble_model_2

# Model 3 (seed=21) - edit config.yaml: training.seed = 21
python src/train.py --config config.yaml --output outputs/ensemble_model_3
```

#### Strategy 2: Different Classification Heads

Train models with different head architectures:

```bash
# Simple head (default)
# Edit config.yaml: model.head_type = "simple"
python src/train.py --config config.yaml --output outputs/ensemble_simple

# ACRNN head (attention-based)
# Edit config.yaml: model.head_type = "acrnn"
python src/train.py --config config.yaml --output outputs/ensemble_acrnn
```

#### Strategy 3: Different Augmentation Levels

Train models with varying augmentation intensities:

```bash
# Light augmentation
# Edit config.yaml: training.augmentation = "light"
python src/train.py --config config.yaml --output outputs/ensemble_light_aug

# Heavy augmentation
# Edit config.yaml: training.augmentation = "heavy"
python src/train.py --config config.yaml --output outputs/ensemble_heavy_aug

# Domain augmentation
# Edit config.yaml: training.augmentation = "domain"
python src/train.py --config config.yaml --output outputs/ensemble_domain_aug
```

#### Strategy 4: Advanced Augmentation Techniques

Use Mixup, CutMix, or Model EMA for diversity:

```bash
# Model with Mixup
# Edit config.yaml: training.mixup.enabled = true, training.mixup.alpha = 1.0
python src/train.py --config config.yaml --output outputs/ensemble_mixup

# Model with CutMix
# Edit config.yaml: training.cutmix.enabled = true, training.cutmix.alpha = 1.0
python src/train.py --config config.yaml --output outputs/ensemble_cutmix

# Model with both (50/50 probability)
# Edit config.yaml: mixup.enabled = true, cutmix.enabled = true, mixup_prob = 0.5
python src/train.py --config config.yaml --output outputs/ensemble_mixed

# Model with EMA
# Edit config.yaml: training.ema.enabled = true, training.ema.decay = 0.9999
python src/train.py --config config.yaml --output outputs/ensemble_ema
```

#### Best Practices for Ensembles

- **Diversity is key**: Models should differ in architecture, hyperparameters, or training strategy
- **3-5 models**: Sweet spot for performance vs. computational cost
- **Use best checkpoints**: Always use `checkpoint_best.pt` for each model
- **Monitor individual performance**: Ensure each model performs well independently (ensemble won't fix bad models)
- **Different splits (optional)**: Vary `data.split_seed` for completely independent train/val/test splits
- **Complementary strengths**: Combine models that excel at different aspects (e.g., one with high precision, one with high recall)

#### Using Your Trained Ensemble

After training multiple models, use them together:

**If you used the automated training scripts**, convenience scripts are already created:

```bash
# Evaluate ensemble (uses auto-generated script)
bash outputs/ensemble_YYYYMMDD_HHMMSS/evaluate_ensemble.sh

# Evaluate with TTA
bash outputs/ensemble_YYYYMMDD_HHMMSS/evaluate_ensemble.sh --use-tta --tta-mode medium

# Predict on an image
bash outputs/ensemble_YYYYMMDD_HHMMSS/predict_ensemble.sh path/to/image.jpg
```

**Manual ensemble usage:**

```bash
# Evaluate ensemble
python src/evaluate.py \
    --checkpoint \
        outputs/ensemble_model_1/checkpoint_best.pt \
        outputs/ensemble_model_2/checkpoint_best.pt \
        outputs/ensemble_model_3/checkpoint_best.pt \
    --test-csv outputs/ensemble_model_1/test_split.csv \
    --images-dir data/HAM10000/images \
    --output evaluation_ensemble

# Inference with ensemble
python src/inference.py \
    --ensemble \
    --checkpoint \
        outputs/ensemble_model_1/checkpoint_best.pt \
        outputs/ensemble_model_2/checkpoint_best.pt \
    --image path/to/image.jpg

# Combine ensemble + TTA for maximum robustness
python src/evaluate.py \
    --checkpoint \
        outputs/ensemble_model_1/checkpoint_best.pt \
        outputs/ensemble_model_2/checkpoint_best.pt \
    --test-csv outputs/ensemble_model_1/test_split.csv \
    --images-dir data/HAM10000/images \
    --use-tta --tta-mode medium \
    --output evaluation_ensemble_tta
```

## Evaluation

After training, evaluate the model on the test set:

```bash
python src/evaluate.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --test-csv outputs/run_xxx/test_split.csv \
    --images-dir data/HAM10000/images \
    --output evaluation_results
```

### Test-Time Augmentation (TTA)

Improve prediction robustness by averaging predictions across multiple augmented versions:

```bash
python src/evaluate.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --test-csv outputs/run_xxx/test_split.csv \
    --images-dir data/HAM10000/images \
    --use-tta \
    --tta-mode medium \
    --tta-aggregation mean \
    --output evaluation_tta
```

**TTA Modes:**
- `light`: 4 augmentations (horizontal/vertical flips)
- `medium`: 8 augmentations (adds rotations 90°/270°)
- `full`: All available augmentations (adds small rotations ±15°)

**Aggregation Methods:**
- `mean`: Arithmetic average of probabilities (default)
- `geometric_mean`: Geometric average (emphasizes agreement)
- `max`: Maximum probability across augmentations

### Ensemble Prediction

Combine multiple models for improved accuracy and uncertainty quantification:

```bash
python src/evaluate.py \
    --checkpoint \
        outputs/run_1/checkpoint_best.pt \
        outputs/run_2/checkpoint_best.pt \
        outputs/run_3/checkpoint_best.pt \
    --test-csv outputs/run_1/test_split.csv \
    --images-dir data/HAM10000/images \
    --ensemble-aggregation weighted_mean \
    --output evaluation_ensemble
```

**Ensemble Aggregation Methods:**
- `mean`: Uniform averaging (all models equal weight)
- `weighted_mean`: Weight by validation performance (default)
- `geometric_mean`: Geometric average (robust to outliers)

### Combined TTA + Ensemble

For maximum robustness, combine both techniques:

```bash
python src/evaluate.py \
    --checkpoint outputs/run_1/checkpoint_best.pt outputs/run_2/checkpoint_best.pt \
    --test-csv outputs/run_1/test_split.csv \
    --images-dir data/HAM10000/images \
    --use-tta \
    --tta-mode medium \
    --ensemble-aggregation weighted_mean \
    --output evaluation_combined
```

### Evaluation Outputs

This generates:
- `evaluation_metrics.json` - All metrics in JSON format (accuracy, precision, recall, F1, AUC)
- `confusion_matrix.png` - Normalized confusion matrix heatmap
- `roc_curves.png` - ROC curves for all classes (one-vs-rest)
- `per_class_metrics.png` - Bar chart of per-class performance
- `calibration_curve.png` - Model calibration analysis (reliability diagram)

### Evaluation Metrics

The evaluation script computes:
- **Overall metrics**: Accuracy, Macro/Weighted Precision, Recall, F1-Score
- **Per-class metrics**: Class-specific precision, recall, F1, support
- **ROC-AUC**: One-vs-rest AUC for each class
- **Confusion matrix**: Normalized and absolute counts
- **Calibration**: Expected vs. observed confidence

## Inference

### Python API

```python
from src.inference import SkinLesionPredictor

# Load model from checkpoint
predictor = SkinLesionPredictor("outputs/run_xxx/checkpoint_best.pt")

# Make prediction on an image
result = predictor.predict("path/to/image.jpg")

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Description: {result['predicted_class_description']}")

# All class probabilities
for class_name, prob in result['probabilities'].items():
    print(f"{class_name}: {prob:.3f}")
```

### Test-Time Augmentation (TTA)

Improve prediction reliability with TTA:

```python
from src.inference import SkinLesionPredictor

predictor = SkinLesionPredictor("outputs/run_xxx/checkpoint_best.pt")

# Predict with TTA
result = predictor.predict_with_tta(
    "path/to/image.jpg",
    mode="medium",  # light, medium, or full
    aggregation="mean"  # mean, geometric_mean, or max
)

print(f"TTA Predicted: {result['predicted_class']}")
print(f"TTA Confidence: {result['confidence']:.2%}")
print(f"Uncertainty: {result['uncertainty']:.4f}")  # Std dev across augmentations
```

### Ensemble Prediction

Combine multiple models for robust predictions:

```python
from src.inference import load_ensemble_predictor

# Load ensemble from multiple checkpoints
ensemble = load_ensemble_predictor([
    "outputs/run_1/checkpoint_best.pt",
    "outputs/run_2/checkpoint_best.pt",
    "outputs/run_3/checkpoint_best.pt"
])

# Predict with ensemble
result = ensemble.predict(
    "path/to/image.jpg",
    aggregation="weighted_mean"  # mean, weighted_mean, or geometric_mean
)

print(f"Ensemble Predicted: {result['predicted_class']}")
print(f"Ensemble Confidence: {result['confidence']:.2%}")
print(f"Model Agreement: {result['uncertainty']:.4f}")  # Std dev across models
```

### Combined TTA + Ensemble

For maximum robustness:

```python
from src.inference import load_ensemble_predictor

ensemble = load_ensemble_predictor([
    "outputs/run_1/checkpoint_best.pt",
    "outputs/run_2/checkpoint_best.pt",
])

# Each model uses TTA, then ensemble aggregates
result = ensemble.predict_with_tta(
    "path/to/image.jpg",
    tta_mode="medium",
    tta_aggregation="mean",
    ensemble_aggregation="weighted_mean"
)

print(f"Combined Predicted: {result['predicted_class']}")
print(f"Combined Confidence: {result['confidence']:.2%}")
```

### Command Line Inference

**Standard Prediction:**
```bash
python src/inference.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --image path/to/image.jpg
```

**With TTA:**
```bash
python src/inference.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --image path/to/image.jpg \
    --use-tta \
    --tta-mode medium
```

**Ensemble Prediction:**
```bash
python src/inference.py \
    --ensemble \
    --checkpoint \
        outputs/run_1/checkpoint_best.pt \
        outputs/run_2/checkpoint_best.pt \
    --image path/to/image.jpg
```

### Batch Inference

```python
from src.inference import SkinLesionPredictor

predictor = SkinLesionPredictor("checkpoint_best.pt")

# Predict on multiple images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
for img_path in image_paths:
    result = predictor.predict(img_path)
    print(f"{img_path}: {result['predicted_class']} ({result['confidence']:.2%})")
```
```

## Model Architecture

The classifier uses **EfficientNet-V2** as the backbone with a custom classification head:

### Architecture Components

1. **Backbone**: EfficientNet-V2 (pretrained on ImageNet-1K)
   - Efficient compound scaling
   - Progressive learning optimized for faster training
   - Fused-MBConv blocks for better speed/accuracy tradeoff

2. **Classification Heads**:
   
   **Simple Head** (default):
   ```
   Dropout(p=0.35)
   → Linear(feature_dim → 512)
   → BatchNorm1d(512)
   → ReLU
   → Dropout(p=0.35)
   → Linear(512 → 256)
   → BatchNorm1d(256)
   → ReLU
   → Dropout(p=0.35)
   → Linear(256 → 7)
   ```
   
   **ACRNN Head** (attention-based):
   ```
   Conv1D(feature_dim → 128, kernel=3)
   → ReLU
   → LSTM(128 → 64, return_sequences=True)
   → Attention(64 → 16)
   → Linear(16 → 7)
   ```

3. **Loss Functions**:
   - **Cross-Entropy**: Standard with optional class weighting
   - **Focal Loss**: Down-weights easy examples, focuses on hard cases (γ=2.0)
   - **Label Smoothing**: Prevents overconfidence (smoothing=0.05)

4. **Optimizer & Scheduler**:
   - **AdamW** with weight decay (0.02) for better generalization
   - **Cosine Annealing with Warm Restarts** (SGDR)
     - Cycles learning rate for better exploration
     - Warm restarts every 20 epochs
     - 5% warmup at start

## Technical Details

### Data Augmentation

**Training augmentations** (medium intensity - dermatoscopy-safe):
- Random horizontal flip (50%)
- Random vertical flip (30%)
- Random rotation (±25°)
- Color jitter:
  - Brightness: ±20%
  - Contrast: ±20%
  - Saturation: ±20%
  - Hue: ±10%
- Gaussian blur (10% probability)
- Random affine (rotation ±15°, translation ±10%, scale 0.9-1.1)
- Cutout (10% probability - random rectangular masking)

**Validation/Test transforms**:
- Resize to 224×224
- Normalize with ImageNet statistics

All augmentations preserve clinical features and avoid distortions that would alter diagnostic characteristics.

### Handling Class Imbalance

The HAM10000 dataset has severe class imbalance (e.g., `nv` has ~6700 images, `df` has ~115). We address this with:

1. **Weighted Random Sampling** (`use_weighted_sampling: true`)
   - Oversample minority classes during training
   - Each class has equal probability of being sampled per epoch

2. **Class-Weighted Loss**
   - Weights inversely proportional to class frequency
   - Automatically computed from training data
   - Formula: `weight[i] = total_samples / (num_classes * class_count[i])`

3. **Focal Loss** (optional)
   - Down-weights easy examples
   - Focuses on hard, misclassified samples
   - Particularly effective for imbalanced data

4. **Label Smoothing** (optional)
   - Prevents overconfidence on dominant classes
   - Target distribution: `(1-ε)` for true class, `ε/(C-1)` for others

### Preventing Data Leakage

The HAM10000 dataset contains multiple images per lesion (identified by `lesion_id`). To prevent data leakage:

1. **Lesion-aware splitting** (`lesion_aware: true`)
   - All images from the same lesion stay in the same split (train/val/test)
   - Uses `GroupShuffleSplit` from scikit-learn
   - Ensures model never sees the same lesion in both training and evaluation

2. **Stratified splitting**
   - Maintains class distribution across splits
   - Ensures rare classes appear in all splits

3. **Reproducibility**
   - Fixed random seed (`seed: 42`)
   - Data splits saved to CSV for exact reproducibility

## Expected Performance

Approximate metrics on the test set with default configuration (`config.yaml`):

| Metric | Score | Notes |
|--------|-------|-------|
| Accuracy | 80-85% | Overall classification accuracy |
| Macro F1 | 70-75% | Average F1 across all classes |
| Weighted F1 | 80-85% | F1 weighted by class frequency |
| ROC-AUC (macro) | 90-95% | One-vs-rest AUC averaged |

### Per-Class Performance (Approximate)

Performance varies significantly by class due to:
- Class frequency (data availability)
- Inter-class similarity
- Visual complexity

**High-performing classes**: `nv` (most samples), `mel` (distinct features)  
**Challenging classes**: `df` (few samples), `bkl` (similar to other classes)

**Note**: Performance varies based on random seed, hyperparameters, hardware, and the specific train/val/test split.

## Ethical Considerations

1. **Not for clinical use**: This is an educational tool, not a medical device. It has not been validated for clinical diagnosis.

2. **Known biases**: The HAM10000 dataset has demographic and imaging biases:
   - Limited skin tone diversity
   - Mostly European/Australian patients
   - Professional dermoscopic images only
   - Potential annotation biases

3. **Limited generalization**: Model may not perform well on:
   - Different imaging equipment
   - Different lighting conditions
   - Different patient populations
   - Clinical photographs (non-dermoscopic)

4. **Probability, not diagnosis**: Model outputs are probabilistic predictions, not medical diagnoses. High confidence does not equal correctness.

5. **Training data limitations**: Dataset is from 2018 and may not reflect current clinical practices or rare variants.

6. **Transparency**: Always disclose when AI is involved in any healthcare context, even research/education.

**Key principle**: This system should augment, never replace, professional medical judgment.

## References

1. **HAM10000 Dataset**  
   Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Sci. Data 5, 180161 (2018).  
   DOI: [10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

2. **EfficientNetV2**  
   Tan, M., & Le, Q. V. (2021). *EfficientNetV2: Smaller Models and Faster Training*. ICML 2021.  
   arXiv: [2104.00298](https://arxiv.org/abs/2104.00298)

3. **Focal Loss**  
   Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection*. ICCV 2017.  
   arXiv: [1708.02002](https://arxiv.org/abs/1708.02002)

4. **Mixed Precision Training**  
   Micikevicius, P., et al. (2017). *Mixed Precision Training*. ICLR 2018.  
   arXiv: [1710.03740](https://arxiv.org/abs/1710.03740)

## Contributing

Contributions are welcome! Areas for improvement:

- Additional model architectures (Vision Transformers, ConvNeXt)
- Explainability methods (Grad-CAM, attention visualization)
- Model quantization and optimization for deployment
- Web/mobile inference interfaces
- Multi-task learning (e.g., joint classification + segmentation)

Please feel free to submit issues and pull requests.

## License

This project is for educational and research purposes. Please check the HAM10000 dataset license terms at [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) for data usage restrictions.

## Acknowledgments

- HAM10000 dataset creators at Medical University of Vienna
- PyTorch and torchvision teams
- EfficientNet authors at Google Research
- Open-source deep learning community
