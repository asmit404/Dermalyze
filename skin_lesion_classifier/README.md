# Educational Skin Lesion Classification System

An educational skin lesion image classification system using EfficientNet-V2 and modern deep learning practices. This system classifies dermoscopic images into seven clinically relevant skin lesion categories defined in the HAM10000 dataset.

> ⚠️ **DISCLAIMER**: This system is for **educational and research purposes only**. It does not provide medical diagnosis or clinical decision-making. Always consult a qualified healthcare professional for medical advice.

## Overview

This project demonstrates how to build a modern, maintainable, and reproducible deep learning system for multi-class skin lesion classification. Key features include:

- **EfficientNet-V2 backbone** with transfer learning
- **Lesion-aware data splitting** to prevent data leakage
- **Class-imbalanced learning** with focal loss and weighted sampling
- **Comprehensive evaluation** with per-class metrics and calibration analysis
- **Modular architecture** for easy extension and deployment

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
├── config.yaml              # Training configuration
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/
│   └── HAM10000/
│       ├── images/         # Dermoscopic images
│       └── labels.csv      # Image labels
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py      # Dataset and data loading
│   ├── models/
│   │   ├── __init__.py
│   │   └── efficientnet.py # EfficientNet-V2 model
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── inference.py        # Inference module
│   └── prepare_data.py     # Data preparation script
├── notebooks/              # Jupyter notebooks
└── outputs/                # Training outputs (created during training)
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

## Training

### Basic Training

```bash
python src/train.py --config config.yaml
```

### Training with Custom Output Directory

```bash
python src/train.py --config config.yaml --output outputs/experiment_1
```

### Resume Training from Checkpoint

```bash
python src/train.py --config config.yaml --resume outputs/run_xxx/checkpoint_latest.pt
```



### Configuration Options

Key settings in `config.yaml`:

```yaml
model:
  size: small              # Options: small, medium, large
  pretrained: true         # Use ImageNet pretrained weights

training:
  batch_size: 32           # Reduce if out of memory
  epochs: 30               # Maximum epochs
  lr: 0.0001              # Learning rate
  use_amp: true           # Mixed precision (CUDA only)

loss:
  type: focal             # Options: focal, cross_entropy, label_smoothing
```

## Evaluation

After training, evaluate the model:

```bash
python src/evaluate.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --test-csv outputs/run_xxx/test_split.csv \
    --images-dir data/HAM10000/images \
    --output evaluation_results
```

This generates:
- `evaluation_metrics.json` - All metrics in JSON format
- `confusion_matrix.png` - Normalized confusion matrix
- `roc_curves.png` - ROC curves for all classes
- `calibration_curve.png` - Model calibration analysis
- `per_class_metrics.png` - Bar chart of per-class performance

## Inference

### Python API

```python
from src.inference import SkinLesionPredictor

# Load model
predictor = SkinLesionPredictor("outputs/run_xxx/checkpoint_best.pt")

# Make prediction
result = predictor.predict("path/to/image.jpg")

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Description: {result['predicted_class_description']}")
```

### Command Line

```bash
python src/inference.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --image path/to/image.jpg
```

## Model Architecture

The classifier uses **EfficientNet-V2** as the backbone:

- **Pretrained weights**: ImageNet-1K
- **Custom head**: Dropout → FC(512) → BN → ReLU → Dropout → FC(256) → BN → ReLU → Dropout → FC(7)
- **Loss function**: Focal Loss (handles class imbalance)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing with warm restarts

### Model Variants

| Size | Parameters | Image Size | GPU Memory |
|------|------------|------------|------------|
| small | ~21M | 224×224 | ~4 GB |
| medium | ~54M | 224×224 | ~8 GB |
| large | ~118M | 224×224 | ~16 GB |

## Technical Details

### Data Augmentation

Training augmentations (medium strength):
- Random horizontal/vertical flip
- Random rotation (±30°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformations
- Random crop with resize

### Handling Class Imbalance

1. **Weighted random sampling**: Oversample minority classes
2. **Focal loss**: Down-weight easy examples
3. **Class-weighted loss**: Weight inversely proportional to frequency

### Preventing Data Leakage

The HAM10000 dataset contains multiple images per lesion. To prevent data leakage:
- Stratified splitting by class
- Group splitting by lesion_id (all images of same lesion stay in same split)

## Expected Performance

Approximate metrics on the test set (may vary):

| Metric | Score |
|--------|-------|
| Accuracy | ~80-85% |
| Macro F1 | ~70-75% |
| ROC-AUC (macro) | ~90-95% |

Note: Performance varies based on random seed, hyperparameters, and hardware.

## Ethical Considerations

1. **Not for clinical use**: This is an educational tool, not a medical device
2. **Known biases**: The HAM10000 dataset has demographic biases
3. **Limited generalization**: May not work well on different imaging conditions
4. **Probability, not diagnosis**: Outputs are probabilistic, not deterministic

## License

This project is for educational purposes. Please check the HAM10000 dataset license for data usage terms.

## References

- HAM10000 Dataset: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018).
- EfficientNet-V2: Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
