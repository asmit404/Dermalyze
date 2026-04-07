# Dermalyze

AI-assisted skin lesion classification system for educational purposes.

> ⚠️ **DISCLAIMER**: Educational/research purposes only. Not for medical diagnosis. Consult healthcare professionals for medical advice.

## Overview

Dermalyze is a full-stack machine learning application that classifies dermoscopic images across 7 skin lesion types using deep learning models trained on the HAM10000 dataset.

**Classes**: akiec (Actinic keratoses), bcc (Basal cell carcinoma), bkl (Benign keratosis), df (Dermatofibroma), mel (Melanoma), nv (Melanocytic nevi), vasc (Vascular lesions)

## Architecture

```
Dermalyze/
├── frontend/              # React + Vite web application
├── inference_service/     # FastAPI inference API
└── skin_lesion_classifier/ # ML training pipeline
```

### Components

| Component | Purpose | Tech Stack |
|-----------|---------|------------|
| [**Frontend**](frontend/README.md) | Web UI for image upload and results | React, Vite, TypeScript, Tailwind CSS |
| [**Inference Service**](inference_service/README.md) | Production-ready classification API with Grad-CAM | FastAPI, PyTorch, uvicorn |
| [**Training Pipeline**](skin_lesion_classifier/README.md) | Model training and evaluation with hyperparameter tuning | PyTorch, EfficientNet (B0-B7), EfficientNetV2 (S/M/L), ConvNeXt-Tiny, ResNeSt-101, SE-ResNeXt-101 |

## Quick Start

### 1. Train a Model

```bash
cd skin_lesion_classifier
bash scripts/install_pytorch.sh
pip install -r requirements.txt
python src/train.py --config config.yaml
```

### 2. Export to Inference Service

```bash
cp skin_lesion_classifier/outputs/run_xxx/checkpoint_best.pt inference_service/model/
```

### 3. Run Inference API

```bash
cd inference_service
pip install -r requirements.txt
uvicorn inference_service.app:app --host 0.0.0.0 --port 8000
```

### 4. Launch Frontend

```bash
cd frontend
npm install
npm run dev
```

Configure `frontend/.env.local`:
```env
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_key
```

## Data Preparation

The `prepare_data.py` script handles the HAM10000 dataset with automatic one-hot to single-label conversion and metadata extraction.

### HAM10000 Dataset Structure

```
data/HAM10000_Training/
    images/
        ISIC_0024306.jpg
        ...
    ground_truth.csv  (one-hot encoded: MEL, NV, BCC, AKIEC, BKL, DF, VASC)
    metadata.csv      (contains age_approx, sex, anatom_site_general)

data/HAM10000_Val/
    images/
    ground_truth.csv

data/HAM10000_Test/
    images/
    ground_truth.csv
```

### Quick Start: Prepare HAM10000 Data

```bash
# Training set (with metadata)
python skin_lesion_classifier/src/prepare_data.py \
  --data-dir data/HAM10000_Training \
  --include-metadata

# Validation set
python skin_lesion_classifier/src/prepare_data.py \
  --data-dir data/HAM10000_Val

# Test set
python skin_lesion_classifier/src/prepare_data.py \
  --data-dir data/HAM10000_Test
```

**Outputs**:
- `HAM10000_Training/labels_with_metadata.csv` (includes age_approx, sex, anatom_site)
- `HAM10000_Val/labels.csv` (image_id and label only)
- `HAM10000_Test/labels.csv` (image_id and label only)

### Key Features

- ✅ **HAM10000 Support**: One-hot encoded ground_truth.csv conversion
- ✅ **Metadata Extraction**: Auto-extracts age_approx, sex, anatom_site from metadata.csv
- ✅ **Smart Column Mapping**: Handles naming variations (anatom_site_general → anatom_site)
- ✅ **Image Validation**: Validates image format and readability
- ✅ **Dataset Statistics**: Displays class distribution and metadata coverage

## Workflow

1. **Prepare Data**: Use `prepare_data.py` to format your dataset and extract metadata
2. **Train**: Use `skin_lesion_classifier/` to train models on prepared data
3. **Export**: Copy best checkpoint to `inference_service/model/`
4. **Deploy**: Run inference API independently from training code
5. **Integrate**: Frontend connects to inference API for predictions

## Features

### Model Architectures
- **EfficientNet family**: B0, B1, B2, B3, B4, B5, B6, B7
- **EfficientNetV2 family**: V2-S (Small), V2-M (Medium), V2-L (Large) - faster training, better parameter efficiency
- **ResNet variants**: ResNeSt-101, SE-ResNeXt-101
- **ConvNeXt-Tiny**: Modern ConvNet architecture
- **Multi-input models**: Combine image features with patient metadata (age, sex, anatomical site)
- **Variable input sizes**: Configurable image resolution (224-600px) for all models

### Image Size Configuration

Configure input resolution in `config.yaml`:

```yaml
model:
  image_size: 224  # Set to 224, 256, 320, 384, 512, etc.
```

**Recommended sizes by model:**
- **EfficientNet-B0**: 224 (default) | B3: 300 | B5: 456 | B7: 600
- **EfficientNetV2-S**: 224-300 | V2-M: 224-384 | V2-L: 300-480 (faster training than B-series)
- **ResNeSt-101**: 224-384
- **ConvNeXt-Tiny**: 224-256

**Memory & Speed Tradeoffs:**

| Size | Relative Memory | Batch Size (16GB) | Best For |
|------|-----------------|-------------------|----------|
| 224  | 1x              | 32-64             | Fast training, low memory |
| 320  | 2x              | 16-32             | Balanced accuracy/speed |
| 384  | 3x              | 12-24             | Better detail capture |
| 512  | 5.3x            | 6-12              | High-resolution lesions |

**Best Practices:**
- ✅ Match training and inference image sizes
- ✅ Reduce batch size when increasing resolution (use gradient accumulation to maintain effective batch size)
- ✅ Enable mixed precision (`use_amp: true`) for high-resolution training
- ✅ Use gradient checkpointing (`use_gradient_checkpointing: true`) to save memory

For complete configuration examples and advanced tuning, see [IMAGE_SIZE_GUIDE.md](IMAGE_SIZE_GUIDE.md).

### Training & Evaluation
- **Configurable best checkpoint selection** - Optimize for macro_recall, macro_f1, or custom metrics
- **Comprehensive macro-averaged metrics** - Track per-class and macro performance for imbalanced datasets
- **Test-Time Augmentation (TTA)** for improved accuracy
- **Model explainability with Grad-CAM** - Visual heatmaps showing decision rationale
- **K-Fold cross-validation** and ensemble training
- **Enhanced hyperparameter tuning** with Optuna - aligned checkpoint selection and explicit eval config
- **Mixed precision training** (AMP) and gradient checkpointing
- **Class imbalance handling**: Configurable weighted sampling, focal loss, and label smoothing
- **Class-conditioned augmentation** - Per-class augmentation policies

### Application Features
- **User authentication** via Supabase
- **Analysis history** tracking
- **Responsive web interface**
- **Grad-CAM visualizations** for model interpretability

## Documentation

- [Frontend Setup & Usage](frontend/README.md)
- [Inference API Documentation](inference_service/README.md)
- [Training Pipeline Guide](skin_lesion_classifier/README.md)

## License

Educational use only. See individual component READMEs for details.
