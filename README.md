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
| [**Inference Service**](inference_service/README.md) | Production-ready classification API | FastAPI, PyTorch, uvicorn |
| [**Training Pipeline**](skin_lesion_classifier/README.md) | Model training and evaluation | PyTorch, EfficientNet (B0-B7), ConvNeXt-Tiny, ResNeSt-101, SE-ResNeXt-101 |

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

The `prepare_data.py` script handles multiple skin lesion datasets (ISIC 2019, ISIC 2018, HAM10000, etc.) with automatic metadata extraction and label conversion.

### Quick Start: Prepare ISIC 2019 Data

```bash
python skin_lesion_classifier/src/prepare_data.py \
  --data-dir data/ISIC2019Training \
  --metadata-columns age_approx sex anatom_site \
  --skip-validation
```

**Output**: `data/ISIC2019Training/labels_with_metadata.csv`
```
image_id,label,lesion_id,age_approx,sex,anatom_site
ISIC_0000000,nv,,55.0,female,anterior torso
ISIC_0000001,nv,,30.0,female,anterior torso
```

### Key Features

- ✅ **Multiple Datasets**: HAM10000, ISIC 2019, ISIC 2018, custom datasets
- ✅ **Metadata Extraction**: Auto-extracts age_approx, sex, anatomic_site
- ✅ **Smart Column Mapping**: Handles naming variations across datasets
- ✅ **One-Hot Label Conversion**: Converts ISIC one-hot encoded to single labels
- ✅ **Balanced Augmentation**: Creates 19K balanced dataset with metadata preservation
- ✅ **Multi-Dataset Support**: Works with separate labels/metadata files

### Dataset Support

| Dataset | Labels | Metadata | Format | Status |
|---------|--------|----------|--------|--------|
| ISIC 2019 Training | ✅ | ✅ | One-hot | ✅ Full support |
| ISIC 2019 Test | ✅ | ✅ | One-hot | ✅ Full support |
| ISIC 2018 Val | ✅ | ✅ | One-hot | ✅ Full support |
| HAM10000 | ✅ | ✅ | Single-label | ✅ Full support |
| balanced_21k | ✅ | ❌ | Single-label | ✅ Labels only |
| braaff-bald | ✅ | ✅ | Single-label | ✅ Full support |

### Create Balanced Dataset with Metadata

```bash
python skin_lesion_classifier/src/prepare_data.py \
  --data-dir data/ISIC2019Training \
  --metadata-columns age_approx sex anatom_site \
  --build-balanced-dataset \
  --balanced-output-dir data/balanced_21k \
  --balanced-output-csv data/balanced_21k/labels_with_metadata.csv
```

Creates balanced training data (21,000 images) with:
- **Metadata preservation**: age_approx, sex, anatom_site preserved through augmentation
- **Class distribution**: mel=3000, nv=3000, bcc=3000, akiec=3000, bkl=3000, df=3000, vasc=3000
- **Augmented images**: New variations retain source metadata

## Workflow

1. **Prepare Data**: Use `prepare_data.py` to format your dataset and extract metadata
2. **Train**: Use `skin_lesion_classifier/` to train models on prepared data
3. **Export**: Copy best checkpoint to `inference_service/model/`
4. **Deploy**: Run inference API independently from training code
5. **Integrate**: Frontend connects to inference API for predictions

## Features

### Model Architectures
- **EfficientNet family**: B0, B1, B2, B3, B4, B5, B6, B7
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
- **Test-Time Augmentation (TTA)** for improved accuracy
- **K-Fold cross-validation** and ensemble training
- **Hyperparameter tuning** with Optuna
- **Mixed precision training** (AMP) and gradient checkpointing
- **Class imbalance handling**: Weighted sampling and loss functions

### Application Features
- **User authentication** via Supabase
- **Analysis history** tracking
- **Responsive web interface**

## Documentation

- [Frontend Setup & Usage](frontend/README.md)
- [Inference API Documentation](inference_service/README.md)
- [Training Pipeline Guide](skin_lesion_classifier/README.md)

## License

Educational use only. See individual component READMEs for details.
