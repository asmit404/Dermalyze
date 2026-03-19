# Dermalyze Inference Service

Standalone FastAPI service for skin lesion classification inference. This package is intentionally decoupled from the training pipeline (`../skin_lesion_classifier/`) to enable independent deployment with the frontend.

> ⚠️ **DISCLAIMER**: Educational/research purposes only. Not for medical diagnosis. Consult healthcare professionals for medical advice.

## Architecture

- **Purpose**: Production inference API for frontend (`../frontend/`)
- **Models**: EfficientNet-B0, ConvNeXt-Tiny (trained on HAM10000 dataset)
- **Classes**: 7 skin lesion types (akiec, bcc, bkl, df, mel, nv, vasc)
- **Features**: Test-Time Augmentation (TTA), CORS support, health checks
- **Dependencies**: Minimal (PyTorch + FastAPI, no training libraries)

## Quick Start

```bash
# 1. Navigate to inference service
cd inference_service

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your trained model checkpoint
mkdir -p model
# Copy checkpoint from training outputs:
# cp ../skin_lesion_classifier/outputs/run_xxx/checkpoint_best.pt model/

# 5. Run the API server
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Alternative** (run from repository root):

```bash
uvicorn inference_service.app:app --host 0.0.0.0 --port 8000
```

API will be available at: **http://localhost:8000**  
API docs: **http://localhost:8000/docs**

## Environment Variables

- `MODEL_CHECKPOINT` (default: `inference_service/model/checkpoint_best.pt`)
- `MODEL_IMAGE_SIZE` (default: `224`)
- `USE_TTA` (`true`/`false`, default: `false`)
- `TTA_MODE` (`light` | `medium` | `full`, default: `medium`)
- `TTA_AGGREGATION` (`mean` | `geometric_mean` | `max`, default: `geometric_mean`)
- `CORS_ORIGINS` (comma-separated frontend origins)
- `GEMINI_API_KEY` (optional — enables Gemini-based validation that rejects non-dermatoscopic images before inference)
- `SUPABASE_JWT_SECRET` (required — JWT secret from Supabase Dashboard → API → JWT Secret; used to authenticate `/classify` requests)

## Frontend Contract

- `POST /classify`
- Content type: `multipart/form-data`
- File field name: `file`
- Accepted image types: JPEG, PNG, WebP
- Max upload size: 20 MB

Response:

```json
{
  "classes": [
    {"id": "mel", "name": "Melanoma", "score": 67.4}
  ]
}
```
Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /classify` - Classify skin lesion image

## Frontend Integration

Configure frontend (`../frontend/.env.local`):

```env
VITE_API_URL=http://localhost:8000
```

See [`../frontend/README.md`](../frontend/README.md) for full frontend setup.

## Model Checkpoint

Place your trained checkpoint at:

```
inference_service/model/checkpoint_best.pt
```

Or train one using the ML pipeline:

```bash
cd ../skin_lesion_classifier
python src/train.py --config config.yaml
# Copy best checkpoint to inference service
cp outputs/run_xxx/checkpoint_best.pt ../inference_service/model/
```

See [`../skin_lesion_classifier/README.md`](../skin_lesion_classifier/README.md) for training details.

## Deployment Files

When deploying, include:
- `inference_service/app.py` - FastAPI application
- `inference_service/predictor.py` - Inference logic
- `inference_service/models/` - Model architectures (efficientnet.py, convnext.py)
- `inference_service/metadata.py` - Class labels and preprocessing
- `inference_service/tta_constants.py` - Test-Time Augmentation configs
- `inference_service/model/checkpoint_best.pt` - Trained model weights
- `inference_service/requirements.txt` - DependenciesE_API_URL=http://localhost:8000
```
