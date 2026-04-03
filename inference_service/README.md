# Dermalyze Inference Service

Standalone FastAPI inference API for Dermalyze. This service is intentionally decoupled from the training pipeline for independent deployment.

> ⚠️ DISCLAIMER: Educational/research purposes only. Not for medical diagnosis.

## Overview

- Framework: FastAPI
- Checkpoint loading: local `.pt` checkpoint
- Supported model architectures in this package: ConvNeXt-Tiny and torchvision EfficientNet variants (`efficientnet_b0`-`efficientnet_b7`, `efficientnetv2_s`, `efficientnetv2_m`, `efficientnetv2_l`)
- Output classes: 7 lesion classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`)

Metadata-fusion checkpoints are supported. If a checkpoint contains `metadata_encoder_state`, the predictor automatically instantiates the multi-input fusion model.

## Quick Start

```bash
cd inference_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Alternative from repo root:

```bash
uvicorn inference_service.app:app --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

## Checkpoint Resolution

Load order:

1. `MODEL_CHECKPOINT` env var (if set)
2. `inference_service/models/checkpoint_best.pt`
3. legacy fallback: `inference_service/model/checkpoint_best.pt`

If no valid checkpoint is found, `/classify` returns `503`.

## API Endpoints

- `GET /` - basic health response
- `POST /classify` - authenticated image classification

There is no separate `/health` route in current code.

## Authentication

`POST /classify` requires `Authorization: Bearer <jwt>`.

Token verification behavior:

- Preferred: JWKS verification via `SUPABASE_URL` (`/auth/v1/.well-known/jwks.json`)
- Fallback: HS256 verification via `SUPABASE_JWT_SECRET`
- If neither is configured, classify requests fail with `503`

## Request/Response Contract

Request:

- method: `POST /classify`
- content type: `multipart/form-data`
- file field: `file`
- optional metadata fields: `age_approx` (number), `sex` (string), `anatom_site` (string), `localization` (string alias)
- accepted MIME types: `image/jpeg`, `image/png`, `image/webp`
- max file size: 10 MB

If metadata is omitted for a metadata-fusion checkpoint, inference still runs using default/zero metadata features.

Response:

```json
{
  "classes": [
    { "id": "mel", "name": "Melanoma", "score": 87.42 },
    { "id": "bcc", "name": "Basal Cell Carcinoma", "score": 9.31 }
  ]
}
```

`classes` are returned in descending score order.

## Validation and Safety Controls

- Magic-byte validation enforces declared MIME type matches file bytes.
- Optional Gemini validation rejects non-dermatoscopic images when `GEMINI_API_KEY` is set.
- Rate limit on `/classify`: `20/minute` per client key.
- Rate-limit key can use `X-Forwarded-For` when request comes from a trusted proxy.

## Environment Variables

Core inference:

- `MODEL_CHECKPOINT` (optional; explicit checkpoint path)
- `MODEL_IMAGE_SIZE` (default: `224`)
- `USE_TTA` (`true|false`, default: `false`)
- `TTA_MODE` (`light|medium|full`, default: `medium`)
- `TTA_AGGREGATION` (`mean|geometric_mean|max`, default: `geometric_mean`)

Auth:

- `SUPABASE_URL` (recommended for JWKS verification)
- `SUPABASE_JWT_SECRET` (legacy fallback)

Validation:

- `GEMINI_API_KEY` (optional)

CORS and proxy:

- `CORS_ORIGINS` (comma-separated additional origins)
- `CORS_ORIGIN_REGEX` (default: `https://([a-zA-Z0-9-]+\.)?dermalyze\.tech`)
- `TRUSTED_PROXY_IPS` (default: `127.0.0.1,::1`)

Built-in CORS defaults always include:

- `http://localhost:3000`
- `http://127.0.0.1:3000`
- `http://localhost:5173`
- `http://127.0.0.1:5173`
- `https://www.dermalyze.tech`
- `https://dermalyze.tech`

## Error Codes

- `400` invalid inference inputs/state
- `401` missing/invalid/expired auth token
- `413` file too large
- `415` unsupported media type or magic-byte mismatch
- `422` non-dermatoscopic image (Gemini-enabled mode)
- `429` rate limit exceeded
- `500` inference runtime failure
- `503` auth service/checkpoint/validation backend unavailable

## Docker Notes

`inference_service/Dockerfile`:

- installs dependencies
- downloads checkpoint to `/app/models/checkpoint_best.pt`
- sets `MODEL_CHECKPOINT=/app/models/checkpoint_best.pt`
- starts Uvicorn on port `7860`

## Integration

- Frontend setup: [`../frontend/README.md`](../frontend/README.md)
- Model training: [`../skin_lesion_classifier/README.md`](../skin_lesion_classifier/README.md)
