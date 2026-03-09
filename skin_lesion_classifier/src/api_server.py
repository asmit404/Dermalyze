"""FastAPI server that exposes skin_lesion_classifier inference to the frontend.

Run from `skin_lesion_classifier/`:
    uvicorn src.api_server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference import SkinLesionPredictor

# Keep IDs aligned with frontend class metadata.
CLASS_IDS: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES: Dict[str, str] = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesion",
}

_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "outputs" / "run_best_yet" / "checkpoint_best.pt"
)
CHECKPOINT_PATH = Path(os.environ.get("MODEL_CHECKPOINT", str(_DEFAULT_CHECKPOINT)))
IMAGE_SIZE = int(os.environ.get("MODEL_IMAGE_SIZE", "224"))
USE_TTA = os.environ.get("USE_TTA", "false").lower() == "true"
TTA_MODE = os.environ.get("TTA_MODE", "medium")
TTA_AGGREGATION = os.environ.get("TTA_AGGREGATION", "geometric_mean")

_raw_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
)
ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app = FastAPI(
    title="Dermalyze Inference API",
    description="Inference API backed by skin_lesion_classifier",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ClassResult(BaseModel):
    id: str
    name: str
    score: float


class ClassifyResponse(BaseModel):
    classes: List[ClassResult]


_predictor: SkinLesionPredictor | None = None


def _get_predictor() -> SkinLesionPredictor:
    global _predictor
    if _predictor is None:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                "Model checkpoint not found at "
                f"'{CHECKPOINT_PATH}'. Set MODEL_CHECKPOINT to a valid .pt file."
            )
        _predictor = SkinLesionPredictor(checkpoint_path=CHECKPOINT_PATH, image_size=IMAGE_SIZE)
    return _predictor


def _to_frontend_response(probabilities: Dict[str, float]) -> List[ClassResult]:
    results = [
        ClassResult(
            id=class_id,
            name=CLASS_NAMES[class_id],
            score=round(float(probabilities.get(class_id, 0.0)) * 100.0, 2),
        )
        for class_id in CLASS_IDS
    ]
    results.sort(key=lambda x: x.score, reverse=True)
    return results


@app.get("/", tags=["Health"])
def health() -> dict:
    return {
        "status": "ok",
        "model_checkpoint": str(CHECKPOINT_PATH),
        "use_tta": USE_TTA,
    }


@app.post("/classify", response_model=ClassifyResponse, tags=["Inference"])
async def classify_image(file: UploadFile = File(...)) -> ClassifyResponse:
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Send JPEG, PNG, or WebP.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit.")

    try:
        predictor = _get_predictor()
        if USE_TTA:
            prediction = predictor.predict_with_tta(
                image=image_bytes,
                tta_mode=TTA_MODE,
                aggregation=TTA_AGGREGATION,
                include_disclaimer=False,
            )
        else:
            prediction = predictor.predict(image=image_bytes, include_disclaimer=False)

        probs = prediction.get("probabilities")
        if not isinstance(probs, dict):
            raise RuntimeError("Inference output did not include class probabilities.")

        return ClassifyResponse(classes=_to_frontend_response(probs))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
