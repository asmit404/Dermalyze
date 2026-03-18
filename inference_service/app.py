"""FastAPI server for standalone skin lesion inference service."""

from __future__ import annotations

import io
import importlib
import logging
import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_dotenv_spec = importlib.util.find_spec("dotenv")
if _dotenv_spec is not None:
    dotenv = importlib.import_module("dotenv")
    load_dotenv = getattr(dotenv, "load_dotenv", None)
    if callable(load_dotenv):
        _env_path = Path(__file__).parent / ".env.local"
        _loaded = load_dotenv(_env_path, override=True)
        print(f"[dotenv] path={_env_path} loaded={_loaded}")

try:
    from .predictor import SkinLesionPredictor
except ImportError:
    from predictor import SkinLesionPredictor

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

_SERVICE_DIR = Path(__file__).resolve().parent
_DEFAULT_CHECKPOINT = _SERVICE_DIR / "models" / "checkpoint_best.pt"
_LEGACY_DEFAULT_CHECKPOINT = _SERVICE_DIR / "model" / "checkpoint_best.pt"

DEFAULT_CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://www.dermalyze.tech",
    "https://dermalyze.tech",
]
PRODUCTION_WWW_ORIGIN = "https://www.dermalyze.tech"


def _resolve_checkpoint_path() -> Path:
    explicit = os.environ.get("MODEL_CHECKPOINT")
    if explicit:
        return Path(explicit)
    if _DEFAULT_CHECKPOINT.exists():
        return _DEFAULT_CHECKPOINT
    return _LEGACY_DEFAULT_CHECKPOINT


CHECKPOINT_PATH = _resolve_checkpoint_path()
IMAGE_SIZE = int(os.environ.get("MODEL_IMAGE_SIZE", "224"))
USE_TTA = os.environ.get("USE_TTA", "false").lower() == "true"
TTA_MODE = os.environ.get("TTA_MODE", "medium")
TTA_AGGREGATION = os.environ.get("TTA_AGGREGATION", "geometric_mean")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
print(
    f"[config] GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'NOT SET'} ({len(GEMINI_API_KEY)} chars)"
)

_raw_origins = os.environ.get(
    "CORS_ORIGINS",
    "",
)
configured_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
ORIGINS = sorted(set(DEFAULT_CORS_ORIGINS + configured_origins))
CORS_ORIGIN_REGEX = os.environ.get(
    "CORS_ORIGIN_REGEX",
    r"https://([a-zA-Z0-9-]+\.)?dermalyze\.tech",
)
print(f"[config] CORS_ORIGINS: {', '.join(ORIGINS)}")
print(f"[config] CORS_ORIGIN_REGEX: {CORS_ORIGIN_REGEX}")

app = FastAPI(
    title="Dermalyze Inference API",
    description="Standalone inference API decoupled from training project",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup_log() -> None:
    if _raw_origins.strip() and PRODUCTION_WWW_ORIGIN not in configured_origins:
        logger.warning(
            "CORS_ORIGINS is set but missing %s; browser requests from the www frontend may fail.",
            PRODUCTION_WWW_ORIGIN,
        )

    if GEMINI_API_KEY:
        logger.info(
            "Gemini validation: ENABLED (key loaded, %d chars)", len(GEMINI_API_KEY)
        )
    else:
        logger.warning("Gemini validation: DISABLED — GEMINI_API_KEY not set")


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
        _predictor = SkinLesionPredictor(
            checkpoint_path=CHECKPOINT_PATH,
            image_size=IMAGE_SIZE,
        )
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


async def _validate_dermatoscopic(
    image_bytes: bytes, mime_type: str = "image/jpeg"
) -> None:
    """Raise HTTPException(422) if the image is not a dermatoscopic image.

    Uses Gemini Flash to perform a binary yes/no check. Skips validation
    when GEMINI_API_KEY is not configured. Returns 503 if Gemini is
    unavailable.
    """
    if not GEMINI_API_KEY:
        return

    try:
        genai = importlib.import_module("google.genai")
        types = getattr(genai, "types")

        client = genai.Client(api_key=GEMINI_API_KEY)
        img = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                img,
                (
                    "Is this a dermatoscopic (dermoscopy) image of a skin lesion? "
                    "Dermoscopy images are close-up photographs of skin taken with a "
                    "dermatoscope, showing detailed skin surface structures under "
                    "magnification — including hair follicles, blood vessels, and "
                    "pigmentation patterns. Answer with ONLY 'yes' or 'no'."
                ),
            ],
        )

        if not response.text.strip().lower().startswith("yes"):
            logger.info("Gemini rejected image: %s", response.text.strip())
            raise HTTPException(
                status_code=422,
                detail=(
                    "The uploaded image does not qualify as a dermatoscopic image. Please upload another."
                ),
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Gemini validation error")
        raise HTTPException(
            status_code=503,
            detail="Image validation is temporarily unavailable. Please try again later.",
        )


@app.get("/", tags=["Health"])
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
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

    await _validate_dermatoscopic(
        image_bytes, mime_type=file.content_type or "image/jpeg"
    )

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
