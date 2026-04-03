"""FastAPI server for standalone skin lesion inference service."""

from __future__ import annotations

import io
import importlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import jwt
from jwt import PyJWKClient

logger = logging.getLogger(__name__)

_dotenv_spec = importlib.util.find_spec("dotenv")
if _dotenv_spec is not None:
    dotenv = importlib.import_module("dotenv")
    load_dotenv = getattr(dotenv, "load_dotenv", None)
    if callable(load_dotenv):
        _env_path = Path(__file__).parent / ".env.local"
        load_dotenv(_env_path, override=True)

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

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "").strip()

print(f"[config] SUPABASE_URL: {'SET - ' + SUPABASE_URL if SUPABASE_URL else 'NOT SET'}")
print(f"[config] SUPABASE_JWT_SECRET: {'SET' if SUPABASE_JWT_SECRET else 'NOT SET'}")

# Initialize JWKS client for ES256 token verification
_jwks_client = None
if SUPABASE_URL:
    _jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    try:
        _jwks_client = PyJWKClient(_jwks_url, cache_keys=True, max_cached_keys=16)
        print(f"[config] JWKS client initialized: {_jwks_url}")
        logger.info(f"JWKS client initialized: {_jwks_url}")
    except Exception as e:
        print(f"[config] ERROR initializing JWKS client: {e}")
        logger.error(f"Failed to initialize JWKS client: {e}")
elif SUPABASE_JWT_SECRET:
    print("[config] Using legacy HS256 JWT verification (SUPABASE_URL not set)")
    logger.info("Using legacy HS256 JWT verification (SUPABASE_URL not set)")
else:
    print("[config] WARNING: Neither SUPABASE_URL nor SUPABASE_JWT_SECRET set")
    logger.warning("Neither SUPABASE_URL nor SUPABASE_JWT_SECRET set — /classify endpoint will reject all requests")

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

_trusted_proxy_ips_raw = os.environ.get("TRUSTED_PROXY_IPS", "127.0.0.1,::1")
TRUSTED_PROXY_IPS = {
    ip.strip() for ip in _trusted_proxy_ips_raw.split(",") if ip.strip()
}


def _rate_limit_key(request: Request) -> str:
    """Use X-Forwarded-For when request comes from a trusted proxy."""
    client_host = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("x-forwarded-for", "")

    if forwarded_for and ("*" in TRUSTED_PROXY_IPS or client_host in TRUSTED_PROXY_IPS):
        # X-Forwarded-For can contain multiple IPs: client, proxy1, proxy2...
        first_hop = forwarded_for.split(",", 1)[0].strip()
        if first_hop:
            return first_hop

    return client_host

limiter = Limiter(key_func=_rate_limit_key)

app = FastAPI(
    title="Dermalyze Inference API",
    description="Standalone inference API decoupled from training project",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


def verify_jwt_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> dict:
    """Verify Supabase JWT token and return decoded payload.

    Raises HTTPException(401) if token is invalid, expired, or missing.
    Raises HTTPException(503) if authentication service is not configured.
    """
    if not _jwks_client and not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Authentication service is not configured. Please contact support.",
        )

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication credentials.",
        )

    token = credentials.credentials
    try:
        # Try JWKS-based verification first (ES256, RS256)
        if _jwks_client:
            try:
                # Get signing key from JWKS
                signing_key = _jwks_client.get_signing_key_from_jwt(token)
                logger.debug(f"Got signing key: {type(signing_key.key)}")
                payload = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["ES256", "RS256", "HS256"],
                    options={"verify_aud": False}
                )
                logger.debug("JWKS verification successful")
            except Exception as jwks_error:
                # Fall back to legacy HS256 if JWKS fails and secret is available
                logger.warning(f"JWKS verification failed: {type(jwks_error).__name__}: {jwks_error}")
                if SUPABASE_JWT_SECRET:
                    logger.debug("Trying legacy HS256 fallback")
                    payload = jwt.decode(
                        token,
                        SUPABASE_JWT_SECRET,
                        algorithms=["HS256", "HS512"],
                        options={"verify_aud": False}
                    )
                else:
                    error_text = str(jwks_error).lower()
                    service_unavailable_markers = (
                        "connection", "network", "timeout", "timed out", "dns", "jwks"
                    )
                    if any(marker in error_text for marker in service_unavailable_markers):
                        raise HTTPException(
                            status_code=503,
                            detail="Authentication service is temporarily unavailable. Please try again.",
                        )
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid authentication token.",
                    )
        # Legacy HS256 verification (when SUPABASE_URL not set)
        elif SUPABASE_JWT_SECRET:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256", "HS512"],
                options={"verify_aud": False}
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Authentication service is not configured.",
            )

        # Check if token has required claims
        if "sub" not in payload:
            raise HTTPException(
                status_code=401,
                detail="Invalid token: missing user identifier.",
            )

        return payload
    except HTTPException:
        raise
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired. Please log in again.",
        )
    except jwt.InvalidTokenError as e:
        logger.warning("JWT validation failed: %s", str(e))
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token.",
        )
    except Exception as e:
        logger.exception("Unexpected JWT verification error: %s", str(e))
        raise HTTPException(
            status_code=503,
            detail="Authentication service is temporarily unavailable. Please try again.",
        )


@app.on_event("startup")
async def _startup_log() -> None:
    if _raw_origins.strip() and PRODUCTION_WWW_ORIGIN not in configured_origins:
        logger.warning(
            "CORS_ORIGINS is set but missing %s; browser requests from the www frontend may fail.",
            PRODUCTION_WWW_ORIGIN,
        )

    if GEMINI_API_KEY:
        logger.info("Gemini validation: ENABLED")
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


def _validate_magic_bytes(image_bytes: bytes, declared_content_type: str) -> None:
    """Validate file magic bytes match declared content type.

    Raises HTTPException(415) if magic bytes don't match expected image format.
    """
    if len(image_bytes) < 12:
        raise HTTPException(
            status_code=415,
            detail="File is too small to be a valid image.",
        )

    # Check magic bytes for common image formats
    magic_signatures = {
        b'\xff\xd8\xff': 'image/jpeg',  # JPEG
        b'\x89PNG\r\n\x1a\n': 'image/png',  # PNG
        b'RIFF': 'image/webp',  # WebP (needs additional check)
    }

    detected_type = None
    for signature, mime_type in magic_signatures.items():
        if image_bytes.startswith(signature):
            # For WebP, need to verify the WEBP string at position 8
            if mime_type == 'image/webp':
                if len(image_bytes) >= 12 and image_bytes[8:12] == b'WEBP':
                    detected_type = mime_type
                    break
            else:
                detected_type = mime_type
                break

    if not detected_type:
        raise HTTPException(
            status_code=415,
            detail="File format not recognized. Please upload a valid JPEG, PNG, or WebP image.",
        )

    # Verify detected type matches declared content type
    # Normalize for comparison (both jpeg and jpg are acceptable)
    normalized_declared = declared_content_type.lower()
    normalized_detected = detected_type.lower()

    if normalized_declared != normalized_detected:
        # Special case: image/jpg is often used but should be image/jpeg
        if not (normalized_declared in ('image/jpg', 'image/jpeg') and
                normalized_detected in ('image/jpg', 'image/jpeg')):
            raise HTTPException(
                status_code=415,
                detail=f"File content ({normalized_detected}) does not match declared type ({normalized_declared}).",
            )


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
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse, tags=["Inference"])
@limiter.limit("20/minute")
async def classify_image(
    request: Request,
    file: UploadFile = File(...),
    age_approx: Optional[float] = Form(None),
    sex: Optional[str] = Form(None),
    anatom_site: Optional[str] = Form(None),
    localization: Optional[str] = Form(None),
    user: dict = Depends(verify_jwt_token)
) -> ClassifyResponse:
    declared_content_type = (file.content_type or "application/octet-stream").lower()
    normalized_content_type = "image/jpeg" if declared_content_type == "image/jpg" else declared_content_type

    if normalized_content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{declared_content_type}'. Send JPEG, PNG, or WebP.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    # Validate file magic bytes match declared content type
    _validate_magic_bytes(image_bytes, normalized_content_type)

    await _validate_dermatoscopic(
        image_bytes, mime_type=normalized_content_type
    )

    try:
        predictor = _get_predictor()
        metadata_payload: dict | None = None
        if (
            age_approx is not None
            or sex is not None
            or anatom_site is not None
            or localization is not None
        ):
            resolved_site = localization if localization is not None else anatom_site
            metadata_payload = {
                "age": age_approx,
                "age_approx": age_approx,
                "sex": sex,
                "anatom_site": resolved_site,
                "localization": resolved_site,
            }

        if USE_TTA:
            prediction = predictor.predict_with_tta(
                image=image_bytes,
                metadata=metadata_payload,
                tta_mode=TTA_MODE,
                aggregation=TTA_AGGREGATION,
                include_disclaimer=False,
            )
        else:
            prediction = predictor.predict(
                image=image_bytes,
                metadata=metadata_payload,
                include_disclaimer=False,
            )

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
