"""Swin model adapter for the Dermalyze inference service."""

from .predict import HF_MODEL_ID, LOCAL_MODEL_DIR, SwinSkinLesionPredictor

__all__ = ["HF_MODEL_ID", "LOCAL_MODEL_DIR", "SwinSkinLesionPredictor"]
