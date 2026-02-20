"""Shared TTA configuration constants used across evaluation and inference."""

from __future__ import annotations

from typing import Dict, Literal


TTAMode = Literal["light", "medium", "full"]

TTA_AUG_COUNTS: Dict[TTAMode, int] = {
    "light": 4,
    "medium": 8,
    "full": 12,
}
