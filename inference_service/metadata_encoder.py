"""Metadata encoding utilities for inference-time metadata fusion."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch


class MetadataEncoder:
    """Encode tabular metadata to the tensor format expected by fusion models."""

    def __init__(
        self,
        age_mean: Optional[float],
        age_std: Optional[float],
        sex_categories: list[str],
        localization_categories: list[str],
        age_column: str,
        sex_column: str,
        localization_column: str,
        default_age: float,
        default_sex: str,
        default_localization: str,
    ):
        self.age_mean = age_mean
        self.age_std = age_std
        self.sex_categories = sex_categories
        self.localization_categories = localization_categories
        self.age_column = age_column
        self.sex_column = sex_column
        self.localization_column = localization_column
        self.default_age = default_age
        self.default_sex = default_sex
        self.default_localization = default_localization

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        return False

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "MetadataEncoder":
        return cls(
            age_mean=state.get("age_mean"),
            age_std=state.get("age_std"),
            sex_categories=list(state.get("sex_categories", ["unknown"])),
            localization_categories=list(
                state.get("localization_categories", ["unknown"])
            ),
            age_column=state.get("age_column", "age"),
            sex_column=state.get("sex_column", "sex"),
            localization_column=state.get("localization_column", "localization"),
            default_age=float(state.get("default_age", 50.0)),
            default_sex=str(state.get("default_sex", "unknown")),
            default_localization=str(state.get("default_localization", "unknown")),
        )

    def get_metadata_dim(self) -> int:
        return 1 + len(self.sex_categories) + len(self.localization_categories)

    def encode_metadata_dict(self, metadata: Dict[str, Any]) -> torch.Tensor:
        features: list[float] = []

        age_value = metadata.get(self.age_column)
        if self._is_missing(age_value):
            age = self.default_age
        else:
            try:
                age = float(age_value)
            except (TypeError, ValueError):
                age = self.default_age

        if self.age_std and self.age_std > 0 and self.age_mean is not None:
            age_normalized = (age - self.age_mean) / self.age_std
        else:
            age_normalized = 0.0
        features.append(float(age_normalized))

        sex_value = metadata.get(self.sex_column)
        if self._is_missing(sex_value):
            sex_value = self.default_sex
        sex_value = str(sex_value).lower()
        features.extend(
            [1.0 if str(cat).lower() == sex_value else 0.0 for cat in self.sex_categories]
        )

        localization_value = metadata.get(self.localization_column)
        if self._is_missing(localization_value):
            localization_value = self.default_localization
        localization_value = str(localization_value).lower()
        features.extend(
            [
                1.0 if str(cat).lower() == localization_value else 0.0
                for cat in self.localization_categories
            ]
        )

        return torch.tensor(features, dtype=torch.float32)
