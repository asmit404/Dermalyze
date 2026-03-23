"""
Metadata encoding and preprocessing utilities for HAM10000 dataset.

This module provides tools for processing patient metadata (age, sex, anatomical site)
to be used as additional features in multi-input models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch


class MetadataEncoder:
    """
    Encoder for HAM10000 patient metadata.

    Handles encoding of categorical variables (sex, localization) and
    normalization of continuous variables (age). Supports missing value imputation.
    """

    def __init__(
        self,
        age_mean: Optional[float] = None,
        age_std: Optional[float] = None,
        sex_categories: Optional[list[str]] = None,
        localization_categories: Optional[list[str]] = None,
    ):
        """
        Initialize the metadata encoder.

        Args:
            age_mean: Mean age for normalization (computed from training if None)
            age_std: Standard deviation of age (computed from training if None)
            sex_categories: List of sex categories (computed from training if None)
            localization_categories: List of localization categories (computed from training if None)
        """
        self.age_mean = age_mean
        self.age_std = age_std
        self.sex_categories = sex_categories
        self.localization_categories = localization_categories

        # Default values for missing data
        self.default_age = 50.0  # Will be updated during fit
        self.default_sex = 'unknown'
        self.default_localization = 'unknown'

    def fit(self, df: pd.DataFrame) -> MetadataEncoder:
        """
        Fit the encoder on training data.

        Args:
            df: Training DataFrame with metadata columns

        Returns:
            Self for method chaining
        """
        # Compute age statistics (ignoring NaN)
        if 'age' in df.columns:
            age_values = df['age'].dropna()
            if len(age_values) > 0:
                self.age_mean = float(age_values.mean())
                self.age_std = float(age_values.std())
                self.default_age = self.age_mean
            else:
                # Fallback if all ages are missing
                self.age_mean = 50.0
                self.age_std = 15.0
                self.default_age = 50.0

        # Get unique sex categories
        if 'sex' in df.columns:
            sex_values = df['sex'].dropna().unique().tolist()
            self.sex_categories = sorted(sex_values) + ['unknown']
        else:
            self.sex_categories = ['unknown']

        # Get unique localization categories
        if 'localization' in df.columns:
            loc_values = df['localization'].dropna().unique().tolist()
            self.localization_categories = sorted(loc_values) + ['unknown']
        else:
            self.localization_categories = ['unknown']

        return self

    def encode_metadata_dict(self, metadata: dict) -> torch.Tensor:
        """
        Encode a single metadata dictionary to a tensor.

        Args:
            metadata: Dictionary with keys 'age', 'sex', 'localization'

        Returns:
            Encoded metadata tensor
        """
        features = []

        # Encode age (normalized)
        age = metadata.get('age')
        if age is None or pd.isna(age):
            age = self.default_age
        else:
            age = float(age)

        # Normalize age
        if self.age_std > 0:
            age_normalized = (age - self.age_mean) / self.age_std
        else:
            age_normalized = 0.0
        features.append(age_normalized)

        # Encode sex (one-hot)
        sex = metadata.get('sex')
        if sex is None or pd.isna(sex):
            sex = self.default_sex
        sex = str(sex).lower()

        sex_encoded = [1.0 if cat.lower() == sex else 0.0 for cat in self.sex_categories]
        features.extend(sex_encoded)

        # Encode localization (one-hot)
        localization = metadata.get('localization')
        if localization is None or pd.isna(localization):
            localization = self.default_localization
        localization = str(localization).lower()

        loc_encoded = [1.0 if cat.lower() == localization else 0.0
                       for cat in self.localization_categories]
        features.extend(loc_encoded)

        return torch.tensor(features, dtype=torch.float32)

    def get_metadata_dim(self) -> int:
        """
        Get the dimension of encoded metadata features.

        Returns:
            Total feature dimension (1 for age + len(sex_categories) + len(localization_categories))
        """
        return 1 + len(self.sex_categories) + len(self.localization_categories)

    def get_feature_names(self) -> list[str]:
        """
        Get names of all features in order.

        Returns:
            List of feature names
        """
        names = ['age_normalized']
        names.extend([f'sex_{cat}' for cat in self.sex_categories])
        names.extend([f'loc_{cat}' for cat in self.localization_categories])
        return names

    def save_state(self) -> dict:
        """
        Save encoder state for serialization.

        Returns:
            Dictionary containing encoder parameters
        """
        return {
            'age_mean': self.age_mean,
            'age_std': self.age_std,
            'sex_categories': self.sex_categories,
            'localization_categories': self.localization_categories,
            'default_age': self.default_age,
            'default_sex': self.default_sex,
            'default_localization': self.default_localization,
        }

    @classmethod
    def from_state(cls, state: dict) -> MetadataEncoder:
        """
        Create encoder from saved state.

        Args:
            state: Dictionary containing encoder parameters

        Returns:
            Initialized MetadataEncoder
        """
        encoder = cls(
            age_mean=state['age_mean'],
            age_std=state['age_std'],
            sex_categories=state['sex_categories'],
            localization_categories=state['localization_categories'],
        )
        encoder.default_age = state.get('default_age', 50.0)
        encoder.default_sex = state.get('default_sex', 'unknown')
        encoder.default_localization = state.get('default_localization', 'unknown')
        return encoder
