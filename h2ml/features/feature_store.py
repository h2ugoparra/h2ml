"""
h2ml/features/feature_store.py

PipelineData — travels alongside numpy arrays to preserve feature names
throughout the pipeline. Converts to DataFrame locally when needed (SHAP,
correlation, plots) without converting the whole pipeline back to pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PipelineData:
    """
    Container that pairs a numpy array with its feature names.

    Args:
        X:             Feature matrix as numpy array (n_samples, n_features).
        feature_names: Ordered list of feature names matching X columns.
        y:             Target array (n_samples,).

    Example:
        >>> store = PipelineData(X=X_arr, feature_names=df.columns.tolist(), y=y_arr)
        >>> store.to_frame()   # recover DataFrame locally when needed
        >>> store.n_features   # 10
    """

    X: np.ndarray
    feature_names: list[str]
    y: np.ndarray
    y_true: Optional[np.ndarray] = None  # original-scale y for back-transformed metrics
    y_transform: Optional[str] = None  # transform name, used to look up inverse fn
    coords: Optional[np.ndarray] = (
        None  # spatial coordinates (n_samples, 2) — lat/lon or x/y
    )

    def __post_init__(self):
        n = self.X.shape[0]
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {self.X.shape[1]} columns but "
                f"{len(self.feature_names)} feature names were provided."
            )
        if self.y.shape[0] != n:
            raise ValueError(f"y has {self.y.shape[0]} rows but X has {n} rows.")
        if self.y_true is not None and self.y_true.shape[0] != n:
            raise ValueError(
                f"y_true has {self.y_true.shape[0]} rows but X has {n} rows."
            )
        if self.coords is not None:
            if self.coords.ndim != 2 or self.coords.shape[1] != 2:
                raise ValueError(
                    f"coords must be shape (n_samples, 2), got {self.coords.shape}."
                )
            if self.coords.shape[0] != n:
                raise ValueError(
                    f"coords has {self.coords.shape[0]} rows but X has {n} rows."
                )

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Recover a DataFrame with correct column names."""
        return pd.DataFrame(self.X, columns=self.feature_names)

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        y: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        y_transform: Optional[str] = None,
        coords: Optional[np.ndarray] = None,
    ) -> "PipelineData":
        """Build a PipelineData directly from a DataFrame."""
        return cls(
            X=df.to_numpy(),
            feature_names=df.columns.tolist(),
            y=y,
            y_true=y_true,
            y_transform=y_transform,
            coords=coords,
        )

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def select(self, features: list[str]) -> "PipelineData":
        """
        Return a new PipelineData with only the selected features.
        Order follows the input features list.
        """
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        missing = set(features) - name_to_idx.keys()
        if missing:
            raise ValueError(f"Features not found in store: {missing}")

        idx = [name_to_idx[f] for f in features]
        return PipelineData(
            X=self.X[:, idx],
            feature_names=features,
            y=self.y,
            y_true=self.y_true,
            y_transform=self.y_transform,
            coords=self.coords,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def __repr__(self) -> str:
        return (
            f"PipelineData("
            f"n_samples={self.n_samples}, "
            f"n_features={self.n_features}, "
            f"features={self.feature_names[:3]}{'...' if self.n_features > 3 else ''})"
        )
