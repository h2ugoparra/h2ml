from __future__ import annotations
from typing import Optional
import numpy as np
from h2ml.features.feature_store import PipelineData
from h2ml.preprocessing.transforms import Y_TRANSFORMS


def build_transform_stores(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    transforms: Optional[list[str]] = None,
    coords: Optional[np.ndarray] = None,
) -> dict[str, PipelineData]:
    """
    Build one PipelineData per y transformation.
    Transforms returning None (e.g. winsorize with no outliers) are skipped.

    Args:
        X:             Feature matrix.
        y:             Raw target array.
        feature_names: Feature names for PipelineData.
        transforms:    List of transform names from Y_TRANSFORMS.
                       Defaults to all registered transforms.

    Returns:
        Dict mapping transform name → PipelineData.
    """
    transforms = list(Y_TRANSFORMS.keys()) if transforms is None else transforms
    stores = {}

    for name in transforms:
        if name not in Y_TRANSFORMS:
            raise KeyError(
                f"Unknown transform '{name}'. Available: {list(Y_TRANSFORMS)}"
            )

        y_transformed = Y_TRANSFORMS[name](y)

        if y_transformed is None:
            continue

        stores[name] = PipelineData(
            X=X,
            feature_names=feature_names,
            y=y_transformed,
            y_true=y,
            y_transform=name,
            coords=coords,
        )

    return stores
