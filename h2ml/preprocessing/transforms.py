"""
h2ml/preprocessing/transforms.py

Y-transform registry for the regression y-transform sweep (steps 1 and 3).

Y_TRANSFORMS:       dict[name → callable(y) -> ndarray | None]
                    A None return value signals "no outliers found" and causes
                    build_transform_stores() to silently skip the transform.

INVERSE_TRANSFORMS: dict[name → callable(y_pred) -> ndarray]
                    Used to back-transform predictions to the original scale.
                    Winsorize clipping is not reversible; only the underlying
                    log/sqrt step is inverted.

Available names: count, log, sqrt, wincount, winlog, winsqrt.
Winsorize-based transforms replace upper outliers with the IQR upper limit.
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np


def log_transform(y: np.ndarray) -> np.ndarray:
    """Apply log1p to y. Inverted by expm1 in INVERSE_TRANSFORMS."""
    return np.log1p(y)


def sqrt_transform(y: np.ndarray) -> np.ndarray:
    """Apply the square root to y. Inverted by squaring in INVERSE_TRANSFORMS."""
    return np.sqrt(y)


def winsorize(y: np.ndarray) -> Optional[np.ndarray]:
    """
    Replaces upper outliers with IQR upper limit.
    Returns None if no outliers found — caller should skip this transform.
    """
    Q1, Q3 = np.percentile(y, [25, 75])
    upper_limit = Q3 + 1.5 * (Q3 - Q1)

    if (y > upper_limit).sum() == 0:
        return None

    return np.where(y > upper_limit, upper_limit, y)


# Transform registry — extend freely
Y_TRANSFORMS = {
    "count": lambda y: y,
    "log": log_transform,
    "sqrt": sqrt_transform,
    "wincount": winsorize,
    "winlog": lambda y: winsorize(log_transform(y)),
    "winsqrt": lambda y: winsorize(sqrt_transform(y)),
}

# Inverse transform registry — mirrors Y_TRANSFORMS
# Winsorize clipping is not reversible; the log/sqrt step underneath is reversed instead.
INVERSE_TRANSFORMS: dict[str, Callable] = {
    "count": lambda y: y,
    "log": np.expm1,  # inverse of log1p
    "sqrt": np.square,  # inverse of sqrt
    "wincount": lambda y: y,
    "winlog": np.expm1,
    "winsqrt": np.square,
}
