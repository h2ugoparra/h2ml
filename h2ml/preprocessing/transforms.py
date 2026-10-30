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

PRE_CLIP_TRANSFORMS: dict[win* name → stateless part of the transform]
                    The IQR limit is fitted on the data, so in CV it must come from
                    the training rows only. Transform stores for win* names hold y
                    before clipping; get_train_target_fn() returns the clipping step
                    that each training fold (and the final refit) applies to itself.
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
    Returns None if no outliers found — caller should skip this transform. Also None
    when the limit sits at the minimum (e.g. >= 75 % zeros, so Q3 = Q1 = 0): clipping
    would flatten y to a constant.
    """
    Q1, Q3 = np.percentile(y, [25, 75])
    upper_limit = Q3 + 1.5 * (Q3 - Q1)

    if upper_limit <= np.min(y) or (y > upper_limit).sum() == 0:
        return None

    return np.where(y > upper_limit, upper_limit, y)


def clip_upper_outliers(y: np.ndarray) -> np.ndarray:
    """Winsorize y with a limit fitted on y itself; y is returned unchanged when it has no outliers."""
    clipped = winsorize(y)
    return y if clipped is None else clipped


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

# Stateless part of each winsorize-based transform — safe to apply before splitting.
# Y_TRANSFORMS[name](y) == clip_upper_outliers(PRE_CLIP_TRANSFORMS[name](y)) when y has outliers.
PRE_CLIP_TRANSFORMS: dict[str, Callable] = {
    "wincount": lambda y: y,
    "winlog": log_transform,
    "winsqrt": sqrt_transform,
}


def get_train_target_fn(y_transform: Optional[str]) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Return the step fitted on and applied to training targets for y_transform, or None if stateless."""
    return clip_upper_outliers if y_transform in PRE_CLIP_TRANSFORMS else None
