"""
h2ml/evaluation/applicability.py

Area of applicability (AOA) — Meyer & Pebesma (2021), "Predicting into unknown
space? Estimating the area of applicability of spatial prediction models",
Methods in Ecology and Evolution 12:1620–1633.

The dissimilarity index (DI) of a new point is its distance to the nearest
training point in importance-weighted, standardised feature space, divided by
the mean pairwise distance between training points. The AOA threshold is the
upper boxplot whisker of the training points' own DI, each measured against
the training rows outside its CV fold — so it reflects how far CV already had
to extrapolate. Points above the threshold lie in feature combinations the
model never learned from; their predictions (and conformal intervals) are not
backed by the CV estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

# Training rows used to estimate the mean pairwise distance — pdist is O(n²).
MAX_PAIRWISE_ROWS = 2_000


@dataclass
class ApplicabilityDomain:
    """
    Fitted area of applicability for one model's feature space.

    Attributes:
        feature_names: Features in model order.
        mean, std:     Standardisation fitted on the training rows.
        weights:       Per-feature weights (normalised to sum to 1) — SHAP importance
                       when available, else equal.
        mean_distance: Mean pairwise distance between weighted, standardised training rows.
        threshold:     DI above which a point is outside the AOA.
        train_di:      Cross-validated DI of each training row (diagnostic).
        nn:            NearestNeighbors fitted on the weighted, standardised training rows.
    """

    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    mean_distance: float
    threshold: float
    train_di: np.ndarray
    nn: NearestNeighbors

    def dissimilarity(self, X: np.ndarray) -> np.ndarray:
        """DI of each row of X (columns in feature_names order)."""
        dist, _ = self.nn.kneighbors(self._embed(np.asarray(X, dtype=float)), n_neighbors=1)
        return dist[:, 0] / self.mean_distance

    def inside(self, X: np.ndarray) -> np.ndarray:
        """Boolean mask: True where the row lies inside the area of applicability."""
        return self.dissimilarity(X) <= self.threshold

    def _embed(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std * self.weights

    def __repr__(self) -> str:
        share = float(np.mean(self.train_di <= self.threshold))
        return (
            f"ApplicabilityDomain(n_features={len(self.feature_names)}, threshold={self.threshold:.3f}, "
            f"train_inside={share:.1%})"
        )


def build_applicability_domain(
    X: np.ndarray,
    feature_names: list[str],
    fold_ids: np.ndarray,
    weights: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> ApplicabilityDomain:
    """
    Fit an ApplicabilityDomain on the training rows and their CV fold assignment.

    Args:
        X:            (n, p) training features in feature_names order.
        feature_names: Feature names.
        fold_ids:     (n,) CV test-fold id of each row; rows with id < 0 (never held
                      out) are kept as neighbours but get no training DI.
        weights:      (p,) non-negative feature weights; None or all-zero → equal.
        random_state: Seed for subsampling the mean pairwise distance.

    Returns:
        ApplicabilityDomain.

    Raises:
        ValueError: If shapes disagree or fewer than two folds hold rows.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if len(feature_names) != p or len(fold_ids) != n:
        raise ValueError(f"X is {X.shape} but got {len(feature_names)} feature names and {len(fold_ids)} fold ids.")
    folds = np.unique(fold_ids[fold_ids >= 0])
    if folds.size < 2:
        raise ValueError("At least two CV folds are needed to calibrate the AOA threshold.")

    w = np.ones(p) if weights is None else np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if not np.isfinite(w).all() or w.sum() == 0:
        w = np.ones(p)
    w = w / w.sum()
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std * w

    rng = np.random.default_rng(random_state)
    sample = Z if n <= MAX_PAIRWISE_ROWS else Z[rng.choice(n, MAX_PAIRWISE_ROWS, replace=False)]
    mean_distance = float(pdist(sample).mean())
    if mean_distance == 0:
        raise ValueError("All training rows are identical in the weighted feature space.")

    train_di = []
    for k in folds:
        test, train = fold_ids == k, fold_ids != k
        dist, _ = NearestNeighbors(n_neighbors=1).fit(Z[train]).kneighbors(Z[test])
        train_di.append(dist[:, 0] / mean_distance)
    train_di = np.concatenate(train_di)

    # Upper boxplot whisker: the largest training DI within Q3 + 1.5 IQR.
    q1, q3 = np.percentile(train_di, [25, 75])
    threshold = float(train_di[train_di <= q3 + 1.5 * (q3 - q1)].max())

    return ApplicabilityDomain(
        feature_names=list(feature_names),
        mean=mean,
        std=std,
        weights=w,
        mean_distance=mean_distance,
        threshold=threshold,
        train_di=train_di,
        nn=NearestNeighbors(n_neighbors=1).fit(Z),
    )
