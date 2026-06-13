"""
h2ml/features/correlation.py

Correlation-based feature removal.

Removes features that are highly correlated with a more important feature,
using SHAP importance as the ranking criterion.

Responsibilities:
    - Compute Pearson, Spearman and Kendall correlation matrices
    - Iterate features in SHAP importance order
    - Remove features that exceed the correlation threshold with a
      higher-ranked feature in ANY of the three methods

What this module does NOT do:
    - SHAP computation          → features/shap_importance.py
    - Scaling                   → CV engine / PipelineStep
    - Pipeline orchestration    → features/selector.py
"""

from __future__ import annotations

from loguru import logger
from typing import Literal, Optional, cast

import pandas as pd


CorrelationMethod = Literal["pearson", "spearman", "kendall"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def remove_correlated_features(
    X: pd.DataFrame,
    feature_importance: pd.Series,
    corr_threshold: float = 0.7,
    methods: Optional[list[CorrelationMethod]] = None,
) -> list[str]:
    """
    Return a list of selected (uncorrelated) features ranked by SHAP importance.

    Features are evaluated in descending SHAP importance order. A feature is
    removed if its absolute correlation with any already-selected higher-ranked
    feature exceeds corr_threshold in ANY of the specified methods.

    Args:
        X:                  DataFrame of features (unscaled — correlation is
                            computed on raw values).
        feature_importance: pd.Series indexed by feature name, sorted descending.
                            Typically the output of get_shap_values().
        corr_threshold:     Absolute correlation threshold (default 0.7).
        methods:            Correlation methods to apply. A feature is removed
                            if it exceeds the threshold in any one of them.

    Returns:
        List of selected feature names in importance order.

    Raises:
        ValueError: If corr_threshold is not in the half-open interval (0, 1].
    """
    if methods is None:
        methods = ["pearson", "spearman", "kendall"]
    if not 0 < corr_threshold <= 1:
        raise ValueError(f"corr_threshold must be in (0, 1], got {corr_threshold}")

    # Pearson/Spearman are cheap matrix ops, so precompute them in full. Kendall's
    # full matrix is O(p² n²) and dominates everything else, so compute it lazily
    # per pair (O(n log n) via scipy) — only for pairs that survive the cheaper
    # screens and are actually reached, with results cached.
    eager_methods: list[CorrelationMethod] = [m for m in methods if m != "kendall"]
    corr_matrices: dict[str, pd.DataFrame] = {method: X.corr(method=method) for method in eager_methods}
    use_kendall = "kendall" in methods

    kendall_cache: dict[tuple, float] = {}

    def _kendall_abs(a, b) -> float:
        key = (a, b) if a <= b else (b, a)
        val = kendall_cache.get(key)
        if val is None:
            from scipy.stats import kendalltau

            tau = kendalltau(X[a].to_numpy(), X[b].to_numpy())[0]
            val = abs(cast(float, tau))
            kendall_cache[key] = val
        return val

    selected_features: list[str] = []
    removed_features: set[str] = set()

    n_original = len(feature_importance)

    for feature in feature_importance.index:
        if feature in removed_features:
            continue

        selected_features.append(feature)

        # Check all remaining features against the current selected feature
        for other in X.columns:
            if other == feature or other in removed_features:
                continue

            # Remove if correlation exceeds threshold in any method. Cheap methods
            # are screened first; Kendall is only computed when they don't already
            # flag the pair.
            correlated = any(
                abs(cast(float, corr_matrices[method].loc[feature, other])) > corr_threshold for method in eager_methods
            )
            if not correlated and use_kendall:
                correlated = _kendall_abs(feature, other) > corr_threshold

            if correlated:
                removed_features.add(other)

    n_removed = n_original - len(selected_features)
    logger.info(
        f"Feature selection: {len(selected_features)}/{n_original} kept, "
        f"{n_removed} removed (threshold={corr_threshold})"
    )

    return selected_features
