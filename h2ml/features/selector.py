"""
h2ml/features/selector.py

FeatureSelector — pipeline step that orchestrates SHAP importance ranking
and correlation-based feature removal.

Plugs into the pipeline between step 2 (best model selection) and
step 4 (CV on reduced features):

    store_reduced = FeatureSelector(best_step, task_type).fit_transform(store, y)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from h2ml.core.base import BasePreprocessor, TaskType
from h2ml.core.feature_store import PipelineData
from h2ml.features.correlation import remove_correlated_features
from h2ml.features.shap_importance import get_oof_shap_values, get_shap_values


class FeatureSelector(BasePreprocessor):
    """
    Pipeline step that reduces features using SHAP importance + correlation filtering.

    Fit phase:
        1. Computes out-of-fold (OOF) SHAP values — each sample's importance is derived
           from the fold where it was held out, so no training sample leaks into its own
           SHAP computation. Set use_oof=False to revert to full-dataset SHAP.
        2. Ranks features by mean absolute SHAP importance
        3. Removes features correlated above threshold with higher-ranked features
        4. Stores selected_features_ for use in transform

    Transform phase:
        - Returns a reduced PipelineData containing only selected features

    Args:
        step:           A ModelWrapper whose class and params seed each fold's model.
                        The selector handles fitting internally.
        task_type:      TaskType.CLASSIFICATION or TaskType.REGRESSION.
        corr_threshold: Correlation threshold for feature removal (default 0.7).
        min_features:   Minimum number of features to retain. If correlation filtering
                        would drop below this, the top SHAP-ranked removed features are
                        restored until the minimum is met (default 1 — no guard).
        use_oof:        If True (default), compute SHAP via out-of-fold CV to avoid
                        information leakage into feature selection. If False, fit on
                        the full training set (faster, mildly optimistic step 3 scores).
        n_splits:       Folds for the OOF CV pass (used only when use_oof=True).
        random_state:   Seed for OOF fold shuffle (used only when use_oof=True).
        shap_save_path: Optional path to save SHAP array as .npy.
        max_background: Max background samples for KernelSHAP (affects non-tree,
                        non-linear models such as SVC/SVR with rbf kernel). Larger
                        training folds are summarised with shap.kmeans to this size.
                        Default 100 — increase for more accurate SHAP at cost of speed.
        verbose:        Log selected/removed feature counts.
        _splitter:      Internal — a pre-built spatial splitter reused for the OOF
                        SHAP pass so it matches the pipeline's CV folds. Leave None
                        for non-spatial use; the pipeline injects it when relevant.

    Example:
        >>> selector = FeatureSelector(best_step, TaskType.CLASSIFICATION, corr_threshold=0.7)
        >>> store_reduced = selector.fit_transform(store, store.y)
        >>> store_reduced.feature_names   # reduced feature list
    """

    def __init__(
        self,
        step: Any,
        task_type: TaskType,
        corr_threshold: float = 0.7,
        min_features: int = 1,
        use_oof: bool = True,
        n_splits: int = 5,
        random_state: int = 42,
        shap_save_path: Optional[Path] = None,
        max_background: int = 100,
        verbose: bool = False,
        _splitter: Any = None,
    ):
        if min_features < 1:
            raise ValueError(f"min_features must be >= 1, got {min_features}")
        super().__init__(name="FeatureSelector", verbose=verbose)
        self.step = step
        self.task_type = task_type
        self.corr_threshold = corr_threshold
        self.min_features = min_features
        self.use_oof = use_oof
        self.n_splits = n_splits
        self.random_state = random_state
        self.shap_save_path = shap_save_path
        self.max_background = max_background
        self._splitter = _splitter

        # Set after fit
        self.selected_features_: list[str] = []
        self.feature_importance_: pd.Series = pd.Series(dtype=float)
        self.shap_values_: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # BasePreprocessor interface
    # ------------------------------------------------------------------

    def fit(
        self,
        store: PipelineData,
        y: Optional[np.ndarray] = None,
    ) -> "FeatureSelector":
        """
        Compute SHAP importance and determine selected features.

        Args:
            store: PipelineData containing the training data.
            y:     Unused — kept for API compatibility with BasePreprocessor.

        Returns:
            self, with shap_values_, feature_importance_ and selected_features_ set.
        """
        if self.use_oof:
            self.shap_values_, self.feature_importance_ = get_oof_shap_values(
                step=self.step,
                store=store,
                task_type=self.task_type,
                n_splits=self.n_splits,
                random_state=self.random_state,
                save_path=self.shap_save_path,
                max_background=self.max_background,
                _splitter=self._splitter,
            )
        else:
            # Full-dataset SHAP: fit once on all training data, then explain.
            # Slightly faster but introduces mild optimistic bias in step 3 scores
            # because the feature subset is chosen with knowledge of all samples.
            from sklearn.preprocessing import StandardScaler

            estimator_cls = self.step.estimator.__class__
            estimator_params = self.step.estimator.get_params()
            X_train = store.X
            if getattr(self.step, "requires_scaling", False):
                X_train = StandardScaler().fit_transform(X_train)
            model = estimator_cls(**estimator_params)
            model.fit(X_train, store.y)
            self.shap_values_, self.feature_importance_ = get_shap_values(
                model=model,
                store=store,
                task_type=self.task_type,
                save_path=self.shap_save_path,
                max_background=self.max_background,
            )

        # Remove correlated features in importance order
        self.selected_features_ = remove_correlated_features(
            X=store.to_frame(),
            feature_importance=self.feature_importance_,
            corr_threshold=self.corr_threshold,
        )

        # Enforce minimum — restore top SHAP-ranked features when threshold is too aggressive
        if len(self.selected_features_) < self.min_features:
            selected_set = set(self.selected_features_)
            restored = []
            for feature in self.feature_importance_.index:
                if feature not in selected_set:
                    restored.append(feature)
                    selected_set.add(feature)
                    if len(selected_set) >= self.min_features:
                        break
            self.selected_features_ = [f for f in self.feature_importance_.index if f in selected_set]
            logger.warning(
                f"FeatureSelector: corr_threshold={self.corr_threshold} reduced features to "
                f"{len(self.selected_features_) - len(restored)} — below min_features={self.min_features}. "
                f"Restored {len(restored)} top-SHAP feature(s): {restored}."
            )

        self._fitted = True

        return self

    def transform(self, store: PipelineData) -> PipelineData:
        """
        Return a new PipelineData with only the selected features.

        Args:
            store: PipelineData to reduce (can be train or test set).

        Returns:
            Reduced PipelineData preserving feature names and y.
        """
        self._check_fitted()
        return store.select(self.selected_features_)

    def fit_transform(
        self,
        store: PipelineData,
        y: Optional[np.ndarray] = None,
    ) -> PipelineData:
        """
        Fit and transform in one call.

        Args:
            store: PipelineData to fit the selector on and then reduce.
            y:     Unused — kept for API compatibility with BasePreprocessor.

        Returns:
            Reduced PipelineData containing only the selected features.
        """
        return self.fit(store, y).transform(store)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def n_selected(self) -> int:
        """Number of features retained after the correlation filter."""
        return len(self.selected_features_)

    @property
    def removed_features(self) -> list[str]:
        """Features that were dropped during fit."""
        self._check_fitted()
        return [f for f in self.feature_importance_.index if f not in self.selected_features_]

    def importance_summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame of all features with their SHAP importance
        and whether they were selected or removed.
        """
        self._check_fitted()
        return pd.DataFrame(
            {
                "feature": self.feature_importance_.index,
                "importance": self.feature_importance_.values,
                "selected": [f in self.selected_features_ for f in self.feature_importance_.index],
            }
        ).reset_index(drop=True)

    def __repr__(self) -> str:
        fitted_info = f"selected={self.n_selected}" if self._fitted else "not fitted"
        shap_mode = "oof" if self.use_oof else "full"
        return (
            f"FeatureSelector("
            f"task_type={self.task_type.value!r}, "
            f"corr_threshold={self.corr_threshold}, "
            f"min_features={self.min_features}, "
            f"shap={shap_mode}, "
            f"{fitted_info})"
        )
