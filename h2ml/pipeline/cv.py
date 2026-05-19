"""
Pure cross-validation engine. Responsibilities:
    - Split data into folds (stratified for classification, standard for regression)
    - Fit each step per fold
    - Collect raw fold results (arrays + timing)
"""

from __future__ import annotations

from loguru import logger

import time
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any, Callable, Sequence, cast

if TYPE_CHECKING:
    from h2ml.features.spatial_cv import SpatialMetric

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from h2ml.pipeline.base import TaskType, PredictorStep

# ---------------------------------------------------------------------------
# FoldResult — raw output of a single fold
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """
    Stores the raw arrays produced in one CV fold.
    Metric computation happens downstream in evaluation/metrics.py.
    """

    fold_id: int
    model_name: str

    # Ground truth
    y_train: np.ndarray
    y_test: np.ndarray

    # Predictions
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray

    # Probabilities — None for regressors
    y_prob_train: Optional[np.ndarray] = None
    y_prob_test: Optional[np.ndarray] = None

    # Indices — useful for traceability / saving test data per fold
    train_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    test_idx: np.ndarray = field(default_factory=lambda: np.array([]))

    # Timing
    fit_time: float = 0.0

    @property
    def task_type(self) -> str:
        """Inferred from whether probabilities are present."""
        return "classification" if self.y_prob_test is not None else "regression"


# ---------------------------------------------------------------------------
# CVResult — aggregates all folds for one model
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """All fold results for a single model run."""

    model_name: str
    task_type: TaskType
    folds: list[FoldResult] = field(default_factory=list)
    failed_folds: list[int] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def fold_arrays(self, attr: str) -> list[np.ndarray]:
        """Helper to extract a single attribute across all folds."""
        return [getattr(f, attr) for f in self.folds]

    @property
    def oof_predictions(self) -> Optional[np.ndarray]:
        """
        Out-of-fold predictions assembled across all successful folds.

        Classification: positive-class probability (binary) or full probability
        matrix (multiclass). Regression: predicted values.
        Positions not covered by any fold (e.g. failed folds) are NaN.
        """
        if not self.folds:
            return None
        n_samples = int(max(f.test_idx.max() for f in self.folds)) + 1
        first = self.folds[0]
        if first.y_prob_test is not None:
            shape = (n_samples, first.y_prob_test.shape[1]) if first.y_prob_test.ndim == 2 else (n_samples,)
            oof = np.full(shape, np.nan)
            for f in self.folds:
                oof[f.test_idx] = f.y_prob_test
        else:
            oof = np.full(n_samples, np.nan)
            for f in self.folds:
                oof[f.test_idx] = f.y_pred_test
        return oof

    @property
    def oof_labels(self) -> Optional[np.ndarray]:
        """True labels paired with oof_predictions (original scale).

        Numeric labels use a float array with NaN for unfilled positions.
        String / object labels use an object array with None for unfilled positions.
        """
        if not self.folds:
            return None
        n_samples = int(max(f.test_idx.max() for f in self.folds)) + 1
        first_dtype = self.folds[0].y_test.dtype
        if np.issubdtype(first_dtype, np.number):
            oof = np.full(n_samples, np.nan)
        else:
            oof = np.empty(n_samples, dtype=object)
            oof[:] = None
        for f in self.folds:
            oof[f.test_idx] = f.y_test
        return oof


# ---------------------------------------------------------------------------
# CrossValidator
# ---------------------------------------------------------------------------


class CrossValidator:
    """
    Fits a step (classifier or regressor) across K folds and returns
    raw FoldResult objects.

    Args:
        n_splits:   Number of CV folds (default 5).
        shuffle:    Whether to shuffle before splitting (default True).
        random_state: Seed for reproducibility (default 42).
        verbose:    Print fold progress (default False).
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        step: PredictorStep,
        X: np.ndarray,
        y: np.ndarray,
        best_params: Optional[dict] = None,
        y_true: Optional[np.ndarray] = None,
        inverse_fn: Optional[Callable] = None,
        coords: Optional[np.ndarray] = None,
        n_blocks_per_fold: int = 5,
        spatial_cv_method: str = "block",
        ahc_threshold: Optional[float] = None,
        spatial_cv_metric: SpatialMetric = "euclidean",
        pca_components: float = 0.95,
        exact_max_samples: int = 5_000,
        knn_neighbors: int = 15,
        _splitter=None,
        _scaled_folds: Optional[list] = None,
    ) -> CVResult:
        """
        Run K-fold CV for a single step.

        Args:
            step:        A fitted or unfitted BaseClassifier / BaseRegressor.
            X:           Feature array (unscaled — scaling handled here if needed).
            y:           Target array. If a transform was applied, pass the transformed y here.
            best_params: If provided, re-instantiates the model with these params
                         each fold (used in the optimized run).
            y_true:      Original-scale target array. When provided alongside inverse_fn,
                         metrics are computed on the original scale (back-transformed).
            inverse_fn:  Callable that back-transforms predictions to the original scale.
                         Applied after predict(), before metric computation.
            _splitter:   Pre-built splitter to reuse across models (internal, set by run_all).

        Returns:
            CVResult with one FoldResult per fold.
        """
        if _splitter is not None:
            splitter = _splitter
        else:
            splitter = self._build_splitter(
                step.task_type,
                X=X,
                y=y,
                coords=coords,
                n_blocks_per_fold=n_blocks_per_fold,
                spatial_cv_method=spatial_cv_method,
                ahc_threshold=ahc_threshold,
                spatial_cv_metric=spatial_cv_metric,
                pca_components=pca_components,
                exact_max_samples=exact_max_samples,
                knn_neighbors=knn_neighbors,
            )
        cv_result = CVResult(model_name=step.name, task_type=step.task_type)

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            if test_idx.size == 0:
                logger.warning(
                    f"[{step.name}] Fold {fold_id} skipped: empty test set (spatial split produced no test samples)."
                )
                cv_result.failed_folds.append(fold_id)
                continue

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Ground truth for metrics — original scale if y_true provided, else y
            y_train_true = y_true[train_idx] if y_true is not None else y_train
            y_test_true = y_true[test_idx] if y_true is not None else y_test

            # Use pre-computed scaled arrays when available; otherwise fit scaler here.
            if _scaled_folds is not None:
                X_train, X_test = _scaled_folds[fold_id]
            else:
                X_train, X_test = self._maybe_scale(step, X_train, X_test)

            # Re-instantiate with best_params if provided (optimized setting)
            fold_step = self._build_fold_model(step, best_params)

            try:
                start = time.time()
                fold_step.fit(X_train, y_train)
                fit_time = time.time() - start
                fold_result = self._collect_fold(
                    fold_step,
                    fold_id,
                    X_train,
                    X_test,
                    y_train_true,
                    y_test_true,
                    train_idx,
                    test_idx,
                    fit_time,
                    inverse_fn=inverse_fn,
                )
                cv_result.folds.append(fold_result)
            except Exception as e:
                logger.warning(f"[{step.name}] Fold {fold_id} skipped: {e}")
                cv_result.failed_folds.append(fold_id)

        if not cv_result.folds:
            logger.warning(f"[{step.name}] All folds failed — model excluded from results.")

        return cv_result

    def run_all(
        self,
        steps: Sequence[PredictorStep],
        X: np.ndarray,
        y: np.ndarray,
        best_params: Optional[dict] = None,
        n_jobs: int = -1,
        y_true: Optional[np.ndarray] = None,
        inverse_fn: Optional[Callable] = None,
        coords: Optional[np.ndarray] = None,
        n_blocks_per_fold: int = 5,
        spatial_cv_method: str = "block",
        ahc_threshold: Optional[float] = None,
        spatial_cv_metric: SpatialMetric = "euclidean",
        pca_components: float = 0.95,
        exact_max_samples: int = 5_000,
        knn_neighbors: int = 15,
        _splitter: Any = None,
    ) -> list[CVResult]:
        """
        Run CV for a list of steps. Returns one CVResult per step.

        Args:
            steps:       List of BaseClassifier or BaseRegressor instances.
            X:           Feature array.
            y:           Target array (possibly transformed).
            best_params: Optional dict keyed by model name with their best params.
            n_jobs:      Number of parallel workers. -1 = one thread per CPU.
            y_true:      Original-scale target array for back-transformed metrics.
            inverse_fn:  Callable to back-transform predictions before metric computation.
            _splitter:   Pre-built splitter to reuse across pipeline steps. When provided,
                         splitter construction is skipped entirely.
        """
        # Use a pre-built splitter when the pipeline provides one; otherwise build here.
        # Spatial splitters are expensive (AHC / k-means); reusing across steps 1, 3, 4
        # avoids redundant clustering of identical coordinate sets.
        if _splitter is not None:
            prebuilt_splitter = _splitter
        elif coords is not None and steps:
            prebuilt_splitter = self._build_splitter(
                steps[0].task_type,
                X=X,
                y=y,
                coords=coords,
                n_blocks_per_fold=n_blocks_per_fold,
                spatial_cv_method=spatial_cv_method,
                ahc_threshold=ahc_threshold,
                spatial_cv_metric=spatial_cv_metric,
                pca_components=pca_components,
                exact_max_samples=exact_max_samples,
                knn_neighbors=knn_neighbors,
            )
        else:
            prebuilt_splitter = None

        def _run_one(step: PredictorStep) -> CVResult:
            import warnings

            warnings.filterwarnings("ignore")
            params = (best_params or {}).get(step.name)
            return self.run(
                step,
                X,
                y,
                best_params=params,
                y_true=y_true,
                inverse_fn=inverse_fn,
                coords=coords,
                n_blocks_per_fold=n_blocks_per_fold,
                spatial_cv_method=spatial_cv_method,
                ahc_threshold=ahc_threshold,
                spatial_cv_metric=spatial_cv_metric,
                pca_components=pca_components,
                exact_max_samples=exact_max_samples,
                knn_neighbors=knn_neighbors,
                _splitter=prebuilt_splitter,
            )

        if n_jobs == 1:
            return [_run_one(step) for step in steps]

        return cast(
            list[CVResult],
            Parallel(n_jobs=n_jobs, backend="loky")(delayed(_run_one)(step) for step in steps),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_splitter(
        self,
        task_type: TaskType,
        coords: Optional[np.ndarray] = None,
        n_blocks_per_fold: int = 5,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        spatial_cv_method: str = "block",
        ahc_threshold: Optional[float] = None,
        spatial_cv_metric: SpatialMetric = "euclidean",
        pca_components: float = 0.95,
        exact_max_samples: int = 5_000,
        knn_neighbors: int = 15,
    ):
        """Spatial splitter when coords provided, otherwise stratified/standard KFold."""
        if coords is not None:
            if spatial_cv_method == "spcv":
                from h2ml.features.spatial_cv import SPCVSplitter

                assert X is not None and y is not None
                return SPCVSplitter(
                    coords=coords,
                    X=X,
                    y=y,
                    n_splits=self.n_splits,
                    threshold=ahc_threshold,
                    random_state=self.random_state,
                    metric=spatial_cv_metric,
                    pca_components=pca_components,
                    exact_max_samples=exact_max_samples,
                    knn_neighbors=knn_neighbors,
                )
            from h2ml.features.spatial_cv import SpatialBlockSplitter

            return SpatialBlockSplitter(
                coords=coords,
                n_splits=self.n_splits,
                n_blocks_per_fold=n_blocks_per_fold,
                random_state=self.random_state,
                metric=spatial_cv_metric,
            )
        kwargs: dict[str, Any] = dict(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        if task_type == TaskType.CLASSIFICATION:
            return StratifiedKFold(**kwargs)
        return KFold(**kwargs)

    def _maybe_scale(
        self,
        step: PredictorStep,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply StandardScaler if the step declares requires_scaling = True.
        Scaler is always fit on X_train only to prevent data leakage.
        """
        if not getattr(step, "requires_scaling", False):
            return X_train, X_test

        # cols  = X_train.columns
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def _build_fold_model(
        self,
        step: PredictorStep,
        best_params: Optional[dict],
    ) -> PredictorStep:
        """
        If best_params provided, return a fresh instance with those params.
        Otherwise return the step as-is (default params).
        """
        if best_params is None:
            return step

        new_estimator = step.estimator.__class__(**best_params)

        return step.__class__(
            estimator=new_estimator,  # type: ignore
            name=step.name,  # type: ignore
            requires_scaling=step.requires_scaling,  # type: ignore
        )

    def _collect_fold(
        self,
        step: PredictorStep,
        fold_id: int,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        fit_time: float,
        inverse_fn: Optional[Callable] = None,
    ) -> FoldResult:
        """Collect predictions and build a FoldResult.

        y_train and y_test are expected to be original-scale ground truth.
        If inverse_fn is provided, predictions are back-transformed before
        being stored so metrics are computed on the original scale.
        """
        y_pred_test = step.predict(X_test)
        y_pred_train = step.predict(X_train)

        if inverse_fn is not None:
            y_pred_test = inverse_fn(y_pred_test)
            y_pred_train = inverse_fn(y_pred_train)

        # Probabilities only for classifiers
        y_prob_test = None
        y_prob_train = None
        if step.task_type == TaskType.CLASSIFICATION:
            proba_test = step.predict_proba(X_test)
            proba_train = step.predict_proba(X_train)
            # Binary: keep only the positive-class column (1D)
            # Multiclass: keep the full probability matrix (2D)
            if proba_test.shape[1] == 2:
                y_prob_test = proba_test[:, 1]
                y_prob_train = proba_train[:, 1]
            else:
                y_prob_test = proba_test
                y_prob_train = proba_train

        return FoldResult(
            fold_id=fold_id,
            model_name=step.name,
            y_train=y_train,
            y_test=y_test,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            y_prob_train=y_prob_train,
            y_prob_test=y_prob_test,
            train_idx=train_idx,
            test_idx=test_idx,
            fit_time=fit_time,
        )
