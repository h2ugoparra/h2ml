"""
h2ml/pipeline/final_model.py

FinalModel — deployment artifact produced after a full pipeline run.

Responsibilities
----------------
- Fit the best model (with best_params) on the full training dataset
- Handle per-sample scaling when the model requires it
- Expose predict() / predict_proba() for inference on new data
- Save / load via joblib

Separation of concerns
-----------------------
PipelineResult  — evaluation artifact (CV metrics, SHAP, stage selection)
FinalModel      — deployment artifact (fitted estimator ready for prediction)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from h2ml.pipeline.pipeline import PipelineResult

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from h2ml.pipeline.base import TaskType


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------

@dataclass
class ConformalCalibration:
    """
    Nonconformity scores from out-of-fold CV predictions, used to construct
    finite-sample coverage-guaranteed prediction intervals (regression) or
    prediction sets (classification).

    Scores are pre-sorted ascending. The threshold at level 1-alpha is the
    ceil((1-alpha)(n+1))/n quantile, which guarantees marginal coverage ≥ 1-alpha.

    Attributes
    ----------
    scores:    Sorted nonconformity scores from calibration folds.
    n:         Number of calibration samples (len(scores)).
    task_type: TaskType of the calibrated model.
    """
    scores:    np.ndarray
    n:         int
    task_type: TaskType

    def threshold(self, alpha: float) -> float:
        """
        Return the nonconformity threshold that gives ≥ 1-alpha coverage.

        Args:
            alpha: Miscoverage level (e.g. 0.10 for 90% coverage).
        """
        level = min(np.ceil((1 - alpha) * (self.n + 1)) / self.n, 1.0)
        return float(np.quantile(self.scores, level))


@dataclass
class FinalModel:
    """
    Fitted model ready for inference on new data.

    Attributes
    ----------
    estimator:        Sklearn-compatible estimator fitted on the full training set.
    feature_names:    Ordered list of features the model was trained on.
    task_type:        TaskType.CLASSIFICATION or TaskType.REGRESSION.
    requires_scaling: Whether StandardScaler was applied before fitting.
    scaler:           Fitted StandardScaler (None when requires_scaling is False).
    best_model_name:  Name of the model as registered in the h2ml registry.
    best_params:      Hyperparameters used for the final fit (None = defaults).

    Example
    -------
    >>> final = result.build_final_model()
    >>> final.predict(X_new_df)
    >>> final.save("models/final_model.pkl")
    >>> final = FinalModel.load("models/final_model.pkl")
    """

    estimator:        Any
    feature_names:    list[str]
    task_type:        TaskType
    requires_scaling: bool                           = False
    scaler:           Optional[Any]                  = None
    best_model_name:  Optional[str]                  = None
    best_params:      Optional[dict]                 = field(default=None)
    conformal:        Optional[ConformalCalibration] = field(default=None)
    y_transform:      Optional[str]                  = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict on new data.

        Args:
            X: DataFrame (columns are aligned by name) or ndarray
                (columns must match feature_names order).

        Returns:
            1-D array of predictions.
        """
        return self.estimator.predict(self._prepare(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Returns:
            Binary:     1-D array of positive-class probabilities.
            Multiclass: 2-D array of shape (n_samples, n_classes).
        """
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks.")
        proba = self.estimator.predict_proba(self._prepare(X))
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def predict_interval(
        self,
        X:     pd.DataFrame | np.ndarray,
        alpha: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Conformal prediction interval for each sample (regression only).

        The interval is centred on the point estimate and has width 2q, where q
        is the conformal threshold calibrated from out-of-fold residuals.
        Coverage is guaranteed to be ≥ 1-alpha in expectation.

        Note: if the pipeline used a y-transform, the interval is in the
        transformed space. Apply the inverse transform to the bounds if needed.

        Args:
            X:     Input features (DataFrame or ndarray).
            alpha: Miscoverage level. Default 0.10 → 90% prediction intervals.

        Returns:
            (lower, upper) as a pair of 1-D arrays.
        """
        if self.task_type != TaskType.REGRESSION:
            raise ValueError("predict_interval is only available for regression tasks.")
        if self.conformal is None:
            raise ValueError(
                "No conformal calibration available. "
                "Rebuild FinalModel via result.build_final_model() after a full pipeline run."
            )
        y_hat = self.predict(X)
        q = self.conformal.threshold(alpha)
        return y_hat - q, y_hat + q

    def predict_set(
        self,
        X:     pd.DataFrame | np.ndarray,
        alpha: float = 0.10,
    ) -> list[np.ndarray]:
        """
        Conformal prediction set for each sample (classification only).

        Each prediction set contains the classes that are plausible given the
        coverage guarantee. A singleton set means the model is confident; a
        larger set means it is uncertain.

        Coverage guarantee: the true label is in the prediction set with
        probability ≥ 1-alpha.

        Nonconformity score per class k: 1 - p_k. Class k is included in the
        set when its score does not exceed the calibrated threshold q.

        Args:
            X:     Input features (DataFrame or ndarray).
            alpha: Miscoverage level. Default 0.10 → 90% coverage sets.

        Returns:
            List of arrays of class labels, one per sample.
        """
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_set is only available for classification tasks.")
        if self.conformal is None:
            raise ValueError(
                "No conformal calibration available. "
                "Rebuild FinalModel via result.build_final_model() after a full pipeline run."
            )
        p = self.predict_proba(X)   # 1-D for binary, 2-D for multiclass
        q = self.conformal.threshold(alpha)
        classes = self.estimator.classes_
        sets = []
        if p.ndim == 1:
            # Binary: p is positive-class probability; nonconformity for class 0 = p, for class 1 = 1-p
            for pi in p:
                labels = []
                if pi <= q:
                    labels.append(classes[0])
                if 1 - pi <= q:
                    labels.append(classes[1])
                sets.append(np.array(labels))
        else:
            # Multiclass: nonconformity for class k = 1 - p_k
            for row in p:
                labels = [classes[k] for k in range(len(classes)) if 1 - row[k] <= q]
                sets.append(np.array(labels))
        return sets

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist to *path* (single .pkl file via joblib). Parent directory is created if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FinalModel":
        """
        Reload a FinalModel saved with :meth:`save`.

        WARNING: only load files from trusted sources — joblib uses pickle,
        which executes arbitrary code on deserialisation.
        """
        return joblib.load(Path(path))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Align columns and apply scaling if required."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].to_numpy()
        X = np.asarray(X)
        if self.requires_scaling and self.scaler is not None:
            X = self.scaler.transform(X)
        return X  # type: ignore

    def __repr__(self) -> str:
        return (
            f"FinalModel("
            f"model={self.best_model_name!r}, "
            f"n_features={len(self.feature_names)}, "
            f"task={self.task_type.value!r})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _build_conformal_calibration(result, classes=None) -> Optional[ConformalCalibration]:
    """
    Build a ConformalCalibration from the out-of-fold predictions in best_cv_result.

    Regression:          nonconformity score = |y_test - y_pred_test|
    Binary classification:   nonconformity score = 1 - p(true_class)
    Multiclass classification: nonconformity score = 1 - p(true_class),
        looked up via the classes array from estimator.classes_.

    Returns None when no CV result is available or when the class mapping
    cannot be resolved for multiclass folds.
    """
    from loguru import logger

    cv = result.best_cv_result
    if cv is None or not cv.folds:
        return None

    task_type = cv.task_type

    if task_type == TaskType.REGRESSION:
        scores = np.concatenate([
            np.abs(f.y_test - f.y_pred_test) for f in cv.folds
        ])
    else:
        sample_scores = []
        for f in cv.folds:
            if f.y_prob_test.ndim == 1:
                # Binary: p is positive-class probability
                # nonconformity = 1 - p(true_class)
                sample_scores.append(
                    np.where(f.y_test == 1, 1.0 - f.y_prob_test, f.y_prob_test)
                )
            else:
                # Multiclass: map each true label to its column index via classes_
                if classes is None:
                    logger.warning(
                        "Conformal calibration skipped: multiclass folds require "
                        "estimator.classes_ to map labels to probability columns."
                    )
                    return None
                label_to_idx = {c: i for i, c in enumerate(classes)}
                try:
                    col_idx = np.array([label_to_idx[label] for label in f.y_test])
                except KeyError as e:
                    logger.warning(
                        f"Conformal calibration skipped: y_test contains label {e} "
                        f"not found in estimator.classes_."
                    )
                    return None
                p_true = f.y_prob_test[np.arange(len(col_idx)), col_idx]
                sample_scores.append(1.0 - p_true)
        scores = np.concatenate(sample_scores)

    return ConformalCalibration(
        scores    = np.sort(scores),
        n         = len(scores),
        task_type = task_type,
    )


def build_final_model(result: "PipelineResult") -> FinalModel:
    """
    Fit the overall best model on the full training dataset and return a
    FinalModel ready for inference.

    Picks the correct feature store (reduced or full) via best_feature_stage,
    applies StandardScaler when the model requires it, and fits with
    best_params when step 4 ran.

    Args:
        result: PipelineResult from H2MLPipeline.run().

    Returns:
        FinalModel instance.
    """
    from h2ml.utils.registry import CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY

    task_type = result.step1_cv_result[0].task_type

    feature_stage = result.best_feature_stage or result.best_stage
    store = (
        result.features_reduced
        if feature_stage == "reduced"
        else result.features
    )

    registry = (
        CLASSIFIER_REGISTRY
        if task_type == TaskType.CLASSIFICATION
        else REGRESSOR_REGISTRY
    )
    entry = registry.get(result.best_model_name)
    if entry is None:
        raise ValueError(
            f"Model '{result.best_model_name}' not found in registry. "
            "Was it removed after the pipeline ran?"
        )

    # Instantiate with best_params when available, otherwise default
    if result.best_params:
        estimator = entry.model_cls(**result.best_params)
    else:
        estimator = entry.model_cls(**entry.default_kwargs)

    # Scale if required — fit scaler on the full training set
    X = store.X
    scaler = None
    if entry.requires_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    estimator.fit(X, store.y)

    classes = getattr(estimator, "classes_", None)

    return FinalModel(
        estimator        = estimator,
        feature_names    = store.feature_names,
        task_type        = task_type,
        requires_scaling = entry.requires_scaling,
        scaler           = scaler,
        best_model_name  = result.best_model_name,
        best_params      = result.best_params,
        conformal        = _build_conformal_calibration(result, classes=classes),
        y_transform      = result.y_transform,
    )
