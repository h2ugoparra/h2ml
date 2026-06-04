"""
h2ml/pipeline/step.py

ModelWrapper — a wrapper that promotes any sklearn-compatible model into a
BaseStep-compliant object, so existing sklearn estimators can be dropped into
the h2ml pipeline without rewriting them.

Responsibilities:
    - Wrap sklearn classifiers / regressors / transformers
    - Delegate fit / transform / predict to the inner model
    - Expose requires_scaling and task_type so the CV engine can read them
    - Provide a clean __repr__ for logging and debugging
"""

from __future__ import annotations

from typing import Optional, Any
from loguru import logger

import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor

from h2ml.core.base import (
    BaseClassifier,
    BaseRegressor,
    TaskType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_task_type(estimator: Any) -> TaskType:
    """Infer TaskType from a sklearn estimator."""
    if is_classifier(estimator):
        return TaskType.CLASSIFICATION
    if is_regressor(estimator):
        return TaskType.REGRESSION
    return TaskType.TRANSFORM


def _has_predict_proba(estimator: Any) -> bool:
    return hasattr(estimator, "predict_proba")


# ---------------------------------------------------------------------------
# ModelWrapper — wraps a classifier or regressor
# ---------------------------------------------------------------------------


class ModelWrapper(BaseClassifier, BaseRegressor):
    """
    Wraps any sklearn-compatible classifier or regressor into a BaseStep.

    Args:
        estimator:            A fitted or unfitted sklearn estimator.
        name:                 Optional display name. Defaults to class name.
        requires_scaling:     Whether this model needs StandardScaler applied
                              before fitting. The CV engine reads this flag.
        verbose:              Print fit/predict progress.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> step = ModelWrapper(RandomForestClassifier(n_estimators=100))
        >>> step.fit(X_train, y_train)
        >>> preds = step.predict(X_test)
    """

    # Override task_type per instance — set in __init__ based on estimator
    task_type: TaskType

    def __init__(
        self,
        estimator: Any,
        name: Optional[str] = None,
        requires_scaling: bool = False,
        verbose: bool = False,
    ):
        # Resolve name before super().__init__ so _check_fitted messages are clean
        resolved_name = name or estimator.__class__.__name__
        super().__init__(name=resolved_name, verbose=verbose)

        self.estimator = estimator
        self.requires_scaling = requires_scaling

        # Infer task type from the wrapped estimator
        self.task_type = _infer_task_type(estimator)

    # ------------------------------------------------------------------
    # BaseStep interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ModelWrapper":
        if y is not None:
            self.estimator.fit(X, y)
            self._fitted = True
        if self.verbose:
            logger.info(f"[{self.name}] Fitted on {X.shape[0]} samples, {X.shape[1]} features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Pass-through for classifiers and regressors.
        For transformers (preprocessors), delegates to estimator.transform().
        """
        self._check_fitted()

        if self.task_type == TaskType.TRANSFORM:
            transformed = self.estimator.transform(X)
            # sklearn transformers may return arrays — wrap back to DataFrame
            if isinstance(transformed, np.ndarray):
                return pd.DataFrame(transformed, columns=self._get_out_columns(X, transformed))
            return transformed

        # Classifier / regressor — pass-through
        return X

    # ------------------------------------------------------------------
    # PredictorMixin interface
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns class probabilities. Only valid for classifiers.
        Raises clearly if called on a regressor or uncalibrated model.
        """
        self._check_fitted()

        if self.task_type != TaskType.CLASSIFICATION:
            raise TypeError(
                f"[{self.name}] predict_proba() is only available for classifiers. "
                f"This step has task_type='{self.task_type}'."
            )

        if not _has_predict_proba(self.estimator):
            raise TypeError(f"[{self.name}] The wrapped estimator does not support predict_proba(). ")

        return self.estimator.predict_proba(X)

    # ------------------------------------------------------------------
    # Optional — feature names after transform
    # ------------------------------------------------------------------

    def get_feature_names_out(self) -> list[str]:
        if hasattr(self.estimator, "get_feature_names_out"):
            return list(self.estimator.get_feature_names_out())
        return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_out_columns(self, X_in: pd.DataFrame, transformed: np.ndarray) -> list[str]:
        """Resolve output column names after a transform step."""
        names = self.get_feature_names_out()
        if names:
            return names
        # Fallback: keep original column names if shape is unchanged
        if transformed.shape[1] == X_in.shape[1]:
            return list(X_in.columns)
        return [f"feature_{i}" for i in range(transformed.shape[1])]

    def __repr__(self) -> str:
        return (
            f"ModelWrapper("
            f"name={self.name!r}, "
            f"task_type={self.task_type.value!r}, "
            f"requires_scaling={self.requires_scaling}, "
            f"fitted={self._fitted}"
            f")"
        )


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def make_classifier(
    estimator: Any,
    name: Optional[str] = None,
    requires_scaling: bool = False,
) -> ModelWrapper:
    """
    Shorthand for wrapping a sklearn classifier.

    Example:
        >>> from sklearn.svm import SVC
        >>> step = make_classifier(SVC(), requires_scaling=True)
    """
    return ModelWrapper(
        estimator=estimator,
        name=name,
        requires_scaling=requires_scaling,
    )


def make_regressor(
    estimator: Any,
    name: Optional[str] = None,
    requires_scaling: bool = False,
) -> ModelWrapper:
    """
    Shorthand for wrapping a sklearn regressor.

    Example:
        >>> from sklearn.linear_model import Ridge
        >>> step = make_regressor(Ridge(), requires_scaling=True)
    """
    return ModelWrapper(
        estimator=estimator,
        name=name,
        requires_scaling=requires_scaling,
    )


def make_preprocessor(
    estimator: Any,
    name: Optional[str] = None,
) -> ModelWrapper:
    """
    Shorthand for wrapping a sklearn transformer.

    Example:
        >>> from sklearn.preprocessing import StandardScaler
        >>> step = make_preprocessor(StandardScaler(), name="scaler")
    """
    return ModelWrapper(estimator=estimator, name=name)
