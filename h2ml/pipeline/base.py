"""
h2ml/pipeline/base.py

Abstract base classes that define the contract every pipeline step must fulfill.
All classifiers, regressors, and preprocessors inherit from these.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Task type
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TRANSFORM = "transform"  # preprocessing steps — no predict()


# ---------------------------------------------------------------------------
# Base step
# ---------------------------------------------------------------------------


class BaseStep(ABC):
    """
    Common interface for every step in the pipeline.

    Subclasses must implement:
        - fit(X, y)
        - transform(X)

    Optionally override:
        - fit_transform(X, y)  — default calls fit then transform
        - get_feature_names_out()

    Args:
        name:    Step name used in logs and __repr__. Defaults to the class name.
        verbose: Log progress during fit/transform.
    """

    # Every subclass sets this at the class level
    task_type: TaskType

    def __init__(self, name: Optional[str] = None, verbose: bool = False):
        self.name = name or self.__class__.__name__
        self.verbose = verbose
        self._fitted = False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseStep":
        """Fit the step to training data. Must return self."""
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted step to data. Returns a DataFrame."""
        ...

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one call. Override for efficiency if needed."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Optional — override when the step produces named features
    # ------------------------------------------------------------------

    def get_feature_names_out(self) -> list[str]:
        """Return output feature names after transform. Override when relevant."""
        return []

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if transform/predict is called before fit."""
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] Step has not been fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={self._fitted})"


# ---------------------------------------------------------------------------
# Predictor mixin — shared by classifiers AND regressors
# ---------------------------------------------------------------------------


class PredictorMixin(ABC):
    """
    Adds predict() to any BaseStep subclass.
    Classifiers and regressors both use this.
    """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for X."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities. Only classifiers need to implement this.
        Raises NotImplementedError by default so regressors stay clean.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support predict_proba().")


# ---------------------------------------------------------------------------
# Concrete base classes
# ---------------------------------------------------------------------------


class BaseClassifier(BaseStep, PredictorMixin):
    """
    Base class for all classifiers.

    Subclasses must implement:
        - fit(X, y)
        - transform(X)   → typically returns X unchanged (pass-through)
        - predict(X)
        - predict_proba(X)  ← required for classifiers
    """

    task_type = TaskType.CLASSIFICATION

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities. Required for all classifiers."""
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Classifiers are terminal steps — transform is a pass-through by default.
        Override if the classifier enriches X (e.g. stacking).
        """
        self._check_fitted()
        return X


class BaseRegressor(BaseStep, PredictorMixin):
    """
    Base class for all regressors.

    Subclasses must implement:
        - fit(X, y)
        - transform(X)   → pass-through by default
        - predict(X)
    """

    task_type = TaskType.REGRESSION

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Regressors are terminal steps — transform is a pass-through by default.
        Override if the regressor enriches X (e.g. stacking).
        """
        self._check_fitted()
        return X


class BasePreprocessor(BaseStep):
    """
    Base class for all preprocessing steps (scalers, encoders, imputers, etc.).

    Subclasses must implement:
        - fit(X, y)
        - transform(X)
    """

    task_type = TaskType.TRANSFORM


@runtime_checkable
class PredictorStep(Protocol):
    """Structural type for any step that can predict."""

    name: str
    task_type: TaskType
    requires_scaling: bool
    estimator: Any

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PredictorStep": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
