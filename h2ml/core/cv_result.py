"""
Raw cross-validation result containers (FoldResult, CVResult).

Foundational, dependency-free data types: the CV engine (pipeline/cv.py) produces
them and the evaluation layer consumes them, so they live in core to keep both
sides off each other's import path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from h2ml.core.base import TaskType

# ---------------------------------------------------------------------------
# FoldResult — raw output of a single fold
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """
    Stores the raw arrays produced in one CV fold.
    Metric computation happens downstream in evaluation/metrics.py.

    Attributes:
        fold_id:      Index of this fold within the CV run.
        model_name:   Name of the model that produced these predictions.
        y_train:      Ground-truth targets for the training rows.
        y_test:       Ground-truth targets for the held-out rows.
        y_pred_train: Point predictions on the training rows.
        y_pred_test:  Point predictions on the held-out rows.
        y_prob_train: Predicted class probabilities on the training rows
                      (None for regressors).
        y_prob_test:  Predicted class probabilities on the held-out rows
                      (None for regressors).
        train_idx:    Row indices used for training — kept for traceability.
        test_idx:     Row indices held out — used to reassemble OOF predictions.
        fit_time:     Wall-clock seconds spent fitting the model on this fold.
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
    def task_type(self) -> TaskType:
        """Inferred from whether probabilities are present.

        Returns a TaskType member, matching CVResult.task_type; string
        comparisons keep working because TaskType is a str enum.
        """
        return TaskType.CLASSIFICATION if self.y_prob_test is not None else TaskType.REGRESSION


# ---------------------------------------------------------------------------
# CVResult — aggregates all folds for one model
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """
    All fold results for a single model run.

    Attributes:
        model_name:   Name of the model these folds belong to.
        task_type:    TaskType.CLASSIFICATION or TaskType.REGRESSION.
        folds:        Successful FoldResults, one per completed fold.
        failed_folds: Indices of folds that raised during fit/predict and were
                      skipped. A non-empty list means metrics are computed on fewer
                      than the requested number of folds.
    """

    model_name: str
    task_type: TaskType
    folds: list[FoldResult] = field(default_factory=list)  # successful folds only
    # Indices of folds that raised during fit/predict and were skipped. A non-empty
    # list means metrics are computed on fewer than the requested number of folds.
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
