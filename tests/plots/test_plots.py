"""
tests/plots/test_plots.py

Tests for cv_diagnostics input validation (binary-only classification panel).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from h2ml.core.base import TaskType
from h2ml.core.cv_result import CVResult, FoldResult
from h2ml.plots.plots import cv_diagnostics

matplotlib.use("Agg")  # headless — tests only save, never show


def _make_clf_cv_result(n_classes: int, n: int = 40, seed: int = 0) -> CVResult:
    """CVResult with one fold; 1-D probabilities for binary, 2-D for multiclass."""
    rng = np.random.default_rng(seed)
    y_test = rng.integers(0, n_classes, n).astype(float)
    if n_classes == 2:
        y_prob = rng.uniform(0.01, 0.99, n)
    else:
        raw = rng.uniform(0.01, 1.0, (n, n_classes))
        y_prob = raw / raw.sum(axis=1, keepdims=True)
    fold = FoldResult(
        fold_id=0,
        model_name="M",
        y_train=y_test.copy(),
        y_test=y_test,
        y_pred_train=y_test.copy(),
        y_pred_test=y_test.copy(),
        y_prob_train=y_prob.copy(),
        y_prob_test=y_prob,
    )
    return CVResult(model_name="M", task_type=TaskType.CLASSIFICATION, folds=[fold])


class TestCvDiagnosticsBinaryOnly:
    def test_multiclass_raises_clear_error(self):
        """A 2-D probability matrix must fail fast with a readable message, not
        crash deep inside roc_curve."""
        cv = _make_clf_cv_result(n_classes=3)
        with pytest.raises(ValueError, match="binary targets only"):
            cv_diagnostics(cv)

    def test_binary_panel_renders(self, tmp_path):
        cv = _make_clf_cv_result(n_classes=2)
        out = tmp_path / "panel.png"
        cv_diagnostics(cv, save_path=out)
        assert out.exists()
