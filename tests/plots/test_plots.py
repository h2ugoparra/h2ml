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


def _fold(fold_id: int, y_test: np.ndarray, y_prob: np.ndarray) -> FoldResult:
    return FoldResult(
        fold_id=fold_id,
        model_name="M",
        y_train=y_test.copy(),
        y_test=y_test,
        y_pred_train=y_test.copy(),
        y_pred_test=(y_prob > 0.5).astype(float),
        y_prob_train=y_prob.copy(),
        y_prob_test=y_prob,
    )


def _offset_folds_cv_result() -> CVResult:
    """Two folds, each perfectly ranked, but fold 1's probabilities sit on a higher
    scale — so pooling puts fold 1's negatives above fold 0's positives."""
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    low = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    return CVResult(
        model_name="M",
        task_type=TaskType.CLASSIFICATION,
        folds=[_fold(0, y, low), _fold(1, y.copy(), low + 0.5)],
    )


class TestCvDiagnosticsFoldAuc:
    def test_fold_aucs_skip_single_class_folds(self):
        from h2ml.plots.plots import _fold_aucs

        both = (np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
        single = (np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
        assert _fold_aucs([both, single]) == [1.0]

    def test_roc_legend_shows_fold_mean_and_pooled(self, monkeypatch):
        import matplotlib.pyplot as plt

        figures = []
        monkeypatch.setattr("h2ml.plots.plots._save_or_show", lambda _path: figures.append(plt.gcf()))
        cv_diagnostics(_offset_folds_cv_result())

        roc_ax = next(ax for ax in figures[0].axes if ax.get_title() == "ROC Curve")
        labels = [t.get_text() for t in roc_ax.get_legend().get_texts()]
        plt.close(figures[0])

        # Each fold ranks perfectly; pooling across the offset scales does not.
        assert "Fold mean AUC = 1.000 ± 0.000" in labels
        pooled = next(label for label in labels if label.startswith("Pooled AUC"))
        assert float(pooled.split("= ")[1]) < 1.0
        assert "Folds" in labels
