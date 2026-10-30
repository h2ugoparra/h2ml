"""
tests/geo/test_geo_predict.py

Tests for _predict_single — exercised directly with in-memory DataFrames so no
h2mare / ParquetIndexer is needed (polars is a core dependency).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from h2ml.core.base import TaskType
from h2ml.geo.geo_predict import _predict_delta_single, _predict_single
from h2ml.pipeline.final_model import DeltaFinalModel, FinalModel

FEATURE_NAMES = ["f0", "f1", "f2"]


def _make_df(X: np.ndarray) -> pl.DataFrame:
    df = pl.DataFrame({name: X[:, i] for i, name in enumerate(FEATURE_NAMES)})
    return df.with_row_index("index")


def _make_clf_final_model(n_classes: int) -> tuple[FinalModel, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, len(FEATURE_NAMES)))
    y = rng.integers(0, n_classes, size=60)
    est = LogisticRegression(max_iter=200, random_state=42).fit(X, y)
    model = FinalModel(
        estimator=est,
        feature_names=FEATURE_NAMES,
        task_type=TaskType.CLASSIFICATION,
        best_model_name="LogisticRegression",
    )
    return model, X


class TestPredictSingleClassification:
    def test_binary_returns_probability_series(self):
        model, X = _make_clf_final_model(n_classes=2)
        (pred,) = _predict_single(_make_df(X), model, col_name="target")
        assert pred.name == "target"
        assert len(pred) == len(X)
        vals = pred.to_numpy()
        assert np.all((vals >= 0) & (vals <= 1))

    def test_multiclass_raises_clear_error(self):
        """Multiclass predict_proba is a matrix and cannot fill a single Float32
        column (regression: it previously died inside polars with a cryptic
        InvalidOperationError instead of a clear message)."""
        model, X = _make_clf_final_model(n_classes=3)
        with pytest.raises(ValueError, match="multiclass"):
            _predict_single(_make_df(X), model, col_name="target")


def _make_log_reg_final_model() -> tuple[FinalModel, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, len(FEATURE_NAMES)))
    y = rng.uniform(1.0, 30.0, size=60)
    est = KNeighborsRegressor(weights="distance").fit(X, np.log1p(y))
    model = FinalModel(estimator=est, feature_names=FEATURE_NAMES, task_type=TaskType.REGRESSION, y_transform="log")
    return model, X, y


class TestPredictYTransform:
    def test_single_inverts_transform_once(self):
        model, X, y = _make_log_reg_final_model()
        (pred,) = _predict_single(_make_df(X), model, col_name="target")
        np.testing.assert_allclose(pred.to_numpy(), y, rtol=1e-5)

    def test_delta_inverts_transform_once(self):
        reg, X, y = _make_log_reg_final_model()
        clf = _make_clf_final_model(n_classes=2)[0]
        (pred,) = _predict_delta_single(_make_df(X), DeltaFinalModel(clf=clf, reg=reg), col_name="target")
        np.testing.assert_allclose(pred.to_numpy(), clf.predict_proba(X) * y, rtol=1e-5)
