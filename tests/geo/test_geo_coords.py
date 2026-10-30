"""
tests/geo/test_geo_coords.py

Local conformal is calibrated on store.coords, documented as (lat, lon), so the geo
paths must pass the scanned columns in that order rather than as (lon, lat).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression, LogisticRegression

from h2ml.core.base import TaskType
from h2ml.geo.geo_predict import _predict_delta_single, _predict_single
from h2ml.pipeline.final_model import DeltaFinalModel, FinalModel

FEATURE_NAMES = ["f0", "f1"]
LAT = np.array([36.0, 38.0, 40.0, 42.0])
LON = np.array([-9.0, -7.0, -5.0, -3.0])


def _frame(X: np.ndarray) -> pl.DataFrame:
    columns = {name: X[:, i] for i, name in enumerate(FEATURE_NAMES)}
    return pl.DataFrame({**columns, "lon": LON, "lat": LAT}).with_row_index("index")


def _regressor(X: np.ndarray, **kwargs) -> FinalModel:
    return FinalModel(
        estimator=LinearRegression().fit(X, X[:, 0]),
        feature_names=FEATURE_NAMES,
        task_type=TaskType.REGRESSION,
        **kwargs,
    )


def _local_conformal() -> MagicMock:
    local = MagicMock()
    local.threshold.return_value = np.ones(len(LAT))
    return local


def _assert_lat_lon(local: MagicMock) -> None:
    coords = local.threshold.call_args.kwargs["coords"]
    np.testing.assert_array_equal(coords, np.column_stack([LAT, LON]))


def test_predict_single_passes_lat_lon():
    X = np.random.default_rng(0).standard_normal((4, 2))
    local = _local_conformal()
    model = _regressor(X, conformal=MagicMock(), local_conformal=local)
    _predict_single(_frame(X), model, col_name="t", alpha=0.1, local=True)
    _assert_lat_lon(local)


def test_predict_delta_single_passes_lat_lon():
    X = np.random.default_rng(1).standard_normal((4, 2))
    clf = FinalModel(
        estimator=LogisticRegression().fit(X, [0, 1, 0, 1]),
        feature_names=FEATURE_NAMES,
        task_type=TaskType.CLASSIFICATION,
    )
    local = _local_conformal()
    delta = DeltaFinalModel(clf=clf, reg=_regressor(X), conformal=MagicMock(), local_conformal=local)
    _predict_delta_single(_frame(X), delta, col_name="t", alpha=0.1, local=True)
    _assert_lat_lon(local)
