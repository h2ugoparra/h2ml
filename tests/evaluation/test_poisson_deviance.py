"""
tests/evaluation/test_poisson_deviance.py

Tests for the PoissonDev regression metric: the score, pooled and fold columns, the
Optuna objective, the pipeline's validation, and multi_metric.

Step 2 (SHAP) is mocked as in test_selection_scoring.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_poisson_deviance

from h2ml.core.base import TaskType
from h2ml.core.cv_result import CVResult, FoldResult
from h2ml.core.feature_store import PipelineData
from h2ml.evaluation.metrics import (
    METRIC_COL,
    METRIC_MINIMIZE,
    POISSON_PRED_FLOOR,
    POOLED_COL,
    aggregate_metrics,
    compute_metrics,
    poisson_deviance,
)
from h2ml.evaluation.ranking import COUNTERPART_METRIC
from h2ml.features.selector import FeatureSelector
from h2ml.optimization.optimizer import _neg_poisson_dev, _score_poisson_dev
from h2ml.pipeline.multi_metric import multi_metric
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig
from h2ml.pipeline.step import make_regressor

FEATURE_NAMES = [f"feat_{i}" for i in range(4)]
N = 120


def _fold(y_test, y_pred, fold_id=0) -> FoldResult:
    y_test, y_pred = np.asarray(y_test, float), np.asarray(y_pred, float)
    y_train = np.array([0.0, 1.0, 2.0, 3.0])
    return FoldResult(
        fold_id=fold_id, model_name="M", y_train=y_train, y_test=y_test, y_pred_train=y_train + 0.1, y_pred_test=y_pred
    )


def _aggregate(folds: list[FoldResult]):
    return aggregate_metrics(compute_metrics(CVResult(model_name="M", task_type=TaskType.REGRESSION, folds=folds)))


def _store(y: np.ndarray | None = None) -> PipelineData:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, len(FEATURE_NAMES)))
    y = np.exp(0.5 * X[:, 0]) if y is None else y
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _pipeline(metric: str = "PoissonDev") -> H2MLPipeline:
    config = PipelineConfig(task_type="regression", metric=metric, n_splits=3, n_trials=1, n_jobs=1)
    return H2MLPipeline(config=config, models=[make_regressor(Ridge(), name="Ridge")])


def _mock_selector():
    mock_sel = MagicMock(spec=FeatureSelector)
    mock_sel.selected_features_ = FEATURE_NAMES[:2]
    mock_sel.fit_transform.side_effect = lambda store, y=None: store.select(FEATURE_NAMES[:2])
    mock_sel.transform.side_effect = lambda store: store.select(FEATURE_NAMES[:2])
    return patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=mock_sel)


class TestScore:
    def test_matches_sklearn(self):
        y, p = np.array([0.0, 1.0, 5.0]), np.array([0.5, 1.5, 4.0])
        assert poisson_deviance(y, p) == pytest.approx(mean_poisson_deviance(y, p))

    def test_zero_prediction_is_floored_not_an_error(self):
        y, p = np.array([0.0, 2.0]), np.array([0.0, 0.0])
        expected = mean_poisson_deviance(y, np.full(2, POISSON_PRED_FLOOR))
        assert poisson_deviance(y, p) == pytest.approx(expected)

    def test_negative_prediction_is_floored(self):
        assert np.isfinite(poisson_deviance(np.array([1.0, 2.0]), np.array([-3.0, 1.0])))

    def test_negative_target_is_nan(self):
        assert np.isnan(poisson_deviance(np.array([-1.0, 2.0]), np.array([1.0, 1.0])))

    def test_is_a_minimised_metric_with_a_counterpart(self):
        assert METRIC_MINIMIZE["PoissonDev"] is True
        assert METRIC_COL["PoissonDev"] == "PoissonDev_Test_Mean"
        assert "PoissonDev" in COUNTERPART_METRIC


class TestColumns:
    def test_pooled_matches_a_direct_computation_on_all_rows(self):
        rng = np.random.default_rng(0)
        y = rng.poisson(2.0, 60).astype(float)
        p = rng.uniform(0.5, 3.0, 60)
        row = _aggregate([_fold(y[:5], p[:5]), _fold(y[5:], p[5:], 1)]).iloc[0]
        assert row[POOLED_COL["PoissonDev"]] == pytest.approx(mean_poisson_deviance(y, p))

    def test_one_negative_fold_makes_the_pooled_score_nan(self):
        row = _aggregate([_fold([1, 2], [1, 2]), _fold([-1, 2], [1, 2], 1)]).iloc[0]
        assert np.isnan(row[POOLED_COL["PoissonDev"]])

    def test_overfit_gap_is_reported(self):
        assert "PoissonDev_Gap_Mean" in _aggregate([_fold([1, 2], [1, 2]), _fold([1, 3], [1, 2], 1)]).columns


class TestOptimizer:
    def test_scorer_negates_so_higher_is_better(self):
        y, p = np.array([1.0, 2.0]), np.array([1.5, 1.5])
        assert _neg_poisson_dev(y, p) == pytest.approx(-mean_poisson_deviance(y, p))

    def test_fold_scorer_uses_the_original_scale(self):
        X = np.arange(4.0)[:, None]
        y_log = np.log1p(np.array([0.0, 1.0, 2.0, 3.0]))
        y_true = np.array([0.0, 1.0, 2.0, 3.0])
        score = _score_poisson_dev(Ridge(alpha=1e-9), X, X, y_log, y_log, y_true_test=y_true, inverse_fn=np.expm1)
        assert score <= 0


class TestPipeline:
    def test_runs_and_reports_the_pooled_score(self):
        with _mock_selector():
            result = _pipeline().run_step1_to_step3(_store())
        assert result.metric == "PoissonDev"
        scores = [df.iloc[0][POOLED_COL["PoissonDev"]] for df in (result.step1_agg_df, result.step3_agg_df)]
        assert result.best_model_value == pytest.approx(min(scores))

    def test_negative_target_raises(self):
        y = np.random.default_rng(1).normal(size=N)
        with pytest.raises(ValueError, match="non-negative"):
            _pipeline().run_step1_only(_store(y))

    def test_other_metrics_still_accept_negative_targets(self):
        y = np.random.default_rng(1).normal(size=N)
        with _mock_selector():
            result = _pipeline("RMSE").run_step1_only(_store(y))
        assert np.isnan(result.step1_agg_df.iloc[0]["PoissonDev_Test_Mean"])
        assert np.isfinite(result.best_model_value)


class TestMultiMetric:
    def test_not_in_the_default_set(self):
        with _mock_selector():
            result = multi_metric(_pipeline("RMSE"), _store())
        assert "PoissonDev" not in result.results

    def test_can_be_requested(self):
        with _mock_selector():
            result = multi_metric(_pipeline("RMSE"), _store(), metrics=["RMSE", "PoissonDev"])
        assert set(result.results) == {"RMSE", "PoissonDev"}
