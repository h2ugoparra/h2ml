"""
tests/pipeline/test_mean_correction.py

Tests for the back-transform check: the [transform] warning, the opt-in
mean_correction="ratio" factor, and how predict, predict_interval and the delta model
carry it. Scores and selection must not change.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from h2ml.core.base import TaskType
from h2ml.core.cv_result import CVResult, FoldResult
from h2ml.core.feature_store import PipelineData
from h2ml.persistence.result_io import load_result, save_result
from h2ml.pipeline.final_model import (
    MEAN_RATIO_BOUNDS,
    DeltaFinalModel,
    FinalModel,
    _delta_oof_residuals,
    build_final_model,
)
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig, PipelineResult
from h2ml.pipeline.step import make_regressor
from h2ml.preprocessing.transforms import total_ratio

NAMES = ["a", "b"]


def _result(
    y_transform: str | None = "log", mean_correction: str | None = "ratio", oof_scale: float = 0.5, n: int = 60
) -> PipelineResult:
    """Result whose two-fold OOF predictions are oof_scale × y, so the ratio is 1 / oof_scale."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 2))
    y = rng.uniform(1.0, 20.0, n)
    idx = np.arange(n)
    folds = [
        FoldResult(
            fold_id=k,
            model_name="LinearRegression",
            y_train=y[idx % 2 != k],
            y_test=y[idx % 2 == k],
            y_pred_train=y[idx % 2 != k],
            y_pred_test=oof_scale * y[idx % 2 == k],
            test_idx=idx[idx % 2 == k],
        )
        for k in range(2)
    ]
    return PipelineResult(
        features=PipelineData(X=X, feature_names=NAMES, y=y),
        step1_cv_result=[
            CVResult(model_name="LinearRegression", task_type=TaskType.REGRESSION, folds=folds, y_transform=y_transform)
        ],
        best_model_name="LinearRegression",
        best_stage="default",
        best_feature_stage="default",
        best_base_estimator=LinearRegression(),
        y_transform=y_transform,
        mean_correction=mean_correction,
    )


def _pipeline(**kwargs) -> H2MLPipeline:
    models = [make_regressor(LinearRegression(), name="LinearRegression")]
    return H2MLPipeline(PipelineConfig(task_type="regression", metric="RMSE", **kwargs), models=models)


class TestTotalRatio:
    def test_ratio_ignores_non_finite_rows(self):
        assert total_ratio(np.array([1.0, 3.0, np.nan]), np.array([2.0, 6.0, 5.0])) == pytest.approx(0.5)

    @pytest.mark.parametrize("y_true, y_pred", [([0.0, 0.0], [1.0, 1.0]), ([1.0, 1.0], [-1.0, 0.0])])
    def test_none_without_positive_totals(self, y_true, y_pred):
        assert total_ratio(np.array(y_true), np.array(y_pred)) is None


class TestConfig:
    def test_default_is_off(self):
        assert PipelineConfig(task_type="regression", metric="RMSE").mean_correction is None

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="mean_correction"):
            PipelineConfig(task_type="regression", metric="RMSE", mean_correction="smearing")


class TestWarning:
    @pytest.mark.parametrize("transform", ["log", "sqrt", "winlog", "winsqrt"])
    def test_warns_and_states_the_share_of_the_total(self, transform):
        result = _result(transform, mean_correction=None, oof_scale=0.5)
        _pipeline()._check_back_transform(result)
        assert len(result.cv_warnings) == 1
        assert result.cv_warnings[0].startswith("[transform]")
        assert "0.50" in result.cv_warnings[0]
        assert "mean_correction='ratio'" in result.cv_warnings[0]

    def test_over_prediction_warns_too(self):
        result = _result(mean_correction=None, oof_scale=1.5)
        _pipeline()._check_back_transform(result)
        assert "1.50" in result.cv_warnings[0]

    @pytest.mark.parametrize("oof_scale", [0.81, 1.0, 1.19])
    def test_silent_within_tolerance(self, oof_scale):
        result = _result(mean_correction=None, oof_scale=oof_scale)
        _pipeline()._check_back_transform(result)
        assert result.cv_warnings == []

    @pytest.mark.parametrize("transform", ["count", "wincount", None])
    def test_silent_for_other_transforms(self, transform):
        result = _result(transform, mean_correction=None, oof_scale=0.5)
        _pipeline()._check_back_transform(result)
        assert result.cv_warnings == []

    def test_says_when_the_correction_is_on(self):
        result = _result(oof_scale=0.5)
        _pipeline()._check_back_transform(result)
        assert "applies the correction" in result.cv_warnings[0]

    def test_resuming_replaces_the_earlier_warning(self):
        result = _result(mean_correction=None, oof_scale=0.5)
        result.cv_warnings.append("other")
        pipeline = _pipeline()
        pipeline._check_back_transform(result)
        pipeline._check_back_transform(result)
        assert len(result.cv_warnings) == 2

        result.y_transform = "count"
        pipeline._check_back_transform(result)
        assert result.cv_warnings == ["other"]

    def test_run_records_it(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 3))
        y = np.where(rng.random(200) < 0.8, 0.0, rng.lognormal(2, 1.5, 200))
        store = PipelineData(X=X, feature_names=["a", "b", "c"], y=y)
        model = make_regressor(LinearRegression(), name="LinearRegression")
        config = PipelineConfig(task_type="regression", metric="RMSE", n_splits=3, n_jobs=1)
        pipeline = H2MLPipeline(config, models=[model])
        result, dev = pipeline._start(store)
        pipeline._run_step1(result, pipeline._build_transform_stores(dev, ["log"]))
        pipeline._finish(result)
        assert result.y_transform == "log"
        assert any(w.startswith("[transform]") for w in result.cv_warnings)


class TestFinalModelRatio:
    def test_ratio_comes_from_the_oof_totals(self):
        assert build_final_model(_result(oof_scale=0.5)).mean_ratio == pytest.approx(2.0)

    def test_predict_is_scaled(self):
        result = _result(oof_scale=0.5)
        plain = build_final_model(_result(mean_correction=None, oof_scale=0.5))
        corrected = build_final_model(result)
        X = result.features.X
        np.testing.assert_allclose(corrected.predict(X), 2.0 * plain.predict(X))

    @pytest.mark.parametrize("transform", ["log", "sqrt", "winlog", "winsqrt"])
    def test_applies_to_the_log_and_sqrt_family(self, transform):
        assert build_final_model(_result(transform, oof_scale=0.5)).mean_ratio == pytest.approx(2.0)

    @pytest.mark.parametrize("transform", ["count", "wincount", None])
    def test_other_transforms_stay_uncorrected(self, transform):
        assert build_final_model(_result(transform, oof_scale=0.5)).mean_ratio == 1.0

    def test_off_by_default(self):
        assert build_final_model(_result(mean_correction=None)).mean_ratio == 1.0

    @pytest.mark.parametrize("oof_scale, bound", [(0.01, MEAN_RATIO_BOUNDS[1]), (50.0, MEAN_RATIO_BOUNDS[0])])
    def test_clamped(self, oof_scale, bound):
        assert build_final_model(_result(oof_scale=oof_scale)).mean_ratio == bound

    def test_skipped_without_positive_totals(self):
        assert build_final_model(_result(oof_scale=-1.0)).mean_ratio == 1.0

    def test_classification_labels_are_untouched(self):
        X = np.random.default_rng(0).standard_normal((20, 2))
        y = np.array(["a", "b"] * 10)
        model = FinalModel(LogisticRegression().fit(X, y), NAMES, TaskType.CLASSIFICATION, mean_ratio=2.0)
        assert set(model.predict(X)) <= {"a", "b"}

    def test_models_saved_before_the_option_load_uncorrected(self):
        result = _result(mean_correction=None)
        model = build_final_model(result)
        del model.__dict__["mean_ratio"]  # as unpickled from an older save
        assert model.mean_ratio == 1.0
        assert model.predict(result.features.X).shape == (60,)


class TestIntervals:
    def test_scores_use_corrected_predictions(self):
        # OOF × ratio equals y here, so the corrected residuals are all zero
        result = _result(oof_scale=0.5)
        model = build_final_model(result)
        np.testing.assert_allclose(model.conformal.scores, 0.0, atol=1e-9)
        lower, upper = model.predict_interval(result.features.X)
        np.testing.assert_allclose(lower, model.predict(result.features.X))
        np.testing.assert_allclose(upper, model.predict(result.features.X))

    def test_scores_unchanged_when_off(self):
        result = _result(mean_correction=None, oof_scale=0.5)
        expected = np.sort(0.5 * np.concatenate([f.y_test for f in result.step1_cv_result[0].folds]))
        np.testing.assert_allclose(build_final_model(result).conformal.scores, expected)


class TestDelta:
    def test_regressor_oof_counts_are_corrected(self):
        n = 30
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 2))
        y = np.r_[rng.uniform(1, 10, 20), np.zeros(10)]
        positive = np.arange(20)
        clf_fold = FoldResult(
            fold_id=0,
            model_name="C",
            y_train=np.array([0, 1]),
            y_test=(y > 0).astype(int),
            y_pred_train=np.array([0, 1]),
            y_pred_test=(y > 0).astype(int),
            y_prob_test=np.ones(n),
            test_idx=np.arange(n),
        )
        reg_fold = FoldResult(
            fold_id=0,
            model_name="R",
            y_train=y[positive],
            y_test=y[positive],
            y_pred_train=y[positive],
            y_pred_test=0.5 * y[positive],
            test_idx=positive,
        )
        clf_result = PipelineResult(
            step1_cv_result=[CVResult("C", TaskType.CLASSIFICATION, folds=[clf_fold])],
            best_model_name="C",
            best_stage="default",
        )
        reg_result = PipelineResult(
            step1_cv_result=[CVResult("R", TaskType.REGRESSION, folds=[reg_fold])],
            best_model_name="R",
            best_stage="default",
        )
        reg_final = FinalModel(
            LinearRegression().fit(X, np.log1p(y)), NAMES, TaskType.REGRESSION, y_transform="log", mean_ratio=2.0
        )
        residuals = _delta_oof_residuals(clf_result, reg_result, reg_final, X, y, positive)
        np.testing.assert_allclose(residuals[positive], 0.0, atol=1e-9)

    def test_only_the_count_part_is_scaled(self):
        X = np.random.default_rng(0).standard_normal((40, 2))
        y = np.exp(X[:, 0])
        clf = FinalModel(LogisticRegression().fit(X, y > 1), NAMES, TaskType.CLASSIFICATION)
        estimator = LinearRegression().fit(X, np.log1p(y))
        reg = FinalModel(estimator, NAMES, TaskType.REGRESSION, y_transform="log", mean_ratio=2.0)
        delta = DeltaFinalModel(clf=clf, reg=reg)
        expected = clf.predict_proba(X) * 2.0 * np.expm1(estimator.predict(X))
        np.testing.assert_allclose(delta.predict(X), expected)


class TestPersistence:
    def test_round_trip(self, tmp_path):
        save_result(_result(), tmp_path / "r")
        assert load_result(tmp_path / "r").mean_correction == "ratio"

    def test_old_results_load_without_it(self, tmp_path):
        save_result(_result(), tmp_path / "r")
        path = tmp_path / "r" / "metadata.json"
        meta = json.loads(path.read_text())
        del meta["mean_correction"]
        path.write_text(json.dumps(meta))
        assert load_result(tmp_path / "r").mean_correction is None


def test_scores_and_selection_ignore_the_option():
    def run(mean_correction):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 3))
        y = rng.lognormal(X[:, 0], 1.0)
        store = PipelineData(X=X, feature_names=["a", "b", "c"], y=y)
        model = make_regressor(LinearRegression(), name="LinearRegression")
        config = PipelineConfig(
            task_type="regression", metric="RMSE", n_splits=3, n_jobs=1, mean_correction=mean_correction
        )
        pipeline = H2MLPipeline(config, models=[model])
        result, dev = pipeline._start(store)
        return pipeline._run_step1(result, pipeline._build_transform_stores(dev, ["log", "count"]))

    off, on = run(None), run("ratio")
    assert (off.best_model_name, off.y_transform) == (on.best_model_name, on.y_transform)
    assert off.best_model_value == on.best_model_value
