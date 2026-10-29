"""
tests/pipeline/test_multi_metric.py

Tests for multi_metric / MultiMetricResult. Step 2 (SHAP) is mocked; RF is tuned
in step 4 (n_trials=2), LR and NB have no enabled search space. Regression runs
Ridge + RF regressors with a y-transform sweep.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB

from h2ml import MultiMetricResult, multi_metric
from h2ml.core.feature_store import PipelineData
from h2ml.evaluation.metrics import METRIC_COL, METRIC_MINIMIZE, select_best
from h2ml.features.selector import FeatureSelector
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig
from h2ml.pipeline.step import make_classifier, make_regressor

FEATURE_NAMES = [f"feat_{i}" for i in range(6)]
SELECTED = FEATURE_NAMES[:3]
METRICS = ["Brier", "LogLoss", "AUC", "AUCPR", "F1"]
REG_METRICS = ["R2", "MAE", "RMSE"]
TRANSFORMS = ["log"]


def _store(n: int = 240, positive_rate: float = 0.4, seed: int = 0) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    score = 2 * X[:, 0] * X[:, 1] + 0.5 * rng.standard_normal(n)  # interaction: favours the tunable RF
    y = (score > np.quantile(score, 1 - positive_rate)).astype(int)
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _reg_store(n: int = 240, seed: int = 0) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = np.exp(0.6 * X[:, 0] * X[:, 1] + 0.3 * X[:, 2] + 0.2 * rng.standard_normal(n))  # positive: "log" applies
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _mock_selector():
    sel = MagicMock(spec=FeatureSelector)
    sel.selected_features_ = SELECTED
    sel.fit_transform.side_effect = lambda store, y=None: store.select(SELECTED)
    sel.transform.side_effect = lambda store: store.select(SELECTED)
    return patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=sel)


def _pipeline(**kwargs) -> H2MLPipeline:
    config = PipelineConfig(
        **{
            "task_type": "classification",
            "metric": "AUC",
            "n_splits": 3,
            "opt_n_splits": 2,
            "n_trials": 2,
            "n_jobs": 1,
            "verbose": False,
            **kwargs,
        }
    )
    models = [
        make_classifier(LogisticRegression(max_iter=500), name="LR"),
        make_classifier(RandomForestClassifier(n_estimators=10, random_state=0), name="RF"),
        make_classifier(GaussianNB(), name="NB"),
    ]
    return H2MLPipeline(config=config, models=models)


def _reg_pipeline(**kwargs) -> H2MLPipeline:
    config = PipelineConfig(
        **{
            "task_type": "regression",
            "metric": "R2",
            "n_splits": 3,
            "opt_n_splits": 2,
            "n_trials": 2,
            "n_jobs": 1,
            "verbose": False,
            **kwargs,
        }
    )
    models = [
        make_regressor(Ridge(), name="Ridge"),
        make_regressor(RandomForestRegressor(n_estimators=10, random_state=0), name="RF"),
    ]
    return H2MLPipeline(config=config, models=models)


def _assert_scores_equal(a, b):
    """Frames equal apart from wall-clock timing columns."""
    if a is None or b is None:
        assert a is None and b is None
        return
    keep = [c for c in a.columns if "Time" not in c]
    pd.testing.assert_frame_equal(a[keep], b[keep])


@pytest.fixture(scope="module")
def mm() -> MultiMetricResult:
    with _mock_selector():
        return multi_metric(_pipeline(holdout_frac=0.2), _store(), metrics=METRICS)


class TestEquivalence:
    """With reuse_step4="none", each metric's result equals a separate run under that metric."""

    @pytest.mark.parametrize("metric", METRICS)
    def test_matches_separate_run(self, mm, metric):
        with _mock_selector():
            separate = _pipeline(holdout_frac=0.2, metric=metric).run(_store())
        shared = mm.results[metric]

        assert shared.metric == separate.metric == metric
        for name in ("best_model_name", "best_stage", "best_feature_stage", "y_transform", "class_weight_balanced"):
            assert getattr(shared, name) == getattr(separate, name), name
        assert shared.best_model_value == pytest.approx(separate.best_model_value)
        assert shared.best_model_std == pytest.approx(separate.best_model_std)
        assert shared.best_params == separate.best_params
        assert shared.cv_warnings == separate.cv_warnings
        np.testing.assert_array_equal(shared.holdout_idx, separate.holdout_idx)
        for name in ("holdout_df", "step1_agg_df", "step3_agg_df", "step4_agg_df"):
            _assert_scores_equal(getattr(shared, name), getattr(separate, name))


class TestResult:
    def test_runs_df_one_row_per_metric(self, mm):
        assert list(mm.runs_df["Metric"]) == METRICS
        assert mm.runs_df["Step4_shared_with"].isna().all()
        assert {"Model", "Stage", "Feature_stage", "N_features", "Class_weight", "CV", "CV_Std", "Holdout"} <= set(
            mm.runs_df.columns
        )

    def test_holdout_scored_under_each_metric(self, mm):
        for metric in METRICS:
            row = mm.runs_df.set_index("Metric").loc[metric]
            assert row["Holdout"] == pytest.approx(mm.results[metric].holdout_df.iloc[0][f"{metric}_Test_Mean"])

    def test_summary_and_counts(self, mm):
        summary = mm.summary().iloc[0]
        assert summary["N_metrics"] == len(METRICS)
        assert summary["N_hpo_runs"] == len(METRICS)
        assert mm.selection_counts()["Count"].sum() == len(METRICS)
        assert mm.feature_frequency().max() == 1.0

    def test_save_load_round_trip(self, mm, tmp_path):
        for result in mm.results.values():
            result.selector = None  # the mocked selector cannot be pickled
        mm.save(tmp_path / "mm")
        loaded = MultiMetricResult.load(tmp_path / "mm")
        assert loaded.metrics == mm.metrics
        assert loaded.reuse_step4 == "none"
        assert list(loaded.results) == METRICS
        assert loaded.results["AUC"].best_model_name == mm.results["AUC"].best_model_name
        pd.testing.assert_frame_equal(loaded.runs_df, mm.runs_df, check_dtype=False)


def _spy(name: str):
    original = getattr(H2MLPipeline, name)
    return patch.object(H2MLPipeline, name, autospec=True, side_effect=original)


class TestSharing:
    def test_step1_cv_runs_once_per_class_weighting(self):
        # 8% positives: "auto" turns weights on, so the AUC group needs its own step-1 CV
        store = _store(n=300, positive_rate=0.08)
        with _mock_selector(), _spy("_cv_step1") as cv1:
            result = multi_metric(
                _pipeline(handle_imbalance="auto"),
                store,
                metrics=["Brier", "LogLoss", "AUC", "AUCPR"],
                handle_imbalance={"Brier": False, "LogLoss": False},
            )
        assert cv1.call_count == 2
        assert list(result.runs_df["Class_weight"]) == [False, False, True, True]

    def test_steps_2_3_run_once_per_distinct_winner(self):
        with _mock_selector(), _spy("_run_step2") as step2, _spy("_cv_step3") as cv3:
            result = multi_metric(_pipeline(), _store(), metrics=METRICS)
        step1_winners = {
            select_best(r.step1_agg_df, metric=METRIC_COL[m], minimize=METRIC_MINIMIZE[m], n_folds=3)["model_name"]
            for m, r in result.results.items()
        }
        assert step2.call_count == cv3.call_count == len(step1_winners)

    def test_reuse_step4_model_tunes_once_per_selection(self):
        with _mock_selector(), _spy("_tune_step4") as tune:
            result = multi_metric(_pipeline(), _store(), metrics=METRICS, reuse_step4="model")
        features = result.runs_df["Metric"].map(lambda m: ",".join(result.selected_features[m]))
        keys = result.runs_df[["Class_weight", "Model", "Feature_stage"]].astype(str).assign(Features=features)
        keys = keys.agg("|".join, axis=1)
        assert tune.call_count == keys.nunique()
        shared = result.runs_df.dropna(subset=["Step4_shared_with"])
        assert len(shared) == len(METRICS) - keys.nunique()
        assert result.summary().iloc[0]["N_hpo_runs"] == keys.nunique()

    def test_reuse_step4_never_crosses_feature_sets(self):
        with _mock_selector():
            result = multi_metric(_pipeline(), _store(), metrics=METRICS, reuse_step4="model")
        for _, row in result.runs_df.dropna(subset=["Step4_shared_with"]).iterrows():
            source = row["Step4_shared_with"]
            assert result.selected_features[row["Metric"]] == result.selected_features[source]

    def test_source_pipeline_weights_do_not_leak(self):
        source = _pipeline(handle_imbalance=True)
        with _mock_selector():
            result = multi_metric(source, _store(), metrics=["Brier", "AUC"], handle_imbalance={"Brier": False})
        assert not result.results["Brier"].class_weight_balanced
        assert result.results["AUC"].class_weight_balanced
        brier_lr = result.results["Brier"]
        if brier_lr.best_model_name == "LR":
            assert brier_lr.best_base_estimator.class_weight is None
        # the caller's pipeline keeps its own (weighted) models
        assert source._model_map["LR"].estimator.class_weight == "balanced"

    def test_class_weight_decision_logged_once(self):
        logged = []
        handler_id = logger.add(lambda msg: logged.append(msg), level="INFO")
        try:
            with _mock_selector():
                multi_metric(
                    _pipeline(handle_imbalance="auto"),
                    _store(),
                    metrics=["Brier", "AUC", "F1"],
                    handle_imbalance={"Brier": False},
                )
        finally:
            logger.remove(handler_id)
        lines = [m for m in logged if "handle_imbalance" in m]
        assert len(lines) == 1
        assert "Brier → off" in lines[0] and "AUC, F1 → not applied" in lines[0]


class TestFailures:
    def test_failed_metric_recorded(self):
        original = H2MLPipeline._select_step4

        def flaky(self, result, tuned, fixed):
            if self.config.metric == "F1":
                raise RuntimeError("boom")
            return original(self, result, tuned, fixed)

        with _mock_selector(), patch.object(H2MLPipeline, "_select_step4", flaky):
            result = multi_metric(_pipeline(), _store(), metrics=["AUC", "F1"])
        assert result.failed_metrics == ["F1"]
        assert list(result.results) == ["AUC"]


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"metrics": ["AUC", "R2"]}, "not classification metrics"),
            ({"metrics": ["Accuracy"]}, "not classification metrics"),
            ({"metrics": []}, "at least one"),
            ({"metrics": ["AUC"], "handle_imbalance": {"Brier": False}}, "not compared"),
            ({"reuse_step4": "all"}, "reuse_step4"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            multi_metric(_pipeline(), _store(), **kwargs)

    def test_defaults_to_all_task_metrics(self):
        with _mock_selector():
            result = multi_metric(_pipeline(n_trials=1), _store(), reuse_step4="model")
        assert sorted(result.metrics) == sorted(METRICS)

    def test_transforms_rejected_for_classification(self):
        with pytest.raises(ValueError, match="regression"):
            multi_metric(_pipeline(), _store(), metrics=["AUC"], transforms=TRANSFORMS)


@pytest.fixture(scope="module")
def reg_mm() -> MultiMetricResult:
    with _mock_selector():
        return multi_metric(_reg_pipeline(holdout_frac=0.2), _reg_store(), metrics=REG_METRICS, transforms=TRANSFORMS)


class TestRegression:
    @pytest.mark.parametrize("metric", REG_METRICS)
    def test_matches_separate_run(self, reg_mm, metric):
        with _mock_selector():
            separate = _reg_pipeline(holdout_frac=0.2, metric=metric).run(_reg_store(), transforms=TRANSFORMS)
        shared = reg_mm.results[metric]

        assert shared.metric == separate.metric == metric
        for name in ("best_model_name", "best_stage", "best_feature_stage", "y_transform"):
            assert getattr(shared, name) == getattr(separate, name), name
        assert shared.best_model_value == pytest.approx(separate.best_model_value)
        assert shared.best_model_std == pytest.approx(separate.best_model_std)
        assert shared.best_params == separate.best_params
        assert shared.cv_warnings == separate.cv_warnings
        np.testing.assert_array_equal(shared.holdout_idx, separate.holdout_idx)
        for name in ("holdout_df", "step1_agg_df", "step3_agg_df", "step4_agg_df"):
            _assert_scores_equal(getattr(shared, name), getattr(separate, name))

    def test_runs_df_and_holdout(self, reg_mm):
        assert list(reg_mm.runs_df["Metric"]) == REG_METRICS
        assert not reg_mm.runs_df["Class_weight"].any()
        assert set(reg_mm.runs_df["Y_transform"].dropna()) <= {"log"}
        for metric in REG_METRICS:
            row = reg_mm.runs_df.set_index("Metric").loc[metric]
            assert row["Holdout"] == pytest.approx(reg_mm.results[metric].holdout_df.iloc[0][METRIC_COL[metric]])

    def test_step1_cv_runs_once(self):
        with _mock_selector(), _spy("_cv_step1") as cv1:
            multi_metric(_reg_pipeline(), _reg_store(), metrics=REG_METRICS, transforms=TRANSFORMS)
        assert cv1.call_count == 1

    def test_reuse_step4_keyed_on_transform_and_features(self):
        with _mock_selector(), _spy("_tune_step4") as tune:
            result = multi_metric(
                _reg_pipeline(), _reg_store(), metrics=REG_METRICS, transforms=TRANSFORMS, reuse_step4="model"
            )
        runs = result.runs_df
        features = runs["Metric"].map(lambda m: ",".join(result.selected_features[m]))
        keys = runs[["Model", "Y_transform", "Feature_stage"]].astype(str).assign(Features=features)
        keys = keys.agg("|".join, axis=1)
        assert tune.call_count == keys.nunique()
        for _, row in runs.dropna(subset=["Step4_shared_with"]).iterrows():
            source = runs.set_index("Metric").loc[row["Step4_shared_with"]]
            assert row["Y_transform"] == source["Y_transform"]
            assert result.selected_features[row["Metric"]] == result.selected_features[row["Step4_shared_with"]]

    def test_defaults_to_all_regression_metrics_in_order(self):
        with _mock_selector():
            result = multi_metric(_reg_pipeline(n_trials=1), _reg_store(), reuse_step4="model")
        assert result.metrics == REG_METRICS

    def test_save_load_round_trip(self, reg_mm, tmp_path):
        for result in reg_mm.results.values():
            result.selector = None  # the mocked selector cannot be pickled
        reg_mm.save(tmp_path / "reg_mm")
        loaded = MultiMetricResult.load(tmp_path / "reg_mm")
        assert loaded.metrics == REG_METRICS
        assert loaded.results["RMSE"].y_transform == reg_mm.results["RMSE"].y_transform
        pd.testing.assert_frame_equal(loaded.runs_df, reg_mm.runs_df, check_dtype=False)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"metrics": ["R2", "AUC"]}, "not regression metrics"),
            ({"metrics": ["R2"], "handle_imbalance": {"R2": False}}, "classification only"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            multi_metric(_reg_pipeline(), _reg_store(), **kwargs)
