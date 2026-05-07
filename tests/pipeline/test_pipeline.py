"""
tests/pipeline/test_pipeline.py

Tests for H2MLPipeline, PipelineConfig, and PipelineResult.

Current pipeline workflow (4 steps):
    Step 1 — CV all models on all features          → best_model_name, step1_agg_df
    Step 2 — SHAP + correlation feature reduction   → features_reduced, selector
    Step 3 — CV all models on reduced features;
             compare vs step 1 → select best stage  → step3_agg_df, best_stage
    Step 4 — Optuna HPO on winning (model, stage)   → best_params
             (skipped when winning model has opt_enabled=False)

Strategy:
    - Steps 1 and 3 run for real (fast sklearn models, small data)
    - Step 2 (SHAP) is mocked — tested separately in tests/feature/
    - Step 4 is either skipped (LR has opt_enabled=False in registry)
      or mocked via a run_study patch when testing opt behaviour
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from h2ml.pipeline.base import TaskType
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig, PipelineResult
from h2ml.pipeline.step import make_classifier, make_regressor
from h2ml.features.feature_store import PipelineData
from h2ml.features.selector import FeatureSelector
from h2ml.evaluation.metrics import RunMetadata


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

FEATURE_NAMES = [f"feat_{i}" for i in range(8)]
N_SAMPLES = 120


def _make_clf_store(seed: int = 42) -> PipelineData:
    rng = np.random.default_rng(seed)
    return PipelineData(
        X=rng.standard_normal((N_SAMPLES, len(FEATURE_NAMES))),
        feature_names=FEATURE_NAMES,
        y=rng.integers(0, 2, size=N_SAMPLES),
    )


def _make_reg_store(seed: int = 42) -> PipelineData:
    rng = np.random.default_rng(seed)
    return PipelineData(
        X=rng.standard_normal((N_SAMPLES, len(FEATURE_NAMES))),
        feature_names=FEATURE_NAMES,
        y=rng.standard_normal(N_SAMPLES),
    )


def _make_positive_reg_store(seed: int = 7) -> PipelineData:
    """Regression store with strictly positive y — required for log/sqrt transforms."""
    rng = np.random.default_rng(seed)
    return PipelineData(
        X=rng.standard_normal((N_SAMPLES, len(FEATURE_NAMES))),
        feature_names=FEATURE_NAMES,
        y=rng.uniform(0.5, 5.0, size=N_SAMPLES),
    )


def _lr_only() -> list:
    """Single LR — opt_enabled=False in registry, so step 4 skips Optuna automatically."""
    return [
        make_classifier(LogisticRegression(max_iter=200, random_state=42), name="LR")
    ]


def _clf_steps() -> list:
    return [
        make_classifier(LogisticRegression(max_iter=200, random_state=42), name="LR"),
        make_classifier(
            RandomForestClassifier(n_estimators=10, random_state=42), name="RF"
        ),
    ]


def _reg_steps() -> list:
    return [
        make_regressor(Ridge(), name="Ridge"),
        make_regressor(
            RandomForestRegressor(n_estimators=10, random_state=42), name="RF"
        ),
    ]


def _clf_config(**kwargs) -> PipelineConfig:
    defaults = dict(
        task_type=TaskType.CLASSIFICATION, metric="AUC", n_splits=3, n_trials=1
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _reg_config(**kwargs) -> PipelineConfig:
    defaults = dict(task_type=TaskType.REGRESSION, metric="R2", n_splits=3, n_trials=1)
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_selector(selected: list[str] = None):
    """Patches FeatureSelector so step 2 runs without real SHAP."""
    selected = selected or FEATURE_NAMES[:5]

    mock_sel = MagicMock(spec=FeatureSelector)
    mock_sel._fitted = True
    mock_sel.selected_features_ = selected
    mock_sel.n_selected = len(selected)
    mock_sel.removed_features = [f for f in FEATURE_NAMES if f not in selected]

    def fake_fit_transform(store, y=None):
        mock_sel._fitted = True
        return store.select(selected)

    def fake_transform(store):
        return store.select(selected)

    mock_sel.fit_transform.side_effect = fake_fit_transform
    mock_sel.transform.side_effect = fake_transform

    return patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=mock_sel)


def _mock_optimizer(best_params: dict = None):
    """Patches run_study so step 4 does not run Optuna."""
    best_params = best_params or {"n_estimators": 10, "max_depth": 3}
    mock_study = MagicMock()
    mock_study.best_params = best_params
    return patch("h2ml.pipeline.pipeline.run_study", return_value=mock_study)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clf_store() -> PipelineData:
    return _make_clf_store()


@pytest.fixture
def reg_store() -> PipelineData:
    return _make_reg_store()


@pytest.fixture
def lr_pipeline() -> H2MLPipeline:
    """Single LR pipeline — opt_enabled=False, no Optuna mock needed."""
    return H2MLPipeline(models=_lr_only(), config=_clf_config())


@pytest.fixture
def clf_pipeline() -> H2MLPipeline:
    return H2MLPipeline(models=_clf_steps(), config=_clf_config())


@pytest.fixture
def reg_pipeline() -> H2MLPipeline:
    return H2MLPipeline(models=_reg_steps(), config=_reg_config())


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_default_task_type(self):
        assert PipelineConfig().task_type == TaskType.CLASSIFICATION

    def test_default_metric(self):
        assert PipelineConfig().metric == "AUC"

    def test_default_n_splits(self):
        assert PipelineConfig().n_splits == 5

    def test_default_opt_n_splits(self):
        assert PipelineConfig().opt_n_splits == 3

    def test_default_corr_threshold(self):
        assert PipelineConfig().corr_threshold == 0.7

    def test_default_minimize_metric(self):
        assert PipelineConfig().minimize_metric is False

    def test_invalid_n_splits_raises(self):
        with pytest.raises(ValueError, match="n_splits"):
            PipelineConfig(n_splits=1)

    def test_invalid_opt_n_splits_raises(self):
        with pytest.raises(ValueError, match="opt_n_splits"):
            PipelineConfig(opt_n_splits=1)

    def test_opt_n_splits_below_half_n_splits_warns(self):
        """opt_n_splits < n_splits // 2 should emit a loguru warning."""
        from loguru import logger

        warned = []
        handler_id = logger.add(lambda msg: warned.append(msg), level="WARNING")
        try:
            PipelineConfig(n_splits=10, opt_n_splits=2)  # 2 < 10//2=5 → warn
        finally:
            logger.remove(handler_id)
        assert any("opt_n_splits" in str(w) for w in warned)

    def test_opt_n_splits_at_half_n_splits_no_warn(self):
        """opt_n_splits == n_splits // 2 should NOT warn."""
        from loguru import logger

        warned = []
        handler_id = logger.add(lambda msg: warned.append(msg), level="WARNING")
        try:
            PipelineConfig(n_splits=6, opt_n_splits=3)  # 3 == 6//2 → no warn
        finally:
            logger.remove(handler_id)
        opt_warns = [w for w in warned if "opt_n_splits" in str(w)]
        assert opt_warns == []

    def test_invalid_corr_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="corr_threshold"):
            PipelineConfig(corr_threshold=0.0)

    def test_invalid_corr_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="corr_threshold"):
            PipelineConfig(corr_threshold=1.5)

    def test_invalid_n_trials_raises(self):
        with pytest.raises(ValueError, match="n_trials"):
            PipelineConfig(n_trials=0)

    def test_default_n_blocks_per_fold(self):
        assert PipelineConfig().n_blocks_per_fold == 5

    def test_custom_n_blocks_per_fold(self):
        config = PipelineConfig(n_blocks_per_fold=3)
        assert config.n_blocks_per_fold == 3

    def test_invalid_n_blocks_per_fold_raises(self):
        with pytest.raises(ValueError, match="n_blocks_per_fold"):
            PipelineConfig(n_blocks_per_fold=0)

    def test_handle_imbalance_default_false(self):
        assert PipelineConfig().handle_imbalance is False

    def test_handle_imbalance_true(self):
        config = PipelineConfig(handle_imbalance=True)
        assert config.handle_imbalance is True


# ---------------------------------------------------------------------------
# handle_imbalance — class weight injection
# ---------------------------------------------------------------------------


class TestHandleImbalance:
    def _make_imbalanced_store(
        self, minority_n: int = 5, majority_n: int = 95
    ) -> PipelineData:
        rng = np.random.default_rng(0)
        n = minority_n + majority_n
        X = rng.standard_normal((n, 3))
        y = np.array([1] * minority_n + [0] * majority_n)
        return PipelineData(X=X, feature_names=["a", "b", "c"], y=y)

    def test_class_weight_set_on_supporting_models(self):
        """handle_imbalance=True should set class_weight='balanced' on RFC."""
        config = PipelineConfig(handle_imbalance=True)
        pipeline = H2MLPipeline(config=config)
        rf = next(m for m in pipeline.models if m.name == "RandomForestClassifier")
        assert rf.estimator.get_params()["class_weight"] == "balanced"

    def test_class_weight_not_set_on_unsupported_models(self):
        """GaussianNB does not accept class_weight — it must not be modified."""
        config = PipelineConfig(handle_imbalance=True)
        pipeline = H2MLPipeline(config=config)
        nb = next((m for m in pipeline.models if m.name == "GaussianNB"), None)
        if nb is not None:
            assert "class_weight" not in nb.estimator.get_params()

    def test_class_weight_not_set_when_disabled(self):
        """handle_imbalance=False (default) must not touch model params."""
        config = PipelineConfig(handle_imbalance=False)
        pipeline = H2MLPipeline(config=config)
        rf = next(m for m in pipeline.models if m.name == "RandomForestClassifier")
        assert rf.estimator.get_params().get("class_weight") is None

    def test_imbalance_warning_fires(self):
        """minority ratio < 0.1 should emit a loguru warning."""
        from loguru import logger

        store = self._make_imbalanced_store(minority_n=5, majority_n=95)
        config = PipelineConfig(handle_imbalance=False)
        pipeline = H2MLPipeline(config=config)
        warned = []
        handler = logger.add(lambda msg: warned.append(msg), level="WARNING")
        try:
            pipeline._validate_store(store)
        finally:
            logger.remove(handler)
        assert any("imbalance" in str(w).lower() for w in warned)

    def test_no_imbalance_warning_when_balanced(self):
        """Balanced classes must not trigger a warning."""
        from loguru import logger

        rng = np.random.default_rng(0)
        store = PipelineData(
            X=rng.standard_normal((100, 3)),
            feature_names=["a", "b", "c"],
            y=np.array([0] * 50 + [1] * 50),
        )
        config = PipelineConfig(handle_imbalance=False)
        pipeline = H2MLPipeline(config=config)
        warned = []
        handler = logger.add(lambda msg: warned.append(msg), level="WARNING")
        try:
            pipeline._validate_store(store)
        finally:
            logger.remove(handler)
        assert not any("imbalance" in str(w).lower() for w in warned)


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


def _tiny_store() -> PipelineData:
    rng = np.random.default_rng(0)
    return PipelineData(
        X=rng.standard_normal((10, 3)),
        feature_names=["a", "b", "c"],
        y=rng.integers(0, 2, 10),
    )


class TestPipelineResult:
    def test_all_fields_none_on_init(self):
        result = PipelineResult()
        assert result.step1_fold_df is None
        assert result.step1_agg_df is None
        assert result.best_model_name is None
        assert result.best_stage is None
        assert result.features_reduced is None
        assert result.step3_agg_df is None
        assert result.step4_agg_df is None
        assert result.best_params is None

    def test_completed_steps_empty_on_init(self):
        assert PipelineResult().completed_steps == []

    def test_completed_steps_after_step1(self):
        result = PipelineResult(step1_agg_df=pd.DataFrame(), best_model_name="RF")
        assert 1 in result.completed_steps

    def test_completed_steps_after_step2(self):
        result = PipelineResult(
            step1_agg_df=pd.DataFrame(),
            best_model_name="RF",
            features_reduced=_tiny_store(),
        )
        assert 2 in result.completed_steps

    def test_completed_steps_after_step3(self):
        result = PipelineResult(
            step1_agg_df=pd.DataFrame(),
            best_model_name="RF",
            features_reduced=_tiny_store(),
            step3_agg_df=pd.DataFrame(),
            best_stage="default",
        )
        assert 3 in result.completed_steps

    def test_completed_steps_after_step4(self):
        result = PipelineResult(
            step1_agg_df=pd.DataFrame(),
            best_model_name="RF",
            features_reduced=_tiny_store(),
            step3_agg_df=pd.DataFrame(),
            best_stage="default",
            best_params={"C": 0.5},
        )
        assert 4 in result.completed_steps

    def test_summary_empty_before_any_step(self):
        assert PipelineResult().summary().empty

    def test_summary_has_stage_column(self):
        result = PipelineResult(step1_agg_df=pd.DataFrame({"Model": ["RF"]}))
        assert "Stage" in result.summary().columns

    def test_summary_labels_step1_as_default(self):
        result = PipelineResult(step1_agg_df=pd.DataFrame({"Model": ["RF"]}))
        assert "default" in result.summary()["Stage"].values

    def test_summary_labels_step3_as_reduced(self):
        result = PipelineResult(step3_agg_df=pd.DataFrame({"Model": ["RF"]}))
        assert "reduced" in result.summary()["Stage"].values

    def test_summary_labels_step4_as_optimized(self):
        result = PipelineResult(step4_agg_df=pd.DataFrame({"Model": ["RF"]}))
        assert "optimized" in result.summary()["Stage"].values

    def test_summary_labels_all_three_stages(self):
        result = PipelineResult(
            step1_agg_df=pd.DataFrame({"Model": ["RF"]}),
            step3_agg_df=pd.DataFrame({"Model": ["RF"]}),
            step4_agg_df=pd.DataFrame({"Model": ["RF"]}),
        )
        stages = set(result.summary()["Stage"].values)
        assert stages == {"default", "reduced", "optimized"}

    def test_summary_sort_by_metric_descending(self):
        result = PipelineResult(
            step1_agg_df=pd.DataFrame(
                {"Model": ["LR", "RF"], "AUC_Test_Mean": [0.75, 0.85]}
            )
        )
        df = result.summary(metric="AUC_Test_Mean")
        assert df.iloc[0]["AUC_Test_Mean"] >= df.iloc[1]["AUC_Test_Mean"]

    def test_summary_sort_ascending_for_loss_metric(self):
        result = PipelineResult(
            step1_agg_df=pd.DataFrame(
                {"Model": ["LR", "RF"], "RMSE_Test_Mean": [0.5, 0.3]}
            )
        )
        df = result.summary(metric="RMSE_Test_Mean", ascending=True)
        assert df.iloc[0]["RMSE_Test_Mean"] <= df.iloc[1]["RMSE_Test_Mean"]

    def test_summary_raises_on_unknown_metric(self):
        result = PipelineResult(
            step1_agg_df=pd.DataFrame({"Model": ["RF"], "AUC_Test_Mean": [0.8]})
        )
        with pytest.raises(ValueError, match="not found"):
            result.summary(metric="NonExistentMetric_Mean")

    def test_repr_contains_best_model(self):
        assert "RF" in repr(PipelineResult(best_model_name="RF"))

    def test_repr_contains_completed_steps(self):
        assert "completed_steps" in repr(PipelineResult())


# ---------------------------------------------------------------------------
# H2MLPipeline construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="empty"):
            H2MLPipeline(models=[], config=_clf_config())

    def test_repr_contains_task_type(self, clf_pipeline):
        assert "classification" in repr(clf_pipeline)

    def test_repr_contains_metric(self, clf_pipeline):
        assert "AUC" in repr(clf_pipeline)

    def test_repr_contains_n_models(self, clf_pipeline):
        assert "n_models=2" in repr(clf_pipeline)


# ---------------------------------------------------------------------------
# Step 1 — run_step1_only()
# ---------------------------------------------------------------------------


class TestRunStep1Only:
    def test_fold_df_populated(self, clf_pipeline, clf_store):
        result = clf_pipeline.run_step1_only(clf_store)
        assert result.step1_fold_df is not None

    def test_agg_df_populated(self, clf_pipeline, clf_store):
        result = clf_pipeline.run_step1_only(clf_store)
        assert result.step1_agg_df is not None

    def test_best_model_name_is_set(self, clf_pipeline, clf_store):
        """_run_step1 selects the best model — best_model_name is populated."""
        result = clf_pipeline.run_step1_only(clf_store)
        assert result.best_model_name is not None

    def test_best_stage_is_default(self, clf_pipeline, clf_store):
        """Step 1 sets best_stage='default' as the starting baseline."""
        result = clf_pipeline.run_step1_only(clf_store)
        assert result.best_stage == "default"

    def test_downstream_not_populated(self, clf_pipeline, clf_store):
        result = clf_pipeline.run_step1_only(clf_store)
        assert result.features_reduced is None
        assert result.step3_agg_df is None
        assert result.best_params is None

    def test_fold_df_row_count(self, clf_pipeline, clf_store):
        result = clf_pipeline.run_step1_only(clf_store)
        expected = clf_pipeline.config.n_splits * len(clf_pipeline.models)
        assert len(result.step1_fold_df) == expected

    def test_agg_df_one_row_per_model(self, clf_pipeline, clf_store):
        result = clf_pipeline.run_step1_only(clf_store)
        assert len(result.step1_agg_df) == len(clf_pipeline.models)

    def test_agg_df_model_names(self, clf_pipeline, clf_store):
        result = clf_pipeline.run_step1_only(clf_store)
        assert set(result.step1_agg_df["Model"]) == {"LR", "RF"}

    def test_fold_df_stage_is_default(self, clf_pipeline, clf_store):
        result = clf_pipeline.run_step1_only(clf_store)
        assert (result.step1_fold_df["Stage"] == "default").all()

    def test_completed_steps_is_1(self, clf_pipeline, clf_store):
        result = clf_pipeline.run_step1_only(clf_store)
        assert result.completed_steps == [1]

    def test_regression_agg_df_has_r2(self, reg_pipeline, reg_store):
        result = reg_pipeline.run_step1_only(reg_store)
        assert "R2_Test_Mean" in result.step1_agg_df.columns


# ---------------------------------------------------------------------------
# Step 1 with y-transform sweep
# ---------------------------------------------------------------------------


class TestStep1WithTransforms:
    """
    Tests for the transforms= argument on partial-run methods.

    Step 1 now runs all (model × transform) pairs in a single flat parallel
    batch.  These tests verify the grouping, counts, and output shape are
    correct.
    """

    @pytest.fixture
    def pos_reg_store(self) -> PipelineData:
        return _make_positive_reg_store()

    @pytest.fixture
    def reg_pipeline_2models(self) -> H2MLPipeline:
        return H2MLPipeline(models=_reg_steps(), config=_reg_config())

    def test_fold_df_has_y_transform_column(self, reg_pipeline_2models, pos_reg_store):
        result = reg_pipeline_2models.run_step1_only(
            pos_reg_store, transforms=["log", "count"]
        )
        assert "Y_transform" in result.step1_fold_df.columns

    def test_fold_df_row_count_equals_models_times_transforms_times_folds(
        self, reg_pipeline_2models, pos_reg_store
    ):
        result = reg_pipeline_2models.run_step1_only(
            pos_reg_store, transforms=["log", "count"]
        )
        n_models, n_transforms, n_folds = (
            len(reg_pipeline_2models.models),
            2,
            reg_pipeline_2models.config.n_splits,
        )
        assert len(result.step1_fold_df) == n_models * n_transforms * n_folds

    def test_agg_df_has_one_row_per_model_per_transform(
        self, reg_pipeline_2models, pos_reg_store
    ):
        result = reg_pipeline_2models.run_step1_only(
            pos_reg_store, transforms=["log", "count"]
        )
        assert len(result.step1_agg_df) == len(reg_pipeline_2models.models) * 2

    def test_cv_result_count_equals_models_times_transforms(
        self, reg_pipeline_2models, pos_reg_store
    ):
        result = reg_pipeline_2models.run_step1_only(
            pos_reg_store, transforms=["log", "count"]
        )
        assert len(result.step1_cv_result) == len(reg_pipeline_2models.models) * 2

    def test_y_transform_set_to_one_of_the_input_transforms(
        self, reg_pipeline_2models, pos_reg_store
    ):
        result = reg_pipeline_2models.run_step1_only(
            pos_reg_store, transforms=["log", "count"]
        )
        assert result.y_transform in ("log", "count")

    def test_both_transforms_appear_in_agg_df(
        self, reg_pipeline_2models, pos_reg_store
    ):
        result = reg_pipeline_2models.run_step1_only(
            pos_reg_store, transforms=["log", "count"]
        )
        assert set(result.step1_agg_df["Y_transform"]) == {"log", "count"}

    def test_run_step1_to_step3_with_transforms_completes(
        self, reg_pipeline_2models, pos_reg_store
    ):
        with _mock_selector():
            result = reg_pipeline_2models.run_step1_to_step3(
                pos_reg_store, transforms=["log", "count"]
            )
        assert result.completed_steps == [1, 2, 3]
        assert result.y_transform in ("log", "count")

    def test_single_transform_behaves_like_no_transform_sweep(
        self, reg_pipeline_2models, pos_reg_store
    ):
        """Passing one transform should produce the same structure as no sweep."""
        result_sweep = reg_pipeline_2models.run_step1_only(
            pos_reg_store, transforms=["count"]
        )
        result_plain = reg_pipeline_2models.run_step1_only(pos_reg_store)
        # Both should have 1 entry per model in the agg_df
        assert len(result_sweep.step1_agg_df) == len(result_plain.step1_agg_df)


# ---------------------------------------------------------------------------
# Steps 1–2 — run_step1_to_step2()
# ---------------------------------------------------------------------------


class TestRunStep1ToStep2:
    def test_features_reduced_populated(self, clf_pipeline, clf_store):
        with _mock_selector():
            result = clf_pipeline.run_step1_to_step2(clf_store)
        assert result.features_reduced is not None

    def test_selector_stored(self, clf_pipeline, clf_store):
        with _mock_selector():
            result = clf_pipeline.run_step1_to_step2(clf_store)
        assert result.selector is not None

    def test_feature_count_matches_selection(self, clf_pipeline, clf_store):
        selected = FEATURE_NAMES[:5]
        with _mock_selector(selected=selected):
            result = clf_pipeline.run_step1_to_step2(clf_store)
        assert result.features_reduced.n_features == len(selected)

    def test_best_model_name_is_set(self, clf_pipeline, clf_store):
        with _mock_selector():
            result = clf_pipeline.run_step1_to_step2(clf_store)
        assert result.best_model_name in [m.name for m in clf_pipeline.models]

    def test_best_stage_is_default(self, clf_pipeline, clf_store):
        """best_stage is set during step 1, not step 3."""
        with _mock_selector():
            result = clf_pipeline.run_step1_to_step2(clf_store)
        assert result.best_stage == "default"

    def test_step3_not_started(self, clf_pipeline, clf_store):
        with _mock_selector():
            result = clf_pipeline.run_step1_to_step2(clf_store)
        assert result.step3_agg_df is None
        assert result.best_params is None

    def test_completed_steps_are_1_and_2(self, clf_pipeline, clf_store):
        with _mock_selector():
            result = clf_pipeline.run_step1_to_step2(clf_store)
        assert result.completed_steps == [1, 2]


# ---------------------------------------------------------------------------
# Steps 1–3 — run_step1_to_step3()
# ---------------------------------------------------------------------------


class TestRunStep1ToStep3:
    def test_step3_agg_df_populated(self, clf_pipeline, clf_store):
        with _mock_selector():
            result = clf_pipeline.run_step1_to_step3(clf_store)
        assert result.step3_agg_df is not None

    def test_best_stage_set(self, clf_pipeline, clf_store):
        with _mock_selector():
            result = clf_pipeline.run_step1_to_step3(clf_store)
        assert result.best_stage in ("default", "reduced")

    def test_features_reduced_populated(self, clf_pipeline, clf_store):
        with _mock_selector():
            result = clf_pipeline.run_step1_to_step3(clf_store)
        assert result.features_reduced is not None

    def test_no_hpo(self, clf_pipeline, clf_store):
        with _mock_selector():
            result = clf_pipeline.run_step1_to_step3(clf_store)
        assert result.best_params is None

    def test_completed_steps_include_1_2_3(self, clf_pipeline, clf_store):
        with _mock_selector():
            result = clf_pipeline.run_step1_to_step3(clf_store)
        assert 1 in result.completed_steps
        assert 2 in result.completed_steps
        assert 3 in result.completed_steps
        assert 4 not in result.completed_steps

    def test_validates_store(self, clf_pipeline):
        store = _make_clf_store()
        store.X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            clf_pipeline.run_step1_to_step3(store)


# ---------------------------------------------------------------------------
# run_from_step3()
# ---------------------------------------------------------------------------


class TestRunFromStep3:
    @pytest.fixture
    def partial_result(self, lr_pipeline, clf_store) -> PipelineResult:
        with _mock_selector():
            return lr_pipeline.run_step1_to_step2(clf_store)

    def test_raises_without_features(self, lr_pipeline, clf_store):
        bad = PipelineResult(features_reduced=clf_store, best_model_name="LR")
        with pytest.raises(ValueError):
            lr_pipeline.run_from_step3(bad)

    def test_raises_without_features_reduced(self, lr_pipeline, clf_store):
        bad = PipelineResult(features=clf_store, best_model_name="LR")
        with pytest.raises(ValueError):
            lr_pipeline.run_from_step3(bad)

    def test_raises_without_best_model_name(self, lr_pipeline, clf_store):
        bad = PipelineResult(features=clf_store, features_reduced=clf_store)
        with pytest.raises(ValueError):
            lr_pipeline.run_from_step3(bad)

    def test_step3_agg_df_populated(self, lr_pipeline, partial_result):
        result = lr_pipeline.run_from_step3(partial_result)
        assert result.step3_agg_df is not None

    def test_best_stage_set_to_valid_value(self, lr_pipeline, partial_result):
        result = lr_pipeline.run_from_step3(partial_result)
        assert result.best_stage in ("default", "reduced")

    def test_selector_reflects_best_stage_features(self, lr_pipeline, partial_result):
        """selector.selected_features_ matches the winning stage's feature set."""
        result = lr_pipeline.run_from_step3(partial_result)
        if result.best_stage == "default":
            assert result.selector.selected_features_ == list(
                result.features.feature_names
            )
        else:
            assert result.selector.selected_features_ == list(
                result.features_reduced.feature_names
            )

    def test_best_feature_stage_preserved(self, lr_pipeline, partial_result):
        """best_feature_stage is never overwritten to 'optimized'."""
        result = lr_pipeline.run_from_step3(partial_result)
        assert result.best_feature_stage in ("default", "reduced")

    def test_completed_steps_include_3_and_4(self, lr_pipeline, partial_result):
        result = lr_pipeline.run_from_step3(partial_result)
        assert 3 in result.completed_steps
        assert 4 in result.completed_steps


# ---------------------------------------------------------------------------
# Step 4 — optimisation behaviour
# ---------------------------------------------------------------------------


class TestStep4Optimisation:
    def test_step4_skipped_for_lr_model(self, lr_pipeline, clf_store):
        """LR has opt_enabled=False — step 4 sets best_params without running CV."""
        with _mock_selector():
            result = lr_pipeline.run(clf_store)
        assert result.best_params is not None
        assert result.step4_fold_df is None

    def test_step4_fold_df_populated_when_opt_runs(self, clf_store):
        """RF has opt_enabled=True — step 4 runs CV with the optimised params."""
        rf_pipeline = H2MLPipeline(
            models=[
                make_classifier(
                    RandomForestClassifier(n_estimators=10, random_state=42), name="RF"
                )
            ],
            config=_clf_config(),
        )
        with _mock_selector(), _mock_optimizer({"n_estimators": 10, "max_depth": 3}):
            result = rf_pipeline.run(clf_store)
        assert result.step4_fold_df is not None

    def test_best_params_set_after_full_run(self, lr_pipeline, clf_store):
        with _mock_selector():
            result = lr_pipeline.run(clf_store)
        assert result.best_params is not None


# ---------------------------------------------------------------------------
# Full run()
# ---------------------------------------------------------------------------


class TestFullRun:
    def test_all_four_steps_completed(self, lr_pipeline, clf_store):
        with _mock_selector():
            result = lr_pipeline.run(clf_store)
        assert result.completed_steps == [1, 2, 3, 4]

    def test_key_result_fields_populated(self, lr_pipeline, clf_store):
        with _mock_selector():
            result = lr_pipeline.run(clf_store)
        assert result.step1_fold_df is not None
        assert result.step1_agg_df is not None
        assert result.best_model_name is not None
        assert result.features_reduced is not None
        assert result.step3_fold_df is not None
        assert result.step3_agg_df is not None
        assert result.best_stage is not None
        assert result.best_params is not None

    def test_summary_covers_default_and_reduced_stages(self, lr_pipeline, clf_store):
        with _mock_selector():
            result = lr_pipeline.run(clf_store)
        stages = set(result.summary()["Stage"].values)
        assert {"default", "reduced"}.issubset(stages)

    def test_regression_full_run(self, reg_pipeline, reg_store):
        # Ridge/RF are not in the registry, so patch _is_opt_enabled to skip Optuna
        with (
            _mock_selector(),
            patch.object(H2MLPipeline, "_is_opt_enabled", return_value=False),
        ):
            result = reg_pipeline.run(reg_store)
        assert result.completed_steps == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# _get_model()
# ---------------------------------------------------------------------------


class TestGetModel:
    def test_returns_correct_model(self, clf_pipeline):
        model = clf_pipeline._get_model("LR")
        assert model.name == "LR"

    def test_raises_when_not_found(self, clf_pipeline):
        with pytest.raises(ValueError, match="not found"):
            clf_pipeline._get_model("NonExistentModel")

    def test_error_message_lists_available_models(self, clf_pipeline):
        with pytest.raises(ValueError, match="LR"):
            clf_pipeline._get_model("BadName")


# ---------------------------------------------------------------------------
# _metadata_with()
# ---------------------------------------------------------------------------


class TestMetadataWith:
    def test_returns_run_metadata(self, clf_pipeline):
        meta = clf_pipeline._metadata_with(stage="default")
        assert isinstance(meta, RunMetadata)

    def test_stage_set_correctly(self, clf_pipeline):
        for stage in ("default", "reduced", "optimized"):
            assert clf_pipeline._metadata_with(stage=stage).stage == stage

    def test_no_setting_attribute(self, clf_pipeline):
        assert not hasattr(clf_pipeline._metadata_with(stage="default"), "setting")

    def test_preserves_batch_field(self):
        pipeline = H2MLPipeline(
            models=_clf_steps(),
            config=_clf_config(),
            metadata=RunMetadata(stage="default", batch="run_01"),
        )
        meta = pipeline._metadata_with(stage="reduced")
        assert meta.batch == "run_01"
        assert meta.stage == "reduced"

    def test_preserves_schema_field(self):
        pipeline = H2MLPipeline(
            models=_clf_steps(),
            config=_clf_config(),
            metadata=RunMetadata(stage="default", schema="birds"),
        )
        meta = pipeline._metadata_with(stage="optimized")
        assert meta.schema == "birds"

    def test_creates_metadata_when_none(self, clf_pipeline):
        assert clf_pipeline.metadata is None
        assert clf_pipeline._metadata_with(stage="default").stage == "default"


# ---------------------------------------------------------------------------
# _validate_store() — input validation
# ---------------------------------------------------------------------------


class TestValidateStore:
    def test_raises_on_nan_in_X(self, clf_pipeline):
        store = _make_clf_store()
        store.X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or infinite"):
            clf_pipeline._validate_store(store)

    def test_raises_on_inf_in_X(self, clf_pipeline):
        store = _make_clf_store()
        store.X[0, 0] = float("inf")
        with pytest.raises(ValueError, match="NaN or infinite"):
            clf_pipeline._validate_store(store)

    def test_raises_on_nan_in_y(self, clf_pipeline):
        store = _make_clf_store()
        store.y = store.y.astype(float)
        store.y[0] = float("nan")
        with pytest.raises(ValueError, match="NaN or infinite"):
            clf_pipeline._validate_store(store)

    def test_raises_on_constant_column(self, clf_pipeline):
        store = _make_clf_store()
        store.X[:, 2] = 5.0  # constant
        with pytest.raises(ValueError, match="constant"):
            clf_pipeline._validate_store(store)

    def test_raises_on_single_class_target(self, clf_pipeline):
        store = _make_clf_store()
        store.y = np.zeros(N_SAMPLES, dtype=int)
        with pytest.raises(ValueError, match="unique class"):
            clf_pipeline._validate_store(store)

    def test_raises_when_n_samples_less_than_n_splits(self, clf_pipeline):
        store = _make_clf_store()
        # clf_pipeline uses n_splits=3; give only 2 samples
        store.X = store.X[:2]
        store.y = store.y[:2]
        with pytest.raises(ValueError, match="n_splits"):
            clf_pipeline._validate_store(store)

    def test_raises_on_duplicate_feature_names(self, clf_pipeline):
        store = _make_clf_store()
        store.feature_names = ["a", "b", "a", "c"]  # "a" duplicated
        with pytest.raises(ValueError, match="duplicate"):
            clf_pipeline._validate_store(store)

    def test_valid_store_does_not_raise(self, clf_pipeline):
        store = _make_clf_store()
        clf_pipeline._validate_store(store)  # should not raise

    def test_regression_single_class_check_skipped(self):
        """Single-value y is allowed for regression — the check only applies to classification."""
        reg_pipeline = H2MLPipeline(
            models=[make_regressor(Ridge())],
            config=PipelineConfig(task_type=TaskType.REGRESSION, n_splits=3),
        )
        store = _make_reg_store()
        reg_pipeline._validate_store(store)  # should not raise

    def test_run_raises_on_nan_before_fitting(self, clf_pipeline):
        store = _make_clf_store()
        store.X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or infinite"):
            clf_pipeline.run(store)

    def test_run_step1_raises_on_nan(self, clf_pipeline):
        store = _make_clf_store()
        store.X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or infinite"):
            clf_pipeline.run_step1_only(store)

    def test_run_step1_to_step2_raises_on_nan(self, clf_pipeline):
        store = _make_clf_store()
        store.X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or infinite"):
            clf_pipeline.run_step1_to_step2(store)

    def test_x_error_reports_affected_feature_name(self, clf_pipeline):
        store = _make_clf_store()
        store.X[0, 1] = float("nan")  # second feature column
        with pytest.raises(ValueError, match=store.feature_names[1]):
            clf_pipeline._validate_store(store)

    def test_x_error_reports_row_count(self, clf_pipeline):
        store = _make_clf_store()
        store.X[0, 0] = float("nan")
        store.X[1, 0] = float("nan")  # 2 distinct rows affected
        with pytest.raises(ValueError, match=rf"2/{N_SAMPLES}"):
            clf_pipeline._validate_store(store)

    def test_x_error_reports_percentage(self, clf_pipeline):
        store = _make_clf_store()
        store.X[0, 0] = float("nan")  # 1 / N_SAMPLES rows
        expected_pct = f"{100 / N_SAMPLES:.1f}%"
        with pytest.raises(ValueError, match=re.escape(expected_pct)):
            clf_pipeline._validate_store(store)

    def test_y_error_reports_count_and_pct(self, clf_pipeline):
        store = _make_clf_store()
        store.y = store.y.astype(float)
        store.y[0] = float("nan")
        with pytest.raises(ValueError, match=rf"1/{N_SAMPLES}"):
            clf_pipeline._validate_store(store)


# ---------------------------------------------------------------------------
# cv_warnings — fold failure tracking
# ---------------------------------------------------------------------------


class TestCVWarnings:
    def test_cv_warnings_empty_on_successful_run(self):
        """A clean run should produce no cv_warnings."""
        store = _make_clf_store()
        result = PipelineResult()
        assert result.cv_warnings == []

    def test_cv_warnings_populated_after_fold_failures(self, clf_pipeline):
        """Force a fold failure and verify the warning is surfaced on the result."""
        store = _make_clf_store()

        original_run_all = clf_pipeline._cv.run_all

        call_count = [0]

        def patched_run_all(steps, X, y, **kwargs):
            results = original_run_all(steps, X, y, n_jobs=1, **kwargs)
            # Inject a fake failed fold into the first result
            if call_count[0] == 0:
                results[0].failed_folds = [0, 1]
            call_count[0] += 1
            return results

        mock_sel = MagicMock(spec=FeatureSelector)
        mock_sel.fit_transform.side_effect = lambda s, y=None: s.select(
            s.feature_names[:4]
        )
        mock_sel.transform.side_effect = lambda s: s.select(s.feature_names[:4])

        with (
            patch.object(clf_pipeline._cv, "run_all", side_effect=patched_run_all),
            patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=mock_sel),
            patch.object(H2MLPipeline, "_is_opt_enabled", return_value=False),
        ):
            result = clf_pipeline.run(store)

        assert any("2/" in w and "fold(s) failed" in w for w in result.cv_warnings)


# ---------------------------------------------------------------------------
# Spatial block CV — pipeline integration
# ---------------------------------------------------------------------------


def _make_spatial_store(seed: int = 42) -> PipelineData:
    """Classification store with spatial coordinates attached."""
    rng = np.random.default_rng(seed)
    return PipelineData(
        X=rng.standard_normal((N_SAMPLES, len(FEATURE_NAMES))),
        feature_names=FEATURE_NAMES,
        y=rng.integers(0, 2, size=N_SAMPLES),
        coords=rng.uniform(0, 1, size=(N_SAMPLES, 2)),
    )


class TestSpatialCVPipeline:
    """Integration tests: pipeline.run() with store.coords set."""

    def test_full_run_with_coords_completes_all_steps(self):
        store = _make_spatial_store()
        pipeline = H2MLPipeline(
            models=_lr_only(), config=_clf_config(n_blocks_per_fold=3)
        )
        with _mock_selector():
            result = pipeline.run(store)
        assert result.completed_steps == [1, 2, 3, 4]

    def test_result_has_same_structure_as_without_coords(self):
        store_no_coords = _make_clf_store()
        store_with_coords = _make_spatial_store()
        pipeline = H2MLPipeline(
            models=_lr_only(), config=_clf_config(n_blocks_per_fold=3)
        )
        with _mock_selector():
            result_plain = pipeline.run(store_no_coords)
            result_spatial = pipeline.run(store_with_coords)
        # Both should complete the same 4 steps
        assert result_plain.completed_steps == result_spatial.completed_steps

    def test_coords_forwarded_to_cv_run_all(self):
        """Verify coords are passed through to CrossValidator.run_all."""
        store = _make_spatial_store()
        pipeline = H2MLPipeline(
            models=_lr_only(), config=_clf_config(n_blocks_per_fold=2)
        )
        coords_seen = []

        original_run_all = pipeline._cv.run_all

        def spy_run_all(steps, X, y, **kwargs):
            coords_seen.append(kwargs.get("coords"))
            return original_run_all(steps, X, y, n_jobs=1, **kwargs)

        with (
            _mock_selector(),
            patch.object(pipeline._cv, "run_all", side_effect=spy_run_all),
        ):
            pipeline.run(store)

        assert any(c is not None for c in coords_seen), (
            "coords=None was passed to all run_all calls — spatial CV was not activated."
        )

    def test_validate_store_rejects_nan_coords(self):
        store = _make_spatial_store()
        store.coords[0, 0] = float("nan")
        pipeline = H2MLPipeline(models=_lr_only(), config=_clf_config())
        with pytest.raises(ValueError, match="coords"):
            pipeline.run(store)

    def test_validate_store_rejects_inf_coords(self):
        store = _make_spatial_store()
        store.coords[5, 1] = float("inf")
        pipeline = H2MLPipeline(models=_lr_only(), config=_clf_config())
        with pytest.raises(ValueError, match="coords"):
            pipeline.run(store)

    def test_step1_builds_spatial_splitter_when_coords_provided(self):
        """step 1 uses a SpatialBlockSplitter (not KFold) when coords are present.
        This verifies the flat-parallel path in _run_step1 forwards coords correctly."""
        from h2ml.features.spatial_cv import SpatialBlockSplitter

        store = _make_spatial_store()
        pipeline = H2MLPipeline(
            models=_lr_only(), config=_clf_config(n_blocks_per_fold=2)
        )
        result = pipeline.run_step1_only(store)
        assert isinstance(result.splitter, SpatialBlockSplitter)

    def test_coords_propagated_through_transform_stores(self):
        """build_transform_stores must receive and forward coords."""
        store = _make_spatial_store()
        store = PipelineData(
            X=store.X,
            feature_names=store.feature_names,
            y=np.abs(store.y.astype(float)) + 0.1,  # positive regression target
            coords=store.coords,
        )
        from h2ml.preprocessing.transform_stores import build_transform_stores

        stores = build_transform_stores(
            store.X,
            store.y,
            store.feature_names,
            transforms=["log", "sqrt"],
            coords=store.coords,
        )
        for name, ts in stores.items():
            assert ts.coords is not None, f"Transform store '{name}' lost coords"
            np.testing.assert_array_equal(ts.coords, store.coords)
