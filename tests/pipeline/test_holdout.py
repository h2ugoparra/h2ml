"""
tests/pipeline/test_holdout.py

Tests for the outer holdout: split_holdout / stage_store helpers, the pipeline
wiring (rows set aside before step 1, scored once at the end), conformal
calibration from holdout residuals, holdout_refit, and persistence.

Step 2 (SHAP) is mocked as in test_pipeline.py; LR has opt_enabled=False, so
step 4 skips Optuna without a mock.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from loguru import logger
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score

from h2ml.core.base import TaskType
from h2ml.core.feature_store import PipelineData
from h2ml.core.spatial_config import SpatialCVConfig
from h2ml.features.selector import FeatureSelector
from h2ml.features.spatial_cv import build_splitter
from h2ml.persistence.result_io import load_result, save_result
from h2ml.pipeline.holdout import concat_stores, split_holdout, stage_store
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig
from h2ml.pipeline.step import make_classifier, make_regressor

FEATURE_NAMES = [f"feat_{i}" for i in range(6)]
N = 120


def _clf_store(seed: int = 0, n: int = N) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(int)
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _reg_store(seed: int = 0, n: int = N) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = np.exp(0.5 * X[:, 0] + 0.1 * rng.standard_normal(n))  # strictly positive — log transform
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _mock_selector(selected: list[str]):
    mock_sel = MagicMock(spec=FeatureSelector)
    mock_sel.selected_features_ = selected
    mock_sel.fit_transform.side_effect = lambda store, y=None: store.select(selected)
    mock_sel.transform.side_effect = lambda store: store.select(selected)
    return patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=mock_sel)


def _lr_pipeline(**kwargs) -> H2MLPipeline:
    config = PipelineConfig(task_type="classification", metric="AUC", n_splits=3, n_trials=1, **kwargs)
    return H2MLPipeline(config=config, models=[make_classifier(LogisticRegression(max_iter=500), name="LR")])


def _ridge_pipeline(**kwargs) -> H2MLPipeline:
    config = PipelineConfig(task_type="regression", metric="R2", n_splits=3, n_trials=1, **kwargs)
    return H2MLPipeline(config=config, models=[make_regressor(Ridge(), name="Ridge")])


# ---------------------------------------------------------------------------
# PipelineData.take / concat_stores / stage_store
# ---------------------------------------------------------------------------


class TestStoreHelpers:
    def test_take_subsets_every_row_array(self):
        store = _clf_store()
        store = PipelineData(
            X=store.X,
            feature_names=FEATURE_NAMES,
            y=store.y,
            y_true=store.y * 2,
            coords=np.arange(2 * N, dtype=float).reshape(N, 2),
            times=np.arange(N).astype("datetime64[D]"),
        )
        idx = np.array([3, 0, 7])
        sub = store.take(idx)
        np.testing.assert_array_equal(sub.X, store.X[idx])
        np.testing.assert_array_equal(sub.y, store.y[idx])
        np.testing.assert_array_equal(sub.y_true, store.y_true[idx])
        np.testing.assert_array_equal(sub.coords, store.coords[idx])
        np.testing.assert_array_equal(sub.times, store.times[idx])

    def test_concat_of_takes_restores_rows(self):
        store = _clf_store()
        a, b = np.arange(0, 80), np.arange(80, N)
        joined = concat_stores(store.take(a), store.take(b))
        np.testing.assert_array_equal(joined.X, store.X)
        np.testing.assert_array_equal(joined.y, store.y)

    def test_stage_store_applies_transform_and_keeps_raw_y_true(self):
        store = _reg_store()
        staged = stage_store(store, "log", ["feat_1", "feat_0"])
        np.testing.assert_allclose(staged.y, np.log1p(store.y))
        np.testing.assert_array_equal(staged.y_true, store.y)
        assert staged.y_transform == "log"
        assert staged.feature_names == ["feat_1", "feat_0"]

    def test_stage_store_winsorize_keeps_pre_clip_y(self):
        store = _reg_store()
        store.y[:5] = 1e4  # outliers
        staged = stage_store(store, "winlog")
        np.testing.assert_allclose(staged.y, np.log1p(store.y))  # not clipped — fitted per training set

    def test_stage_store_leaves_already_transformed_store(self):
        store = stage_store(_reg_store(), "log")
        assert stage_store(store, "log").y is store.y


# ---------------------------------------------------------------------------
# split_holdout
# ---------------------------------------------------------------------------


class TestSplitHoldout:
    def _check_partition(self, dev, test, n):
        assert np.intersect1d(dev, test).size == 0
        np.testing.assert_array_equal(np.sort(np.concatenate([dev, test])), np.arange(n))

    def test_random_classification_is_stratified_partition(self):
        store = _clf_store()
        dev, test, strategy = split_holdout(store, TaskType.CLASSIFICATION, 0.25, "auto", random_state=0)
        assert strategy == "random"
        self._check_partition(dev, test, N)
        assert test.size == 30
        assert abs(store.y[test].mean() - store.y.mean()) < 0.05

    def test_time_holds_out_latest_rows(self):
        store = _reg_store()
        rng = np.random.default_rng(1)
        times = (np.datetime64("2020-01-01") + rng.permutation(N)).astype("datetime64[D]")
        store = PipelineData(X=store.X, feature_names=FEATURE_NAMES, y=store.y, times=times)
        dev, test, strategy = split_holdout(store, TaskType.REGRESSION, 0.2, "time", random_state=0)
        assert strategy == "time"
        self._check_partition(dev, test, N)
        assert test.size == 24
        assert times[test].min() > times[dev].max()

    def test_time_without_times_raises(self):
        with pytest.raises(ValueError, match="requires store.times"):
            split_holdout(_reg_store(), TaskType.REGRESSION, 0.2, "time", random_state=0)

    def test_auto_with_coords_holds_out_a_spatial_fold(self):
        base = _reg_store(n=200)
        rng = np.random.default_rng(2)
        coords = rng.uniform(0, 100, size=(200, 2))
        store = PipelineData(X=base.X, feature_names=FEATURE_NAMES, y=base.y, coords=coords)
        spatial = SpatialCVConfig(spatial_cv_method="block")
        dev, test, strategy = split_holdout(store, TaskType.REGRESSION, 0.25, "auto", random_state=0, spatial=spatial)
        assert strategy == "spatial"
        self._check_partition(dev, test, 200)
        splitter = build_splitter(
            TaskType.REGRESSION, n_splits=4, random_state=0, coords=coords, X=store.X, y=store.y, spatial=spatial
        )
        folds = [np.sort(te) for _, te in splitter.split(store.X, store.y)]
        assert any(np.array_equal(test, f) for f in folds)
        # whole blocks: no spatial block is split between dev and holdout
        assert np.intersect1d(splitter.block_id_[test], splitter.block_id_[dev]).size == 0

    def test_random_strategy_ignores_coords(self):
        base = _reg_store()
        store = PipelineData(
            X=base.X, feature_names=FEATURE_NAMES, y=base.y, coords=np.random.default_rng(0).uniform(size=(N, 2))
        )
        _, _, strategy = split_holdout(store, TaskType.REGRESSION, 0.2, "random", random_state=0)
        assert strategy == "random"

    def test_single_class_holdout_raises(self):
        store = _clf_store()
        y = np.zeros(N, dtype=int)
        y[:20] = 1  # the latest rows (by time) are all class 0
        times = np.arange(N).astype("datetime64[D]")
        store = PipelineData(X=store.X, feature_names=FEATURE_NAMES, y=y, times=times)
        with pytest.raises(ValueError, match="single class"):
            split_holdout(store, TaskType.CLASSIFICATION, 0.2, "time", random_state=0)


# ---------------------------------------------------------------------------
# PipelineConfig validation
# ---------------------------------------------------------------------------


class TestHoldoutConfig:
    def test_defaults_disable_holdout(self):
        config = PipelineConfig()
        assert config.holdout_frac == 0.0
        assert config.holdout_strategy == "auto"
        assert config.holdout_refit == "dev"

    @pytest.mark.parametrize("frac", [-0.1, 0.6])
    def test_invalid_frac_raises(self, frac):
        with pytest.raises(ValueError, match="holdout_frac"):
            PipelineConfig(holdout_frac=frac)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="holdout_strategy"):
            PipelineConfig(holdout_strategy="spatial")

    def test_invalid_refit_raises(self):
        with pytest.raises(ValueError, match="holdout_refit"):
            PipelineConfig(holdout_refit="train")


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


class TestPipelineHoldout:
    def test_no_holdout_by_default(self):
        store = _clf_store()
        with _mock_selector(FEATURE_NAMES[:3]):
            result = _lr_pipeline().run(store)
        assert result.features is store
        assert result.holdout_store is None and result.holdout_df is None
        assert "holdout" not in set(result.summary()["Stage"])

    def test_steps_run_on_dev_rows_only(self):
        store = _clf_store()
        with _mock_selector(FEATURE_NAMES[:3]):
            result = _lr_pipeline(holdout_frac=0.25).run(store)
        assert result.features.n_samples == 90
        assert result.holdout_store.n_samples == 30
        assert result.holdout_strategy == "random"
        np.testing.assert_array_equal(result.holdout_store.X, store.X[result.holdout_idx])
        cv_rows = np.concatenate([f.test_idx for f in result.step1_cv_result[0].folds])
        assert cv_rows.max() < 90  # CV never indexes beyond the dev rows

    def test_holdout_score_matches_manual_fit(self):
        store = _clf_store()
        with _mock_selector(FEATURE_NAMES[:3]):
            result = _lr_pipeline(holdout_frac=0.25).run(store)
        names = result.features_reduced.feature_names if result.best_stage == "reduced" else FEATURE_NAMES
        dev, hold = result.features.select(names), result.holdout_store.select(names)
        model = LogisticRegression(max_iter=500).fit(dev.X, dev.y)
        expected = roc_auc_score(hold.y, model.predict_proba(hold.X)[:, 1])
        assert result.holdout_df.iloc[0]["AUC_Test_Mean"] == pytest.approx(expected)
        assert "holdout" in set(result.summary()["Stage"])

    def test_holdout_score_uses_winning_transform(self):
        store = _reg_store()
        with _mock_selector(FEATURE_NAMES[:3]):
            result = _ridge_pipeline(holdout_frac=0.25).run(store, transforms=["log"])
        assert result.y_transform == "log"
        names = result.features_reduced.feature_names if result.best_stage == "reduced" else FEATURE_NAMES
        dev, hold = result.features.select(names), result.holdout_store.select(names)
        model = Ridge().fit(dev.X, np.log1p(dev.y))
        expected = r2_score(hold.y, np.expm1(model.predict(hold.X)))  # scored on the original scale
        assert result.holdout_df.iloc[0]["R2_Test_Mean"] == pytest.approx(expected)

    def test_partial_run_keeps_holdout_and_resume_scores_it(self):
        store = _clf_store()
        pipeline = _lr_pipeline(holdout_frac=0.25)
        with _mock_selector(FEATURE_NAMES[:3]):
            step1 = pipeline.run_step1_only(store)
            assert step1.holdout_store is not None and step1.holdout_df is None
            result = pipeline.run_step1_to_step3(store)
        assert result.holdout_df is not None
        result.holdout_df = None
        result = pipeline.run_step4_only(result)
        assert result.holdout_df is not None

    def test_large_gap_adds_warning(self):
        store = _clf_store()
        with _mock_selector(FEATURE_NAMES[:3]):
            pipeline = _lr_pipeline(holdout_frac=0.25)
            result = pipeline.run(store)
        gap_warning = "worse than the CV estimate"
        assert not any(w.startswith("[holdout]") and gap_warning in w for w in result.cv_warnings)
        result.best_model_value, result.best_model_std = 1.5, 0.01  # CV claims far more than the holdout gives
        result = pipeline._score_holdout(result)
        assert any(w.startswith("[holdout]") and gap_warning in w for w in result.cv_warnings)

    @pytest.mark.parametrize("positive_rate, warned", [(0.08, True), (0.5, False)])
    def test_rare_class_in_holdout_warns(self, positive_rate, warned):
        rng = np.random.default_rng(0)
        n = 400
        X = rng.standard_normal((n, len(FEATURE_NAMES)))
        y = (X[:, 0] + rng.standard_normal(n) > np.quantile(X[:, 0], 1 - positive_rate)).astype(int)
        store = PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)  # 20% holdout: ~6 vs ~40 positives
        with _mock_selector(FEATURE_NAMES[:3]):
            result = _lr_pipeline(holdout_frac=0.2).run(store)
        rare = [w for w in result.cv_warnings if w.startswith("[holdout]") and "rarest class" in w]
        assert bool(rare) is warned

    def test_rare_class_warning_skipped_for_regression(self):
        with _mock_selector(FEATURE_NAMES[:3]):
            result = _ridge_pipeline(holdout_frac=0.2).run(_reg_store())
        assert not any("rarest class" in w for w in result.cv_warnings)


# ---------------------------------------------------------------------------
# Final model: conformal from holdout residuals, holdout_refit
# ---------------------------------------------------------------------------


class TestFinalModelHoldout:
    def test_conformal_calibrated_on_holdout_residuals(self):
        with _mock_selector(FEATURE_NAMES[:3]):
            result = _ridge_pipeline(holdout_frac=0.25).run(_reg_store())
        model = result.build_final_model()
        assert model.conformal.n == result.holdout_store.n_samples

    @pytest.mark.parametrize(("refit", "n_rows"), [("dev", 90), ("all", 120)])
    def test_refit_rows(self, refit, n_rows):
        store = _reg_store()
        with _mock_selector(FEATURE_NAMES[:3]):
            result = _ridge_pipeline(holdout_frac=0.25, holdout_refit=refit).run(store)
        model = result.build_final_model()
        names = model.feature_names
        rows = np.arange(N) if refit == "all" else np.setdiff1d(np.arange(N), result.holdout_idx)
        expected = Ridge().fit(store.select(names).X[rows], store.y[rows])
        assert len(rows) == n_rows
        np.testing.assert_allclose(model.estimator.coef_, expected.coef_)

    def test_interval_coverage_near_nominal(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((1600, 3))
        y = X @ np.array([1.0, -2.0, 0.5]) + rng.standard_normal(1600)
        train = PipelineData(X=X[:600], feature_names=["a", "b", "c"], y=y[:600])
        with _mock_selector(["a", "b", "c"]):
            result = _ridge_pipeline(holdout_frac=0.3).run(train)
        lo, hi = result.build_final_model().predict_interval(X[600:], alpha=0.1)
        coverage = np.mean((y[600:] >= lo) & (y[600:] <= hi))
        assert 0.85 <= coverage <= 0.95


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestHoldoutPersistence:
    def test_roundtrip_without_unpersisted_fields_warning(self, tmp_path):
        with _mock_selector(FEATURE_NAMES[:3]):
            result = _lr_pipeline(holdout_frac=0.25, holdout_refit="all").run(_clf_store())
        result.selector = None  # the mocked selector cannot be pickled
        messages: list[str] = []
        sink = logger.add(messages.append, level="WARNING")
        try:
            save_result(result, tmp_path / "run")
        finally:
            logger.remove(sink)
        assert not any("not persisted" in m for m in messages)

        loaded = load_result(tmp_path / "run")
        np.testing.assert_array_equal(loaded.holdout_idx, result.holdout_idx)
        np.testing.assert_array_equal(loaded.holdout_store.X, result.holdout_store.X)
        assert loaded.holdout_strategy == "random"
        assert loaded.holdout_refit == "all"
        assert loaded.holdout_df.iloc[0]["AUC_Test_Mean"] == pytest.approx(result.holdout_df.iloc[0]["AUC_Test_Mean"])
        assert loaded.holdout_cv_result is not None
        assert "holdout" in set(loaded.summary()["Stage"])
