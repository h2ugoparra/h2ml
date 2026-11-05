"""
tests/pipeline/test_fold_selection.py

Tests for select_features_in_folds: CrossValidator fold_columns, BlockGroupSplitter,
and the pipeline wiring (per-fold selection on training rows only in step 3, the
same per-fold sets in step 4's acceptance CV, persistence).

A scripted fake selector stands in for SHAP so the per-fold feature sets are known;
one integration test runs the real selector.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from h2ml.core.feature_store import PipelineData
from h2ml.features.spatial_cv import BlockGroupSplitter, SpatialBlockSplitter
from h2ml.persistence.result_io import load_result, save_result
from h2ml.pipeline.cv import CrossValidator
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig
from h2ml.pipeline.step import make_classifier, make_regressor

FEATURE_NAMES = [f"feat_{i}" for i in range(6)]
N = 120
STEP2_SET = ["feat_0", "feat_1", "feat_2"]
FOLD_SETS = [["feat_0", "feat_3"], ["feat_0", "feat_1", "feat_4"], ["feat_5", "feat_0"]]


def _clf_store(seed: int = 0) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, len(FEATURE_NAMES)))
    y = (X[:, 0] + 0.5 * rng.standard_normal(N) > 0).astype(int)
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _reg_store(seed: int = 0, coords: bool = False) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, len(FEATURE_NAMES)))
    y = X[:, 0] + 0.3 * rng.standard_normal(N)
    xy = rng.uniform(0, 100, size=(N, 2)) if coords else None
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y, coords=xy)


class _ScriptedSelectors:
    """Patches FeatureSelector: step 2 (fit_transform) gets STEP2_SET, each per-fold fit the next FOLD_SETS entry."""

    def __init__(self):
        self.fold_fits: list[tuple[np.ndarray, object]] = []  # (training X, inner splitter) per fold
        self._next = iter(FOLD_SETS)

    def _make(self, **kwargs):
        sel = MagicMock()

        def fit(store, y=None):
            names = next(self._next)
            self.fold_fits.append((store.X.copy(), kwargs.get("_splitter")))
            sel.selected_features_ = names
            sel.n_selected = len(names)
            return sel

        def fit_transform(store, y=None):
            sel.selected_features_ = STEP2_SET
            return store.select(STEP2_SET)

        sel.fit.side_effect = fit
        sel.fit_transform.side_effect = fit_transform
        sel.transform.side_effect = lambda store: store.select(sel.selected_features_)
        return sel

    def patch(self):
        return patch("h2ml.pipeline.pipeline.FeatureSelector", side_effect=self._make)


def _lr_pipeline(**kwargs) -> H2MLPipeline:
    config = PipelineConfig(**{"task_type": "classification", "metric": "AUC", "n_splits": 3, "n_trials": 1, **kwargs})
    return H2MLPipeline(config=config, models=[make_classifier(LogisticRegression(max_iter=500), name="LR")])


# ---------------------------------------------------------------------------
# CrossValidator fold_columns
# ---------------------------------------------------------------------------


class TestFoldColumns:
    @pytest.mark.parametrize("requires_scaling", [False, True])
    def test_each_fold_fits_on_its_own_columns(self, requires_scaling):
        store = _clf_store()
        cols = [np.array([0, 3]), np.array([0, 1, 4]), np.array([5, 0])]
        cv = CrossValidator(n_splits=3, random_state=0)
        step = make_classifier(LogisticRegression(max_iter=500), name="LR", requires_scaling=requires_scaling)
        result = cv.run(step, store.X, store.y, fold_columns=cols)
        folds = StratifiedKFold(3, shuffle=True, random_state=0).split(store.X, store.y)
        for fold, (train_idx, test_idx), c in zip(result.folds, folds, cols, strict=True):
            X_train, X_test = store.X[train_idx][:, c], store.X[test_idx][:, c]
            if requires_scaling:
                mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
                X_train, X_test = (X_train - mu) / sd, (X_test - mu) / sd
            model = LogisticRegression(max_iter=500).fit(X_train, store.y[train_idx])
            np.testing.assert_allclose(fold.y_prob_test, model.predict_proba(X_test)[:, 1])

    def test_run_all_forwards_fold_columns_with_precomputed_scaling(self):
        store = _clf_store()
        cols = [np.array([0]), np.array([1]), np.array([2])]
        cv = CrossValidator(n_splits=3, random_state=0)
        steps = [make_classifier(LogisticRegression(max_iter=500), name="LR", requires_scaling=True)]
        splitter = StratifiedKFold(3, shuffle=True, random_state=0)
        via_all = cv.run_all(steps, store.X, store.y, n_jobs=1, fold_columns=cols, _splitter=splitter)[0]
        direct = cv.run(steps[0], store.X, store.y, fold_columns=cols, _splitter=splitter)
        for a, b in zip(via_all.folds, direct.folds, strict=True):
            np.testing.assert_allclose(a.y_prob_test, b.y_prob_test)

    def test_length_mismatch_raises(self):
        store = _clf_store()
        cv = CrossValidator(n_splits=3, random_state=0)
        step = make_classifier(LogisticRegression(max_iter=500), name="LR")
        with pytest.raises(ValueError, match="fold_columns has 2 entries"):
            cv.run(step, store.X, store.y, fold_columns=[np.array([0]), np.array([1])])


# ---------------------------------------------------------------------------
# BlockGroupSplitter
# ---------------------------------------------------------------------------


class TestBlockGroupSplitter:
    def test_partitions_rows_along_whole_blocks(self):
        block_ids = np.repeat(np.arange(8), 5)
        splitter = BlockGroupSplitter(block_ids, n_splits=4, random_state=0)
        tests = [te for _, te in splitter.split(np.zeros((40, 1)))]
        assert len(tests) == 4
        np.testing.assert_array_equal(np.sort(np.concatenate(tests)), np.arange(40))
        for train, test in splitter.split(np.zeros((40, 1))):
            assert np.intersect1d(block_ids[train], block_ids[test]).size == 0

    def test_deterministic_for_a_seed(self):
        block_ids = np.repeat(np.arange(6), 3)
        a = [te.tolist() for _, te in BlockGroupSplitter(block_ids, 3, random_state=1).split(np.zeros((18, 1)))]
        b = [te.tolist() for _, te in BlockGroupSplitter(block_ids, 3, random_state=1).split(np.zeros((18, 1)))]
        assert a == b

    @pytest.mark.parametrize("n_splits", [1, 4])
    def test_invalid_n_splits_raises(self, n_splits):
        with pytest.raises(ValueError, match="n_splits"):
            BlockGroupSplitter(np.array([0, 0, 1, 1, 2, 2]), n_splits=n_splits)


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


class TestPipelineFoldSelection:
    def test_off_by_default(self):
        assert PipelineConfig().select_features_in_folds is False
        scripted = _ScriptedSelectors()
        with scripted.patch():
            result = _lr_pipeline().run_step1_to_step3(_clf_store())
        assert result.step3_fold_features is None
        assert scripted.fold_fits == []

    def test_selection_runs_on_each_training_fold_only(self):
        store = _clf_store()
        scripted = _ScriptedSelectors()
        with scripted.patch():
            result = _lr_pipeline(select_features_in_folds=True).run_step1_to_step3(store)
        assert result.step3_fold_features == FOLD_SETS
        assert result.features_reduced.feature_names == STEP2_SET  # deployed set is still step 2's
        folds = StratifiedKFold(3, shuffle=True, random_state=42).split(store.X, store.y)
        for (fit_X, inner), (train_idx, _) in zip(scripted.fold_fits, folds, strict=True):
            np.testing.assert_array_equal(fit_X, store.X[train_idx])
            assert inner is None  # non-spatial: selector's own KFold

    def test_step3_reduced_scores_use_fold_sets(self):
        store = _clf_store()
        scripted = _ScriptedSelectors()
        with scripted.patch():
            result = _lr_pipeline(select_features_in_folds=True).run_step1_to_step3(store)
        cv = result.step3_cv_result[0]
        folds = StratifiedKFold(3, shuffle=True, random_state=42).split(store.X, store.y)
        for fold, (train_idx, test_idx), names in zip(cv.folds, folds, FOLD_SETS, strict=True):
            c = [FEATURE_NAMES.index(f) for f in names]
            model = LogisticRegression(max_iter=500).fit(store.X[train_idx][:, c], store.y[train_idx])
            np.testing.assert_allclose(fold.y_prob_test, model.predict_proba(store.X[test_idx][:, c])[:, 1])

    def test_step4_acceptance_cv_uses_fold_sets_when_reduced_won(self):
        store = _reg_store()
        rf = make_regressor(RandomForestRegressor(n_estimators=10, random_state=0), name="RF")
        config = PipelineConfig(
            task_type="regression", metric="R2", n_splits=3, n_trials=1, select_features_in_folds=True
        )
        pipeline = H2MLPipeline(config=config, models=[rf])
        scripted = _ScriptedSelectors()
        with scripted.patch():
            result = pipeline.run_step1_to_step3(store)
        result.best_stage = result.best_feature_stage = "reduced"  # force the reduced branch
        study = MagicMock(best_params={"max_depth": 3})
        with patch("h2ml.pipeline.pipeline.run_study", return_value=study):
            result = pipeline.run_step4_only(result)
        folds = KFold(3, shuffle=True, random_state=42).split(store.X, store.y)
        for fold, (train_idx, test_idx), names in zip(result.step4_cv_result.folds, folds, FOLD_SETS, strict=True):
            c = [FEATURE_NAMES.index(f) for f in names]
            model = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
            model.fit(store.X[train_idx][:, c], store.y[train_idx])
            np.testing.assert_allclose(fold.y_pred_test, model.predict(store.X[test_idx][:, c]))

    def test_spatial_inner_folds_follow_outer_blocks(self):
        store = _reg_store(coords=True)
        config = PipelineConfig(
            task_type="regression",
            metric="R2",
            n_splits=3,
            n_trials=1,
            spatial_cv_method="block",
            n_blocks_per_fold=3,
            select_features_in_folds=True,
        )
        pipeline = H2MLPipeline(config=config, models=[make_regressor(Ridge(), name="Ridge")])
        scripted = _ScriptedSelectors()
        with scripted.patch():
            result = pipeline.run_step1_to_step3(store)
        assert isinstance(result.splitter, SpatialBlockSplitter)
        for (fit_X, inner), (train_idx, _) in zip(
            scripted.fold_fits, result.splitter.split(store.X, store.y), strict=True
        ):
            assert isinstance(inner, BlockGroupSplitter)
            np.testing.assert_array_equal(inner.block_ids, result.splitter.block_id_[train_idx])
            np.testing.assert_array_equal(fit_X, store.X[train_idx])

    def test_fold_features_persisted(self, tmp_path):
        scripted = _ScriptedSelectors()
        with scripted.patch():
            result = _lr_pipeline(select_features_in_folds=True).run_step1_to_step3(_clf_store())
        result.selector = None  # the fake selector cannot be pickled
        save_result(result, tmp_path / "run")
        assert load_result(tmp_path / "run").step3_fold_features == FOLD_SETS

    def test_real_selector_end_to_end(self):
        store = _reg_store(seed=3)
        rf = make_regressor(RandomForestRegressor(n_estimators=10, random_state=0), name="RF")
        config = PipelineConfig(
            task_type="regression", metric="R2", n_splits=3, n_trials=1, select_features_in_folds=True
        )
        result = H2MLPipeline(config=config, models=[rf]).run_step1_to_step3(store)
        assert len(result.step3_fold_features) == 3
        assert all(set(names) <= set(FEATURE_NAMES) and names for names in result.step3_fold_features)
        assert result.step3_agg_df is not None and result.best_stage in ("default", "reduced")
