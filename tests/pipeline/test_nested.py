"""
tests/pipeline/test_nested.py

Tests for nested_cv / NestedCVResult. Step 2 (SHAP) is mocked as in
test_pipeline.py; LR and Ridge have opt_enabled=False, so step 4 skips Optuna.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from h2ml import NestedCVResult, nested_cv
from h2ml.core.feature_store import PipelineData
from h2ml.features.selector import FeatureSelector
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig
from h2ml.pipeline.step import make_classifier, make_regressor

FEATURE_NAMES = [f"feat_{i}" for i in range(6)]
N = 150


def _clf_store(seed: int = 0) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, len(FEATURE_NAMES)))
    y = (X[:, 0] + 0.5 * rng.standard_normal(N) > 0).astype(int)
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _reg_store(seed: int = 0, coords: bool = False) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, len(FEATURE_NAMES)))
    y = np.exp(0.5 * X[:, 0] + 0.1 * rng.standard_normal(N))
    xy = rng.uniform(0, 100, size=(N, 2)) if coords else None
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y, coords=xy)


def _mock_selector(selected: list[str]):
    mock_sel = MagicMock(spec=FeatureSelector)
    mock_sel.selected_features_ = selected
    mock_sel.fit_transform.side_effect = lambda store, y=None: store.select(selected)
    mock_sel.transform.side_effect = lambda store: store.select(selected)
    return patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=mock_sel)


def _lr_pipeline(**kwargs) -> H2MLPipeline:
    config = PipelineConfig(**{"task_type": "classification", "metric": "AUC", "n_splits": 3, "n_trials": 1, **kwargs})
    return H2MLPipeline(config=config, models=[make_classifier(LogisticRegression(max_iter=500), name="LR")])


def _ridge_pipeline(**kwargs) -> H2MLPipeline:
    config = PipelineConfig(**{"task_type": "regression", "metric": "R2", "n_splits": 3, "n_trials": 1, **kwargs})
    return H2MLPipeline(config=config, models=[make_regressor(Ridge(), name="Ridge")])


@pytest.fixture(scope="module")
def clf_nested() -> tuple[NestedCVResult, PipelineData]:
    store = _clf_store()
    with _mock_selector(FEATURE_NAMES[:3]):
        return nested_cv(_lr_pipeline(), store, outer_splits=3), store


class TestNestedCV:
    def test_one_row_per_outer_fold(self, clf_nested):
        nested, _ = clf_nested
        assert nested.n_folds == 3
        assert list(nested.folds_df["Outer_Fold"]) == [0, 1, 2]
        assert nested.failed_folds == []
        assert len(nested.results) == len(nested.selected_features) == 3
        assert nested.cv_type == "random"

    def test_outer_folds_are_stratified_kfold_and_disjoint_from_inner_rows(self, clf_nested):
        nested, store = clf_nested
        expected = list(StratifiedKFold(3, shuffle=True, random_state=42).split(store.X, store.y))
        for result, (train_idx, test_idx) in zip(nested.results, expected, strict=True):
            np.testing.assert_array_equal(result.holdout_idx, test_idx)
            np.testing.assert_array_equal(result.features.X, store.X[train_idx])  # inner run saw train rows only

    def test_outer_score_matches_manual_fit(self, clf_nested):
        nested, _ = clf_nested
        for (_, row), result, names in zip(
            nested.folds_df.iterrows(), nested.results, nested.selected_features, strict=True
        ):
            dev, test = result.features.select(names), result.holdout_store.select(names)
            model = LogisticRegression(max_iter=500).fit(dev.X, dev.y)
            assert row["AUC_Test"] == pytest.approx(roc_auc_score(test.y, model.predict_proba(test.X)[:, 1]))
            assert row["Inner_AUC"] == pytest.approx(result.best_model_value)

    def test_summary_optimism_sign(self, clf_nested):
        nested, _ = clf_nested
        s = nested.summary().iloc[0]
        assert s["Outer_AUC_Mean"] == pytest.approx(nested.folds_df["AUC_Test"].mean())
        assert s["Optimism_Mean"] == pytest.approx((nested.folds_df["Inner_AUC"] - nested.folds_df["AUC_Test"]).mean())
        assert s["N_folds"] == 3

    def test_selection_counts_and_feature_frequency(self, clf_nested):
        nested, _ = clf_nested
        counts = nested.selection_counts()
        assert counts["Count"].sum() == 3
        assert counts["Fraction"].sum() == pytest.approx(1.0)
        freq = nested.feature_frequency()
        assert freq.max() <= 1.0
        assert set(freq.index) <= set(FEATURE_NAMES)

    def test_regression_with_transform_minimised_metric(self):
        store = _reg_store()
        with _mock_selector(FEATURE_NAMES[:3]):
            nested = nested_cv(_ridge_pipeline(metric="MAE"), store, outer_splits=3, transforms=["log"])
        assert set(nested.folds_df["Y_transform"]) == {"log"}
        df = nested.folds_df
        s = nested.summary().iloc[0]
        # error metric: optimism is outer − inner
        assert s["Optimism_Mean"] == pytest.approx((df["MAE_Test"] - df["Inner_MAE"]).mean())
        result, names = nested.results[0], nested.selected_features[0]
        dev, test = result.features.select(names), result.holdout_store.select(names)
        pred = np.expm1(Ridge().fit(dev.X, np.log1p(dev.y)).predict(test.X))
        assert df.iloc[0]["R2_Test"] == pytest.approx(r2_score(test.y, pred))

    def test_spatial_outer_folds(self):
        store = _reg_store(coords=True)
        with _mock_selector(FEATURE_NAMES[:3]):
            nested = nested_cv(_ridge_pipeline(spatial_cv_method="block"), store, outer_splits=3)
        assert nested.cv_type == "spatial"
        assert nested.n_folds == 3
        test_rows = np.concatenate([r.holdout_idx for r in nested.results])
        assert np.unique(test_rows).size == test_rows.size == N  # outer test folds partition the rows

    def test_holdout_frac_ignored_inside_loop(self):
        with _mock_selector(FEATURE_NAMES[:3]):
            nested = nested_cv(_lr_pipeline(holdout_frac=0.2), _clf_store(), outer_splits=3)
        for result in nested.results:
            assert result.features.n_samples + result.holdout_store.n_samples == N

    def test_failed_fold_is_skipped(self):
        pipeline = _lr_pipeline(n_jobs=1)  # a patched pipeline cannot be pickled for parallel workers
        real_run = pipeline.run
        calls = {"n": 0}

        def flaky_run(store, transforms=None):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("boom")
            return real_run(store, transforms=transforms)

        with _mock_selector(FEATURE_NAMES[:3]), patch.object(pipeline, "run", side_effect=flaky_run):
            nested = nested_cv(pipeline, _clf_store(), outer_splits=3)
        assert nested.failed_folds == [1]
        assert nested.n_folds == 2

    def test_all_folds_failing_raises(self):
        pipeline = _lr_pipeline()
        with patch.object(pipeline, "run", side_effect=ValueError("boom")), pytest.raises(RuntimeError, match="all 3"):
            nested_cv(pipeline, _clf_store(), outer_splits=3)

    def test_outer_splits_below_two_raises(self):
        with pytest.raises(ValueError, match="outer_splits"):
            nested_cv(_lr_pipeline(), _clf_store(), outer_splits=1)
