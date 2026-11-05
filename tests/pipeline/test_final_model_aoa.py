"""
tests/pipeline/test_final_model_aoa.py

Area of applicability wired through build_final_model, FinalModel persistence,
and geo_predict output columns. Step 2 is mocked with known SHAP importances.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
from loguru import logger
from sklearn.linear_model import LogisticRegression, Ridge

from h2ml.core.feature_store import PipelineData
from h2ml.features.selector import FeatureSelector
from h2ml.geo.geo_predict import _predict_delta_single, _predict_single
from h2ml.pipeline.final_model import DeltaFinalModel, FinalModel
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig
from h2ml.pipeline.step import make_classifier, make_regressor

FEATURE_NAMES = ["f0", "f1", "f2", "f3"]
SELECTED = ["f0", "f1", "f2"]
IMPORTANCE = pd.Series([3.0, 1.0, 0.0, 0.5], index=["f0", "f1", "f2", "f3"])
N = 120


def _store(task: str, seed: int = 0) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, len(FEATURE_NAMES)))
    if task == "classification":
        y = (X[:, 0] + 0.5 * rng.standard_normal(N) > 0).astype(int)
    else:
        y = X[:, 0] + 0.3 * rng.standard_normal(N)
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _mock_selector():
    sel = MagicMock(spec=FeatureSelector)
    sel.selected_features_ = SELECTED
    sel.feature_importance_ = IMPORTANCE
    sel.fit_transform.side_effect = lambda store, y=None: store.select(SELECTED)
    sel.transform.side_effect = lambda store: store.select(SELECTED)
    return patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=sel)


def _run(task: str, **kwargs):
    if task == "classification":
        model = make_classifier(LogisticRegression(max_iter=500), name="LR")
        config = PipelineConfig(task_type=task, metric="AUC", n_splits=3, n_trials=1, **kwargs)
    else:
        model = make_regressor(Ridge(), name="Ridge")
        config = PipelineConfig(task_type=task, metric="R2", n_splits=3, n_trials=1, **kwargs)
    with _mock_selector():
        return H2MLPipeline(config=config, models=[model]).run(_store(task))


@pytest.fixture(scope="module")
def reg_result():
    return _run("regression")


class TestBuildFinalModelAOA:
    def test_fitted_on_cv_rows_with_shap_weights_and_cv_folds(self, reg_result):
        final = reg_result.build_final_model()
        aoa = final.applicability
        assert aoa is not None
        assert aoa.feature_names == final.feature_names
        expected_w = IMPORTANCE.reindex(final.feature_names).to_numpy()
        np.testing.assert_allclose(aoa.weights, expected_w / expected_w.sum())
        assert aoa.nn.n_samples_fit_ == N
        assert aoa.train_di.size == sum(f.test_idx.size for f in reg_result.best_cv_result.folds)

    def test_area_of_applicability_accepts_dataframe(self, reg_result):
        final = reg_result.build_final_model()
        X = pd.DataFrame(reg_result.features.X, columns=FEATURE_NAMES)
        di, inside = final.area_of_applicability(X)
        assert di.shape == inside.shape == (N,)
        np.testing.assert_allclose(di, 0.0, atol=1e-12)  # training rows
        far = X.iloc[:1].copy()
        far[final.feature_names[0]] = 1e3
        assert not final.area_of_applicability(far)[1][0]

    def test_without_selector_importance_weights_are_equal(self, reg_result):
        final = replace(reg_result, selector=None).build_final_model()  # copy — the fixture is shared
        p = len(final.feature_names)
        np.testing.assert_allclose(final.applicability.weights, np.full(p, 1 / p))

    def test_holdout_refit_all_keeps_dev_rows_as_reference(self):
        result = _run("regression", holdout_frac=0.25, holdout_refit="all")
        final = result.build_final_model()
        assert final.applicability.nn.n_samples_fit_ == result.features.n_samples == 90

    def test_persists_with_final_model(self, reg_result, tmp_path):
        final = reg_result.build_final_model()
        final.save(tmp_path / "m.pkl")
        loaded = FinalModel.load(tmp_path / "m.pkl")
        X = reg_result.features.select(final.feature_names).X[:5] + 0.1
        np.testing.assert_allclose(loaded.area_of_applicability(X)[0], final.area_of_applicability(X)[0])

    def test_model_without_applicability_raises(self, reg_result):
        final = reg_result.build_final_model()
        final.applicability = None
        with pytest.raises(ValueError, match="No area of applicability"):
            final.area_of_applicability(reg_result.features.X)


# ---------------------------------------------------------------------------
# geo_predict columns
# ---------------------------------------------------------------------------


def _grid(final: FinalModel, X: np.ndarray) -> pl.DataFrame:
    df = pl.DataFrame({name: X[:, i] for i, name in enumerate(FEATURE_NAMES)})
    df = df.with_columns(pl.when(pl.int_range(pl.len()) == 2).then(None).otherwise(pl.col("f0")).alias("f0"))
    return df.with_row_index("index")


class TestGeoPredictAOA:
    def test_single_adds_aligned_di_and_aoa_columns(self, reg_result):
        final = reg_result.build_final_model()
        X = np.vstack([reg_result.features.X[:4], np.full((1, 4), 1e3)])
        series = _predict_single(_grid(final, X), final, col_name="sp", aoa=True)
        names = [s.name for s in series]
        assert names == ["sp", "sp_di", "sp_aoa"]
        di, aoa = series[1], series[2]
        assert di[2] is None and aoa[2] is None  # null feature row stays null
        keep = [0, 1, 3, 4]
        expected_di, expected_in = final.area_of_applicability(X[keep][:, [FEATURE_NAMES.index(f) for f in SELECTED]])
        np.testing.assert_allclose(di.to_numpy()[keep], expected_di, rtol=1e-5)
        assert aoa.to_list() == [True, True, None, True, False]
        assert list(expected_in) == [True, True, True, False]

    def test_aoa_off_by_default(self, reg_result):
        final = reg_result.build_final_model()
        series = _predict_single(_grid(final, reg_result.features.X[:4]), final, col_name="sp")
        assert [s.name for s in series] == ["sp"]

    def test_model_without_applicability_warns_and_skips(self, reg_result):
        final = reg_result.build_final_model()
        final.applicability = None
        messages: list[str] = []
        sink = logger.add(messages.append, level="WARNING")
        try:
            series = _predict_single(_grid(final, reg_result.features.X[:4]), final, col_name="sp", aoa=True)
        finally:
            logger.remove(sink)
        assert [s.name for s in series] == ["sp"]
        assert any("no area of applicability" in m for m in messages)

    def test_delta_aoa_requires_both_submodels_inside(self, reg_result):
        reg = reg_result.build_final_model()
        clf = _run("classification").build_final_model()
        delta = DeltaFinalModel(clf=clf, reg=reg)
        X = np.vstack([reg_result.features.X[:3], np.full((1, 4), 1e3)])
        series = _predict_delta_single(_grid(reg, X), delta, col_name="sp", alpha=None, aoa=True)
        assert [s.name for s in series] == ["sp", "sp_aoa"]
        assert series[1].to_list() == [True, True, None, False]
