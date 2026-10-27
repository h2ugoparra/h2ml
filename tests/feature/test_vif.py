"""
tests/feature/test_vif.py

Tests for the variance-inflation-factor filter: variance_inflation_factors(),
remove_high_vif_features(), FeatureSelector(max_vif=...) and PipelineConfig.max_vif.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from h2ml.core.base import TaskType
from h2ml.core.feature_store import PipelineData
from h2ml.features.correlation import (
    remove_correlated_features,
    remove_high_vif_features,
    variance_inflation_factors,
)
from h2ml.features.selector import FeatureSelector
from h2ml.pipeline.pipeline import PipelineConfig
from h2ml.pipeline.step import make_classifier


def _collinear_frame(n: int = 300, seed: int = 0, noise: float = 0.05) -> pd.DataFrame:
    """a, b, c independent; d ≈ a + b + c — each pairwise |r| with d is only ~0.58."""
    rng = np.random.default_rng(seed)
    a, b, c, e = rng.standard_normal((4, n))
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": a + b + c + noise * rng.standard_normal(n), "e": e})


class TestVarianceInflationFactors:
    def test_matches_one_over_one_minus_r2(self):
        df = _collinear_frame()
        vif = variance_inflation_factors(df)
        for col in df.columns:
            others = df.drop(columns=col)
            r2 = LinearRegression().fit(others, df[col]).score(others, df[col])
            assert vif[col] == pytest.approx(1 / (1 - r2), rel=1e-6)

    def test_exact_linear_combination_is_infinite(self):
        df = _collinear_frame(noise=0.0)
        assert np.isinf(variance_inflation_factors(df)["d"])

    def test_independent_features_near_one(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.standard_normal((500, 3)), columns=["x", "y", "z"])
        assert (variance_inflation_factors(df) < 1.1).all()


class TestRemoveHighVifFeatures:
    def test_pairwise_filter_misses_what_vif_catches(self):
        df = _collinear_frame()
        ranking = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=["a", "b", "c", "d", "e"])
        after_corr = remove_correlated_features(df, ranking, corr_threshold=0.7)
        assert after_corr == ["a", "b", "c", "d", "e"]  # no pair above 0.7
        assert remove_high_vif_features(df, after_corr, max_vif=10) == ["a", "b", "c", "e"]

    def test_importance_decides_which_collinear_member_survives(self):
        df = _collinear_frame()
        # d ranked first: the lowest-ranked of the collinear group (c) is dropped instead
        kept = remove_high_vif_features(df, ["d", "a", "b", "c", "e"], max_vif=10)
        assert kept == ["d", "a", "b", "e"]
        assert (variance_inflation_factors(df[kept]) <= 10).all()

    def test_nothing_dropped_below_threshold(self):
        df = _collinear_frame(noise=3.0)  # weak dependence
        assert remove_high_vif_features(df, list(df.columns), max_vif=10) == list(df.columns)

    def test_single_feature_kept(self):
        df = _collinear_frame()
        assert remove_high_vif_features(df, ["a"], max_vif=2) == ["a"]

    def test_invalid_max_vif_raises(self):
        with pytest.raises(ValueError, match="max_vif"):
            remove_high_vif_features(_collinear_frame(), ["a", "b"], max_vif=1.0)


class TestSelectorMaxVif:
    def _fit(self, max_vif):
        df = _collinear_frame()
        store = PipelineData(X=df.to_numpy(), feature_names=list(df.columns), y=(df["a"] > 0).astype(int).to_numpy())
        importance = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=list(df.columns))
        selector = FeatureSelector(
            make_classifier(RandomForestClassifier(n_estimators=5, random_state=0)),
            TaskType.CLASSIFICATION,
            corr_threshold=0.7,
            max_vif=max_vif,
        )
        with patch(
            "h2ml.features.selector.get_oof_shap_values",
            return_value=(np.zeros((len(df), 5)), importance),
        ):
            selector.fit(store)
        return selector

    def test_off_by_default(self):
        assert self._fit(None).selected_features_ == ["a", "b", "c", "d", "e"]

    def test_drops_multicollinear_feature(self):
        selector = self._fit(10)
        assert selector.selected_features_ == ["a", "b", "c", "e"]
        assert "d" in selector.removed_features

    def test_invalid_max_vif_raises(self):
        with pytest.raises(ValueError, match="max_vif"):
            FeatureSelector(
                make_classifier(RandomForestClassifier()),
                TaskType.CLASSIFICATION,
                max_vif=0.5,
            )


class TestConfigMaxVif:
    def test_default_none(self):
        assert PipelineConfig().max_vif is None

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="max_vif"):
            PipelineConfig(max_vif=1.0)

    def test_forwarded_to_selector(self):
        from h2ml.pipeline.pipeline import H2MLPipeline

        pipeline = H2MLPipeline(PipelineConfig(max_vif=5.0))
        rng = np.random.default_rng(0)
        store = PipelineData(X=rng.standard_normal((20, 2)), feature_names=["x", "y"], y=rng.integers(0, 2, 20))
        selector = pipeline._make_selector(pipeline.models[0].name, store, None)
        assert selector.max_vif == 5.0
