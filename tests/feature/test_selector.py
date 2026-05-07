"""
tests/test_features/test_selector.py

Tests for FeatureSelector.

Note: SHAP computation is mocked in all tests — shap_importance.py has its
own test file. FeatureSelector tests focus on orchestration, state management,
and the fit/transform contract.

Coverage:
    Construction:
        - Attributes set correctly
        - _fitted is False before fit

    fit():
        - Sets _fitted to True
        - Returns self
        - selected_features_ is populated
        - feature_importance_ is a pd.Series
        - shap_values_ is a numpy array
        - Calls get_shap_values with correct args
        - Calls remove_correlated_features with correct args
        - Extracts inner estimator from PipelineStep

    transform():
        - Raises before fit
        - Returns PipelineData
        - Returned store has only selected features
        - y is preserved
        - Can transform a different store than fit store (test set)

    fit_transform():
        - Equivalent to fit then transform on same store
        - Returns reduced PipelineData

    Inspection helpers:
        - n_selected matches len(selected_features_)
        - removed_features raises before fit
        - removed_features returns correct list after fit
        - importance_summary returns DataFrame with correct columns
        - importance_summary selected column is boolean

    __repr__:
        - Contains task_type
        - Contains corr_threshold
        - Shows "not fitted" before fit
        - Shows selected count after fit
"""

from __future__ import annotations
from typing import Optional

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from h2ml.pipeline.base import TaskType
from h2ml.pipeline.step import make_classifier, make_regressor
from h2ml.features.feature_store import PipelineData
from h2ml.features.selector import FeatureSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]


def _make_store(n_samples: int = 100, seed: int = 42) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=n_samples)
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _mock_shap(selected: Optional[list[str]] = None, use_oof: bool = True):
    """
    Returns (patch_shap, patch_corr) context managers that stub out
    SHAP and correlation so FeatureSelector tests don't depend on SHAP.

    Patches get_oof_shap_values when use_oof=True (default),
    or get_shap_values when use_oof=False.
    """
    selected = selected or ["feat_a", "feat_c", "feat_e"]
    fake_importance = pd.Series(
        [0.5, 0.4, 0.3, 0.2, 0.1], index=FEATURE_NAMES
    ).sort_values(ascending=False)
    fake_shap_arr = np.ones((100, 5))

    shap_target = (
        "h2ml.features.selector.get_oof_shap_values"
        if use_oof else
        "h2ml.features.selector.get_shap_values"
    )
    patch_shap = patch(shap_target, return_value=(fake_shap_arr, fake_importance))
    patch_corr = patch(
        "h2ml.features.selector.remove_correlated_features",
        return_value=selected,
    )
    return patch_shap, patch_corr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> PipelineData:
    return _make_store()


@pytest.fixture
def clf_step():
    return make_classifier(RandomForestClassifier(n_estimators=10, random_state=42))


@pytest.fixture
def reg_step():
    return make_regressor(RandomForestRegressor(n_estimators=10, random_state=42))


@pytest.fixture
def selector(clf_step) -> FeatureSelector:
    return FeatureSelector(
        step           = clf_step,
        task_type      = TaskType.CLASSIFICATION,
        corr_threshold = 0.7,
    )


@pytest.fixture
def fitted_selector(selector, store) -> FeatureSelector:
    patch_shap, patch_corr = _mock_shap(["feat_a", "feat_c", "feat_e"])
    with patch_shap, patch_corr:
        selector.fit(store)
    return selector


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_task_type_set(self, selector):
        assert selector.task_type == TaskType.CLASSIFICATION

    def test_corr_threshold_set(self, selector):
        assert selector.corr_threshold == 0.7

    def test_not_fitted_on_init(self, selector):
        assert selector._fitted is False

    def test_selected_features_empty_on_init(self, selector):
        assert selector.selected_features_ == []

    def test_feature_importance_empty_on_init(self, selector):
        assert selector.feature_importance_.empty

    def test_shap_values_empty_on_init(self, selector):
        assert selector.shap_values_.size == 0


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_sets_fitted_flag(self, selector, store):
        patch_shap, patch_corr = _mock_shap()
        with patch_shap, patch_corr:
            selector.fit(store)
        assert selector._fitted is True

    def test_fit_returns_self(self, selector, store):
        patch_shap, patch_corr = _mock_shap()
        with patch_shap, patch_corr:
            result = selector.fit(store)
        assert result is selector

    def test_fit_populates_selected_features(self, selector, store):
        selected = ["feat_a", "feat_c"]
        patch_shap, patch_corr = _mock_shap(selected)
        with patch_shap, patch_corr:
            selector.fit(store)
        assert selector.selected_features_ == selected

    def test_fit_populates_feature_importance(self, selector, store):
        patch_shap, patch_corr = _mock_shap()
        with patch_shap, patch_corr:
            selector.fit(store)
        assert isinstance(selector.feature_importance_, pd.Series)
        assert len(selector.feature_importance_) == len(FEATURE_NAMES)

    def test_fit_populates_shap_values(self, selector, store):
        patch_shap, patch_corr = _mock_shap()
        with patch_shap, patch_corr:
            selector.fit(store)
        assert isinstance(selector.shap_values_, np.ndarray)
        assert selector.shap_values_.size > 0

    def test_fit_passes_corr_threshold(self, clf_step, store):
        """corr_threshold should be forwarded to remove_correlated_features."""
        selector = FeatureSelector(clf_step, TaskType.CLASSIFICATION, corr_threshold=0.5)
        patch_shap, patch_corr = _mock_shap()
        with patch_shap, patch_corr as mock_corr:
            selector.fit(store)
            _, call_kwargs = mock_corr.call_args
            assert call_kwargs.get("corr_threshold") == 0.5 or \
                   mock_corr.call_args[0][2] == 0.5  # positional fallback

    def test_fit_calls_oof_shap_by_default(self, clf_step, store):
        """Default use_oof=True should call get_oof_shap_values with step=self.step."""
        selector = FeatureSelector(clf_step, TaskType.CLASSIFICATION)
        patch_shap, patch_corr = _mock_shap(use_oof=True)
        with patch_shap as mock_oof, patch_corr:
            selector.fit(store)
        called_step = mock_oof.call_args[1].get("step") or mock_oof.call_args[0][0]
        assert called_step is clf_step

    def test_fit_use_oof_false_calls_full_shap(self, clf_step, store):
        """use_oof=False should call get_shap_values (full-dataset path)."""
        selector = FeatureSelector(clf_step, TaskType.CLASSIFICATION, use_oof=False)
        patch_shap, patch_corr = _mock_shap(use_oof=False)
        with patch_shap as mock_full, patch_corr:
            selector.fit(store)
        assert mock_full.called


# ---------------------------------------------------------------------------
# transform()
# ---------------------------------------------------------------------------

class TestTransform:

    def test_transform_raises_before_fit(self, selector, store):
        with pytest.raises(RuntimeError, match="not been fitted"):
            selector.transform(store)

    def test_transform_returns_feature_store(self, fitted_selector, store):
        result = fitted_selector.transform(store)
        assert isinstance(result, PipelineData)

    def test_transform_reduces_features(self, fitted_selector, store):
        result = fitted_selector.transform(store)
        assert result.n_features < store.n_features

    def test_transform_feature_names_match_selected(self, fitted_selector, store):
        result = fitted_selector.transform(store)
        assert result.feature_names == fitted_selector.selected_features_

    def test_transform_preserves_y(self, fitted_selector, store):
        result = fitted_selector.transform(store)
        np.testing.assert_array_equal(result.y, store.y)

    def test_transform_on_different_store(self, fitted_selector):
        """Transform should work on a test store with the same features."""
        test_store = _make_store(n_samples=30, seed=99)
        result = fitted_selector.transform(test_store)
        assert result.feature_names == fitted_selector.selected_features_
        assert result.n_samples == 30

    def test_transform_n_samples_unchanged(self, fitted_selector, store):
        result = fitted_selector.transform(store)
        assert result.n_samples == store.n_samples


# ---------------------------------------------------------------------------
# fit_transform()
# ---------------------------------------------------------------------------

class TestFitTransform:

    def test_fit_transform_returns_feature_store(self, selector, store):
        patch_shap, patch_corr = _mock_shap(["feat_a", "feat_c"])
        with patch_shap, patch_corr:
            result = selector.fit_transform(store)
        assert isinstance(result, PipelineData)

    def test_fit_transform_equivalent_to_fit_then_transform(self, clf_step, store):
        patch_shap, patch_corr = _mock_shap(["feat_a", "feat_b"])

        sel_a = FeatureSelector(clf_step, TaskType.CLASSIFICATION, corr_threshold=0.7)
        sel_b = FeatureSelector(clf_step, TaskType.CLASSIFICATION, corr_threshold=0.7)

        with patch_shap, patch_corr:
            result_ft    = sel_a.fit_transform(store)
            _ = sel_b.fit(store)

        # Re-patch for the second transform call
        with patch_shap, patch_corr:
            result_f_t   = sel_b.transform(store)

        assert result_ft.feature_names == result_f_t.feature_names

    def test_fit_transform_sets_fitted(self, selector, store):
        patch_shap, patch_corr = _mock_shap()
        with patch_shap, patch_corr:
            selector.fit_transform(store)
        assert selector._fitted is True


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

class TestInspectionHelpers:

    def test_n_selected_matches_selected_features(self, fitted_selector):
        assert fitted_selector.n_selected == len(fitted_selector.selected_features_)

    def test_removed_features_raises_before_fit(self, selector):
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = selector.removed_features

    def test_removed_features_correct(self, fitted_selector):
        removed = fitted_selector.removed_features
        selected = set(fitted_selector.selected_features_)
        all_features = set(fitted_selector.feature_importance_.index)
        assert set(removed) == all_features - selected

    def test_removed_plus_selected_equals_all(self, fitted_selector):
        all_features = set(fitted_selector.feature_importance_.index)
        selected     = set(fitted_selector.selected_features_)
        removed      = set(fitted_selector.removed_features)
        assert selected | removed == all_features
        assert selected & removed == set()

    def test_importance_summary_raises_before_fit(self, selector):
        with pytest.raises(RuntimeError, match="not been fitted"):
            selector.importance_summary()

    def test_importance_summary_returns_dataframe(self, fitted_selector):
        df = fitted_selector.importance_summary()
        assert isinstance(df, pd.DataFrame)

    def test_importance_summary_columns(self, fitted_selector):
        df = fitted_selector.importance_summary()
        assert set(df.columns) == {"feature", "importance", "selected"}

    def test_importance_summary_selected_is_boolean(self, fitted_selector):
        df = fitted_selector.importance_summary()
        assert df["selected"].dtype == bool

    def test_importance_summary_row_count(self, fitted_selector):
        df = fitted_selector.importance_summary()
        assert len(df) == len(FEATURE_NAMES)

    def test_importance_summary_selected_count(self, fitted_selector):
        df = fitted_selector.importance_summary()
        assert df["selected"].sum() == fitted_selector.n_selected


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_task_type(self, selector):
        assert "classification" in repr(selector)

    def test_repr_contains_corr_threshold(self, selector):
        assert "0.7" in repr(selector)

    def test_repr_not_fitted_before_fit(self, selector):
        assert "not fitted" in repr(selector)

    def test_repr_shows_selected_count_after_fit(self, fitted_selector):
        assert f"selected={fitted_selector.n_selected}" in repr(fitted_selector)

    def test_repr_contains_min_features(self, selector):
        assert "min_features" in repr(selector)


# ---------------------------------------------------------------------------
# min_features guarantee
# ---------------------------------------------------------------------------

class TestMinFeatures:

    def _make_selector(self, clf_step, min_features: int, selected: list[str]) -> FeatureSelector:
        """Build a selector whose corr filter would return `selected`, then fit with mocks."""
        selector = FeatureSelector(
            clf_step, TaskType.CLASSIFICATION,
            corr_threshold=0.01,   # very low — triggers the guard
            min_features=min_features,
        )
        patch_shap, patch_corr = _mock_shap(selected=selected)
        store = _make_store()
        with patch_shap, patch_corr:
            selector.fit(store)
        return selector

    def test_no_guard_by_default(self, clf_step):
        """Default min_features=1: single feature returned → no restoration."""
        sel = self._make_selector(clf_step, min_features=1, selected=["feat_a"])
        assert sel.selected_features_ == ["feat_a"]

    def test_restores_to_min_when_below(self, clf_step):
        """Correlation filter returns 1 feature; min_features=3 → 3 features total."""
        sel = self._make_selector(clf_step, min_features=3, selected=["feat_a"])
        assert len(sel.selected_features_) == 3

    def test_restored_features_are_top_shap(self, clf_step):
        """Restored features should come from the top of feature_importance_."""
        sel = self._make_selector(clf_step, min_features=3, selected=["feat_a"])
        # feat_a is already selected; restoration should pull feat_b then feat_c
        # (importance order: feat_a=0.5, feat_b=0.4, feat_c=0.3, feat_d=0.2, feat_e=0.1)
        assert set(sel.selected_features_) == {"feat_a", "feat_b", "feat_c"}

    def test_order_preserved_by_importance(self, clf_step):
        """Selected features maintain importance order after restoration."""
        sel = self._make_selector(clf_step, min_features=3, selected=["feat_a"])
        expected_order = [f for f in sel.feature_importance_.index if f in set(sel.selected_features_)]
        assert sel.selected_features_ == expected_order

    def test_no_restoration_when_already_above_min(self, clf_step):
        """When corr filter keeps enough features, no restoration occurs."""
        sel = self._make_selector(clf_step, min_features=2, selected=["feat_a", "feat_b", "feat_c"])
        assert len(sel.selected_features_) == 3

    def test_min_features_exceeds_total_features(self, clf_step):
        """min_features > n_features → all features retained."""
        sel = self._make_selector(clf_step, min_features=10, selected=["feat_a"])
        assert len(sel.selected_features_) == len(FEATURE_NAMES)

    def test_invalid_min_features_raises(self, clf_step):
        with pytest.raises((ValueError, TypeError)):
            FeatureSelector(clf_step, TaskType.CLASSIFICATION, min_features=0)
