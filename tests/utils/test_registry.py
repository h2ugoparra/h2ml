"""
tests/utils/test_registry.py

Tests for ModelEntry, CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY, and build_models().
"""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from h2ml.utils.registry import (
    ModelEntry,
    CLASSIFIER_REGISTRY,
    REGRESSOR_REGISTRY,
    build_models,
)
from h2ml.pipeline.step import ModelWrapper


# ---------------------------------------------------------------------------
# ModelEntry
# ---------------------------------------------------------------------------


class TestModelEntry:
    def test_build_model_returns_model_wrapper(self):
        entry = CLASSIFIER_REGISTRY["RandomForestClassifier"]
        assert isinstance(entry.build_model(), ModelWrapper)

    def test_build_model_uses_default_kwargs(self):
        entry = CLASSIFIER_REGISTRY["RandomForestClassifier"]
        wrapper = entry.build_model()
        assert wrapper.estimator.random_state == 42

    def _entry_with_param_fn(self, **kwargs) -> ModelEntry:
        """Minimal valid entry: param_fn provided so opt_enabled=True is legal."""
        return ModelEntry(model_cls=RandomForestClassifier, param_fn=lambda t: {}, **kwargs)

    def test_requires_scaling_false_by_default(self):
        entry = self._entry_with_param_fn()
        assert entry.requires_scaling is False

    def test_opt_enabled_true_by_default(self):
        entry = self._entry_with_param_fn()
        assert entry.opt_enabled is True

    def test_opt_enabled_true_without_param_fn_raises(self):
        """opt_enabled=True with param_fn=None is an invalid combination."""
        with pytest.raises(ValueError, match="param_fn"):
            ModelEntry(model_cls=RandomForestClassifier, opt_enabled=True, param_fn=None)

    def test_opt_enabled_false_without_param_fn_is_valid(self):
        """opt_enabled=False with param_fn=None is the legitimate disabled state."""
        entry = ModelEntry(model_cls=LogisticRegression, opt_enabled=False, param_fn=None)
        assert entry.opt_enabled is False


# ---------------------------------------------------------------------------
# CLASSIFIER_REGISTRY
# ---------------------------------------------------------------------------


class TestClassifierRegistry:
    def test_contains_logistic_regression(self):
        assert "LogisticRegression" in CLASSIFIER_REGISTRY

    def test_contains_random_forest(self):
        assert "RandomForestClassifier" in CLASSIFIER_REGISTRY

    def test_contains_gaussian_nb(self):
        assert "GaussianNB" in CLASSIFIER_REGISTRY

    def test_lr_requires_scaling(self):
        assert CLASSIFIER_REGISTRY["LogisticRegression"].requires_scaling is True

    def test_svc_requires_scaling(self):
        assert CLASSIFIER_REGISTRY["SVC"].requires_scaling is True

    def test_knn_requires_scaling(self):
        assert CLASSIFIER_REGISTRY["KNeighborsClassifier"].requires_scaling is True

    def test_rf_does_not_require_scaling(self):
        assert CLASSIFIER_REGISTRY["RandomForestClassifier"].requires_scaling is False

    def test_lr_opt_disabled(self):
        assert CLASSIFIER_REGISTRY["LogisticRegression"].opt_enabled is False

    def test_gaussian_nb_opt_disabled(self):
        assert CLASSIFIER_REGISTRY["GaussianNB"].opt_enabled is False

    def test_knn_opt_disabled(self):
        assert CLASSIFIER_REGISTRY["KNeighborsClassifier"].opt_enabled is False

    def test_rf_opt_enabled(self):
        assert CLASSIFIER_REGISTRY["RandomForestClassifier"].opt_enabled is True

    def test_rf_has_param_fn(self):
        assert CLASSIFIER_REGISTRY["RandomForestClassifier"].param_fn is not None

    def test_lr_param_fn_is_none(self):
        assert CLASSIFIER_REGISTRY["LogisticRegression"].param_fn is None

    def test_all_entries_are_model_entry(self):
        for name, entry in CLASSIFIER_REGISTRY.items():
            assert isinstance(entry, ModelEntry), f"{name} is not a ModelEntry"


# ---------------------------------------------------------------------------
# REGRESSOR_REGISTRY
# ---------------------------------------------------------------------------


class TestRegressorRegistry:
    def test_contains_random_forest_regressor(self):
        assert "RandomForestRegressor" in REGRESSOR_REGISTRY

    def test_contains_poisson_regressor(self):
        assert "PoissonRegressor" in REGRESSOR_REGISTRY

    def test_poisson_requires_scaling(self):
        assert REGRESSOR_REGISTRY["PoissonRegressor"].requires_scaling is True

    def test_svr_requires_scaling(self):
        assert REGRESSOR_REGISTRY["SVR"].requires_scaling is True

    def test_rf_does_not_require_scaling(self):
        assert REGRESSOR_REGISTRY["RandomForestRegressor"].requires_scaling is False

    def test_poisson_opt_disabled(self):
        assert REGRESSOR_REGISTRY["PoissonRegressor"].opt_enabled is False

    def test_knn_opt_disabled(self):
        assert REGRESSOR_REGISTRY["KNeighborsRegressor"].opt_enabled is False

    def test_rf_opt_enabled(self):
        assert REGRESSOR_REGISTRY["RandomForestRegressor"].opt_enabled is True

    def test_all_entries_are_model_entry(self):
        for name, entry in REGRESSOR_REGISTRY.items():
            assert isinstance(entry, ModelEntry), f"{name} is not a ModelEntry"


# ---------------------------------------------------------------------------
# build_models()
# ---------------------------------------------------------------------------


class TestBuildModels:
    def test_classification_returns_list(self):
        models = build_models("classification")
        assert isinstance(models, list)
        assert len(models) > 0

    def test_regression_returns_list(self):
        models = build_models("regression")
        assert isinstance(models, list)
        assert len(models) > 0

    def test_classification_returns_model_wrappers(self):
        for m in build_models("classification"):
            assert isinstance(m, ModelWrapper)

    def test_regression_returns_model_wrappers(self):
        for m in build_models("regression"):
            assert isinstance(m, ModelWrapper)

    def test_classification_scaling_flags_respected(self):
        models = build_models("classification")
        model_map = {m.name: m for m in models}
        assert model_map["LogisticRegression"].requires_scaling is True
        assert model_map["RandomForestClassifier"].requires_scaling is False

    def test_classification_and_regression_models_differ(self):
        clf_names = {m.name for m in build_models("classification")}
        reg_names = {m.name for m in build_models("regression")}
        assert clf_names != reg_names
