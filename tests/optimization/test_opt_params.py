"""
tests/optimization/test_opt_params.py

Tests for opt_params.get_entry() — registry lookup for the optimizer.
"""

from __future__ import annotations

import pytest

from h2ml.optimization.opt_params import get_entry, ModelEntry


class TestGetEntry:
    # -----------------------------------------------------------------------
    # Successful lookups
    # -----------------------------------------------------------------------

    def test_returns_model_entry_for_rf_classifier(self):
        entry = get_entry("RandomForestClassifier", "classification")
        assert isinstance(entry, ModelEntry)

    def test_returns_model_entry_for_rf_regressor(self):
        entry = get_entry("RandomForestRegressor", "regression")
        assert isinstance(entry, ModelEntry)

    def test_returned_entry_has_param_fn(self):
        entry = get_entry("RandomForestClassifier", "classification")
        assert entry.param_fn is not None

    def test_returned_entry_class_matches_name(self):
        entry = get_entry("RandomForestClassifier", "classification")
        assert entry.model_cls.__name__ == "RandomForestClassifier"

    def test_regression_entry_is_different_from_classification(self):
        clf_entry = get_entry("RandomForestClassifier", "classification")
        reg_entry = get_entry("RandomForestRegressor", "regression")
        assert clf_entry.model_cls is not reg_entry.model_cls

    # -----------------------------------------------------------------------
    # KeyError — model not in registry
    # -----------------------------------------------------------------------

    def test_unknown_model_raises_key_error(self):
        with pytest.raises(KeyError, match="not found"):
            get_entry("FakeModel", "classification")

    def test_wrong_task_raises_key_error(self):
        """RandomForestClassifier is not in the regression registry."""
        with pytest.raises(KeyError):
            get_entry("RandomForestClassifier", "regression")

    # -----------------------------------------------------------------------
    # ValueError — opt_enabled=False
    # -----------------------------------------------------------------------

    def test_lr_raises_value_error(self):
        """LogisticRegression has opt_enabled=False."""
        with pytest.raises(ValueError, match="optimization disabled"):
            get_entry("LogisticRegression", "classification")

    def test_gaussian_nb_raises_value_error(self):
        with pytest.raises(ValueError, match="optimization disabled"):
            get_entry("GaussianNB", "classification")

    def test_knn_classifier_raises_value_error(self):
        with pytest.raises(ValueError, match="optimization disabled"):
            get_entry("KNeighborsClassifier", "classification")

    def test_poisson_regressor_raises_value_error(self):
        with pytest.raises(ValueError, match="optimization disabled"):
            get_entry("PoissonRegressor", "regression")
