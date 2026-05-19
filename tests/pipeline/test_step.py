"""
tests/test_pipeline/test_step.py

Tests for ModelWrapper and factory functions.

Coverage:
    - Task type inference from sklearn estimators
    - fit() sets _fitted flag correctly
    - transform() pass-through for classifiers/regressors
    - transform() delegates for preprocessors
    - predict() works after fit
    - predict_proba() works for classifiers
    - predict_proba() raises for regressors
    - requires_scaling flag is set and readable
    - _check_fitted raises before fit() is called
    - __repr__ contains expected fields
    - factory functions produce correct step types
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from h2ml.pipeline.base import TaskType
from h2ml.pipeline.step import (
    make_classifier,
    make_preprocessor,
    make_regressor,
)


# ---------------------------------------------------------------------------
# Task type inference
# ---------------------------------------------------------------------------


class TestTaskTypeInference:
    def test_classifier_inferred(self, rf_classifier_step):
        assert rf_classifier_step.task_type == TaskType.CLASSIFICATION

    def test_regressor_inferred(self, rf_regressor_step):
        assert rf_regressor_step.task_type == TaskType.REGRESSION

    def test_transformer_inferred(self, scaler_step):
        assert scaler_step.task_type == TaskType.TRANSFORM


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_sets_fitted_flag(self, rf_classifier_step, classification_data):
        X, y = classification_data
        assert rf_classifier_step._fitted is False
        rf_classifier_step.fit(X, y)
        assert rf_classifier_step._fitted is True

    def test_fit_returns_self(self, rf_classifier_step, classification_data):
        X, y = classification_data
        result = rf_classifier_step.fit(X, y)
        assert result is rf_classifier_step

    def test_fit_regressor(self, rf_regressor_step, regression_data):
        X, y = regression_data
        rf_regressor_step.fit(X, y)
        assert rf_regressor_step._fitted is True

    def test_fit_preprocessor(self, scaler_step, classification_data):
        X, y = classification_data
        scaler_step.fit(X, y)
        assert scaler_step._fitted is True


# ---------------------------------------------------------------------------
# _check_fitted guard
# ---------------------------------------------------------------------------


class TestCheckFitted:
    def test_predict_raises_before_fit(self, rf_classifier_step, classification_data):
        X, _ = classification_data
        with pytest.raises(RuntimeError, match="not been fitted"):
            rf_classifier_step.predict(X)

    def test_transform_raises_before_fit(self, rf_classifier_step, classification_data):
        X, _ = classification_data
        with pytest.raises(RuntimeError, match="not been fitted"):
            rf_classifier_step.transform(X)

    def test_predict_proba_raises_before_fit(self, rf_classifier_step, classification_data):
        X, _ = classification_data
        with pytest.raises(RuntimeError, match="not been fitted"):
            rf_classifier_step.predict_proba(X)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_classifier_shape(self, fitted_rf_classifier, classification_data):
        X, y = classification_data
        preds = fitted_rf_classifier.predict(X)
        assert preds.shape == (100,)

    def test_predict_classifier_values(self, fitted_rf_classifier, classification_data):
        X, _ = classification_data
        preds = fitted_rf_classifier.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_regressor_shape(self, fitted_rf_regressor, regression_data):
        X, _ = regression_data
        preds = fitted_rf_regressor.predict(X)
        assert preds.shape == (100,)

    def test_predict_regressor_is_continuous(self, fitted_rf_regressor, regression_data):
        X, _ = regression_data
        preds = fitted_rf_regressor.predict(X)
        assert preds.dtype in [np.float32, np.float64]


# ---------------------------------------------------------------------------
# predict_proba()
# ---------------------------------------------------------------------------


class TestPredictProba:
    def test_predict_proba_classifier_shape(self, fitted_rf_classifier, classification_data):
        X, _ = classification_data
        proba = fitted_rf_classifier.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_predict_proba_sums_to_one(self, fitted_rf_classifier, classification_data):
        X, _ = classification_data
        proba = fitted_rf_classifier.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(100), atol=1e-6)

    def test_predict_proba_raises_on_regressor(self, fitted_rf_regressor, regression_data):
        X, _ = regression_data
        with pytest.raises(TypeError, match="only available for classifiers"):
            fitted_rf_regressor.predict_proba(X)


# ---------------------------------------------------------------------------
# transform()
# ---------------------------------------------------------------------------


class TestTransform:
    def test_transform_classifier_passthrough(self, fitted_rf_classifier, classification_data):
        X, _ = classification_data
        result = fitted_rf_classifier.transform(X)
        np.testing.assert_array_equal(result, X)

    def test_transform_regressor_passthrough(self, fitted_rf_regressor, regression_data):
        X, _ = regression_data
        result = fitted_rf_regressor.transform(X)
        np.testing.assert_array_equal(result, X)

    def test_transform_preprocessor_changes_data(self, scaler_step, classification_data):
        X, y = classification_data
        scaler_step.fit(X, y)
        result = scaler_step.transform(X)
        assert result.shape == X.shape
        # Scaled output should have near-zero mean and unit variance
        np.testing.assert_allclose(result.mean(axis=0), np.zeros(5), atol=0.1)
        np.testing.assert_allclose(result.std(axis=0), np.ones(5), atol=0.1)


# ---------------------------------------------------------------------------
# requires_scaling flag
# ---------------------------------------------------------------------------


class TestRequiresScaling:
    def test_requires_scaling_default_false(self, rf_classifier_step):
        assert rf_classifier_step.requires_scaling is False

    def test_requires_scaling_set_true(self, svc_step):
        assert svc_step.requires_scaling is True

    def test_requires_scaling_readable_as_attribute(self, rf_classifier_step):
        assert hasattr(rf_classifier_step, "requires_scaling")


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_name(self, rf_classifier_step):
        assert "RandomForestClassifier" in repr(rf_classifier_step)

    def test_repr_contains_task_type(self, rf_classifier_step):
        assert "classification" in repr(rf_classifier_step)

    def test_repr_contains_fitted_false(self, rf_classifier_step):
        assert "fitted=False" in repr(rf_classifier_step)

    def test_repr_contains_fitted_true(self, fitted_rf_classifier):
        assert "fitted=True" in repr(fitted_rf_classifier)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    def test_make_classifier_task_type(self):
        step = make_classifier(RandomForestClassifier())
        assert step.task_type == TaskType.CLASSIFICATION

    def test_make_regressor_task_type(self):
        step = make_regressor(RandomForestRegressor())
        assert step.task_type == TaskType.REGRESSION

    def test_make_preprocessor_task_type(self):
        step = make_preprocessor(StandardScaler())
        assert step.task_type == TaskType.TRANSFORM

    def test_make_classifier_custom_name(self):
        step = make_classifier(RandomForestClassifier(), name="my_rf")
        assert step.name == "my_rf"

    def test_make_classifier_default_name(self):
        step = make_classifier(RandomForestClassifier())
        assert step.name == "RandomForestClassifier"
