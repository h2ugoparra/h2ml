"""
Shared fixtures for all h2ml tests.
All data is deterministic via fixed random seeds.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from h2ml.pipeline.step import ModelWrapper, make_classifier, make_regressor, make_preprocessor

# ---------------------------------------------------------------------------
# Raw data fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """100 samples, 5 features, binary target."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = rng.integers(0, 2, size=100)
    return X, y

@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    """100 samples, 5 features, continuous target."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = rng.standard_normal(100)
    return X, y

@pytest.fixture
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    """100 samples, 5 features, 3-class target."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = rng.integers(0, 3, size=100)
    return X, y

# ---------------------------------------------------------------------------
# ModelWrapper fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rf_classifier_step() -> ModelWrapper:
    """Unfitted RandomForest classifier step."""
    return make_classifier(RandomForestClassifier(n_estimators=10, random_state=42))

@pytest.fixture
def rf_regressor_step() -> ModelWrapper:
    """Unfitted RandomForest regressor step."""
    return make_regressor(RandomForestRegressor(n_estimators=10, random_state=42))


@pytest.fixture
def svc_step() -> ModelWrapper:
    """SVC step with calibration enabled (required for predict_proba)."""
    return make_classifier(
        SVC(random_state=42),
        requires_scaling=True
    )


@pytest.fixture
def scaler_step() -> ModelWrapper:
    """StandardScaler preprocessor step."""
    return make_preprocessor(StandardScaler(), name="scaler")


@pytest.fixture
def fitted_rf_classifier(
    rf_classifier_step: ModelWrapper,
    classification_data: tuple[np.ndarray, np.ndarray],
) -> ModelWrapper:
    """Pre-fitted RandomForest classifier step — use when fit() is not what's being tested."""
    X, y = classification_data
    return rf_classifier_step.fit(X, y)


@pytest.fixture
def fitted_rf_regressor(
    rf_regressor_step: ModelWrapper,
    regression_data: tuple[np.ndarray, np.ndarray],
) -> ModelWrapper:
    """Pre-fitted RandomForest regressor step."""
    X, y = regression_data
    return rf_regressor_step.fit(X, y)
