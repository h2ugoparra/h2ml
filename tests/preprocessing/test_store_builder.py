"""
tests/test_preprocessing/test_store_builder.py

Tests for build_transformed_stores() in preprocessing/store_builder.py.

Coverage:
    Return type and structure:
        - Returns a dict
        - Dict values are PipelineData instances
        - Dict keys are transform names

    Feature handling:
        - X is shared across all stores (same array)
        - feature_names match across all stores
        - n_features correct across all stores

    y transformations:
        - count store y equals original y
        - log store y equals log1p(y)
        - sqrt store y equals sqrt(y)
        - wincount store is skipped when no outliers
        - wincount store y has clipped outliers when present
        - winlog store is skipped when log(y) has no outliers
        - winsqrt store is skipped when sqrt(y) has no outliers

    None filtering:
        - Transforms returning None are excluded from result
        - Result only contains non-None transforms
        - No KeyError when all transforms are skipped

    transforms parameter:
        - Subset of transforms when list provided
        - All transforms when None provided
        - Unknown transform name raises KeyError

    PipelineData integrity:
        - Each store passes PipelineData shape validation
        - y length matches n_samples in each store
        - feature_names length matches n_features in each store
"""

from __future__ import annotations

import numpy as np
import pytest

from h2ml.features.feature_store import PipelineData
from h2ml.preprocessing.transform_stores import (
    build_transform_stores as build_transformed_stores,
)
from h2ml.preprocessing.transforms import Y_TRANSFORMS, log_transform, sqrt_transform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_NAMES = [f"feat_{i}" for i in range(5)]


@pytest.fixture
def clean_y() -> np.ndarray:
    """No outliers — winsorize-based transforms will be skipped."""
    return np.random.default_rng(42).uniform(1, 10, size=100)


@pytest.fixture
def outlier_y() -> np.ndarray:
    """Has clear upper outliers — winsorize-based transforms will run."""
    rng = np.random.default_rng(42)
    base = rng.uniform(1, 10, size=95)
    return np.concatenate([base, np.array([500.0, 1000.0, 800.0, 750.0, 900.0])])


@pytest.fixture
def X() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((100, len(FEATURE_NAMES)))


@pytest.fixture
def X_outlier() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((100, len(FEATURE_NAMES)))


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnTypeAndStructure:
    def test_returns_dict(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        assert isinstance(result, dict)

    def test_values_are_feature_stores(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for store in result.values():
            assert isinstance(store, PipelineData)

    def test_keys_are_strings(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for key in result.keys():
            assert isinstance(key, str)

    def test_keys_are_transform_names(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        assert set(result.keys()).issubset(set(Y_TRANSFORMS.keys()))


# ---------------------------------------------------------------------------
# Feature handling
# ---------------------------------------------------------------------------


class TestFeatureHandling:
    def test_x_shared_across_stores(self, X, clean_y):
        """All stores should reference the same X array."""
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for store in result.values():
            np.testing.assert_array_equal(store.X, X)

    def test_feature_names_consistent(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for store in result.values():
            assert store.feature_names == FEATURE_NAMES

    def test_n_features_correct(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for store in result.values():
            assert store.n_features == len(FEATURE_NAMES)

    def test_n_samples_correct(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for store in result.values():
            assert store.n_samples == len(clean_y)


# ---------------------------------------------------------------------------
# y transformations — with clean data
# ---------------------------------------------------------------------------


class TestYTransformationsClean:
    def test_count_y_equals_original(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        assert "count" in result
        np.testing.assert_array_equal(result["count"].y, clean_y)

    def test_log_y_equals_log1p(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        assert "log" in result
        np.testing.assert_array_almost_equal(result["log"].y, np.log1p(clean_y))

    def test_sqrt_y_equals_sqrt(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        assert "sqrt" in result
        np.testing.assert_array_almost_equal(result["sqrt"].y, np.sqrt(clean_y))

    def test_wincount_skipped_when_no_outliers(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        assert "wincount" not in result

    def test_always_has_count_log_sqrt(self, X, clean_y):
        """These three transforms never return None."""
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        assert "count" in result
        assert "log" in result
        assert "sqrt" in result


# ---------------------------------------------------------------------------
# y transformations — with outlier data
# ---------------------------------------------------------------------------


class TestYTransformationsOutlier:
    def test_wincount_present_when_outliers(self, X_outlier, outlier_y):
        result = build_transformed_stores(X_outlier, outlier_y, FEATURE_NAMES)
        assert "wincount" in result

    def test_wincount_y_clips_outliers(self, X_outlier, outlier_y):
        result = build_transformed_stores(X_outlier, outlier_y, FEATURE_NAMES)
        Q1, Q3 = np.percentile(outlier_y, [25, 75])
        upper_limit = Q3 + 1.5 * (Q3 - Q1)
        assert (result["wincount"].y <= upper_limit + 1e-9).all()

    def test_wincount_max_equals_upper_limit(self, X_outlier, outlier_y):
        result = build_transformed_stores(X_outlier, outlier_y, FEATURE_NAMES)
        Q1, Q3 = np.percentile(outlier_y, [25, 75])
        upper_limit = Q3 + 1.5 * (Q3 - Q1)
        assert result["wincount"].y.max() == pytest.approx(upper_limit, rel=1e-6)

    def test_winlog_present_when_outliers_survive_log(self, X_outlier, outlier_y):
        log_y = log_transform(outlier_y)
        Q1, Q3 = np.percentile(log_y, [25, 75])
        upper = Q3 + 1.5 * (Q3 - Q1)
        has_log_outliers = (log_y > upper).any()

        result = build_transformed_stores(X_outlier, outlier_y, FEATURE_NAMES)

        if has_log_outliers:
            assert "winlog" in result
        else:
            assert "winlog" not in result

    def test_winsqrt_present_when_outliers_survive_sqrt(self, X_outlier, outlier_y):
        sqrt_y = sqrt_transform(outlier_y)
        Q1, Q3 = np.percentile(sqrt_y, [25, 75])
        upper = Q3 + 1.5 * (Q3 - Q1)
        has_sqrt_outliers = (sqrt_y > upper).any()

        result = build_transformed_stores(X_outlier, outlier_y, FEATURE_NAMES)

        if has_sqrt_outliers:
            assert "winsqrt" in result
        else:
            assert "winsqrt" not in result


# ---------------------------------------------------------------------------
# None filtering
# ---------------------------------------------------------------------------


class TestNoneFiltering:
    def test_none_transforms_excluded(self, X, clean_y):
        """Winsorize-based transforms return None for clean data — excluded."""
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for store in result.values():
            assert store is not None

    def test_result_only_contains_valid_stores(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for key, store in result.items():
            assert isinstance(store, PipelineData), (
                f"Store for '{key}' is not a PipelineData"
            )

    def test_empty_result_when_all_skipped(self):
        """
        Edge case: if all requested transforms return None,
        result should be an empty dict, not raise an error.
        """
        y = np.ones(50) * 5.0  # uniform — winsorize always returns None
        X = np.ones((50, 3))
        # Only request winsorize-based transforms
        result = build_transformed_stores(
            X, y, ["f1", "f2", "f3"], transforms=["wincount", "winlog", "winsqrt"]
        )
        assert result == {}


# ---------------------------------------------------------------------------
# transforms parameter
# ---------------------------------------------------------------------------


class TestTransformsParameter:
    def test_subset_of_transforms(self, X, clean_y):
        result = build_transformed_stores(
            X, clean_y, FEATURE_NAMES, transforms=["count", "log"]
        )
        assert set(result.keys()).issubset({"count", "log"})

    def test_count_only(self, X, clean_y):
        result = build_transformed_stores(
            X, clean_y, FEATURE_NAMES, transforms=["count"]
        )
        assert list(result.keys()) == ["count"]

    def test_all_transforms_when_none(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES, transforms=None)
        # All non-None transforms should be present
        assert "count" in result
        assert "log" in result
        assert "sqrt" in result

    def test_unknown_transform_raises_key_error(self, X, clean_y):
        with pytest.raises(KeyError, match="unknown_transform"):
            build_transformed_stores(
                X, clean_y, FEATURE_NAMES, transforms=["count", "unknown_transform"]
            )

    def test_empty_transforms_list_returns_empty_dict(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES, transforms=[])
        assert result == {}


# ---------------------------------------------------------------------------
# PipelineData integrity
# ---------------------------------------------------------------------------


class TestPipelineDataIntegrity:
    def test_each_store_passes_shape_validation(self, X, clean_y):
        """PipelineData.__post_init__ validates shape — should not raise."""
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for name, store in result.items():
            assert store.X.shape[1] == len(store.feature_names), (
                f"Shape mismatch in store '{name}'"
            )

    def test_y_length_matches_n_samples(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for store in result.values():
            assert len(store.y) == store.n_samples

    def test_feature_names_length_matches_n_features(self, X, clean_y):
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for store in result.values():
            assert len(store.feature_names) == store.n_features

    def test_store_to_frame_works(self, X, clean_y):
        """Each store should be able to recover a DataFrame."""
        result = build_transformed_stores(X, clean_y, FEATURE_NAMES)
        for name, store in result.items():
            df = store.to_frame()
            assert list(df.columns) == FEATURE_NAMES, (
                f"Column mismatch in store '{name}'"
            )
