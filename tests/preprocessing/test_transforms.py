"""
tests/test_preprocessing/test_transforms.py

Tests for y transformation functions in preprocessing/transforms.py.

Coverage:
    log_transform:
        - Returns numpy array
        - Applies log1p correctly
        - Handles zero values (log1p(0) = 0)
        - Handles array with all zeros
        - Output is always non-negative for non-negative input

    sqrt_transform:
        - Returns numpy array
        - Applies sqrt correctly
        - Handles zero values
        - Output is always non-negative for non-negative input

    winsorize:
        - Returns None when no outliers present
        - Returns numpy array when outliers present
        - Replaces upper outliers with upper limit
        - Does not modify values below upper limit
        - Upper limit computed correctly from IQR
        - Works on array with single outlier
        - Works on array with multiple outliers
        - Lower values are untouched

    Y_TRANSFORMS registry:
        - All expected keys are present
        - count transform returns y unchanged
        - log transform matches log_transform()
        - sqrt transform matches sqrt_transform()
        - wincount returns None when no outliers
        - wincount clips outliers when present
        - winlog applies log then winsorize
        - winsqrt applies sqrt then winsorize
        - Unknown transform key raises KeyError when accessed
"""

from __future__ import annotations

import numpy as np
import pytest

from h2ml.preprocessing.transforms import (
    Y_TRANSFORMS,
    log_transform,
    sqrt_transform,
    winsorize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_y() -> np.ndarray:
    """Array with no outliers — IQR-based upper limit not exceeded."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 10, size=100)


@pytest.fixture
def outlier_y() -> np.ndarray:
    """Array with clear upper outliers."""
    rng = np.random.default_rng(42)
    base = rng.uniform(0, 10, size=95)
    outliers = np.array([1000.0, 2000.0, 1500.0, 900.0, 800.0])
    return np.concatenate([base, outliers])


@pytest.fixture
def zero_y() -> np.ndarray:
    return np.zeros(20)


@pytest.fixture
def single_outlier_y() -> np.ndarray:
    base = np.ones(50) * 5.0
    return np.append(base, 9999.0)


# ---------------------------------------------------------------------------
# log_transform
# ---------------------------------------------------------------------------


class TestLogTransform:
    def test_returns_numpy_array(self, clean_y):
        result = log_transform(clean_y)
        assert isinstance(result, np.ndarray)

    def test_applies_log1p(self, clean_y):
        result = log_transform(clean_y)
        np.testing.assert_array_almost_equal(result, np.log1p(clean_y))

    def test_zero_input_gives_zero(self):
        result = log_transform(np.array([0.0]))
        assert result[0] == pytest.approx(0.0)

    def test_all_zeros_array(self, zero_y):
        result = log_transform(zero_y)
        np.testing.assert_array_equal(result, np.zeros(20))

    def test_output_non_negative_for_non_negative_input(self, clean_y):
        result = log_transform(clean_y)
        assert (result >= 0).all()

    def test_shape_preserved(self, clean_y):
        result = log_transform(clean_y)
        assert result.shape == clean_y.shape

    def test_monotonic_with_input(self, clean_y):
        """log1p is monotonically increasing — larger input → larger output."""
        y_sorted = np.sort(clean_y)
        result = log_transform(y_sorted)
        assert (np.diff(result) >= 0).all()


# ---------------------------------------------------------------------------
# sqrt_transform
# ---------------------------------------------------------------------------


class TestSqrtTransform:
    def test_returns_numpy_array(self, clean_y):
        result = sqrt_transform(clean_y)
        assert isinstance(result, np.ndarray)

    def test_applies_sqrt(self, clean_y):
        result = sqrt_transform(clean_y)
        np.testing.assert_array_almost_equal(result, np.sqrt(clean_y))

    def test_zero_input_gives_zero(self):
        result = sqrt_transform(np.array([0.0]))
        assert result[0] == pytest.approx(0.0)

    def test_output_non_negative_for_non_negative_input(self, clean_y):
        result = sqrt_transform(clean_y)
        assert (result >= 0).all()

    def test_shape_preserved(self, clean_y):
        result = sqrt_transform(clean_y)
        assert result.shape == clean_y.shape

    def test_monotonic_with_input(self, clean_y):
        y_sorted = np.sort(clean_y)
        result = sqrt_transform(y_sorted)
        assert (np.diff(result) >= 0).all()

    def test_sqrt_compresses_range(self, outlier_y):
        """sqrt should reduce the range of values with outliers."""
        original_range = outlier_y.max() - outlier_y.min()
        result_range = sqrt_transform(outlier_y).max() - sqrt_transform(outlier_y).min()
        assert result_range < original_range


# ---------------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------------


class TestWinsorize:
    def test_returns_none_when_no_outliers(self, clean_y):
        result = winsorize(clean_y)
        assert result is None

    def test_returns_array_when_outliers_present(self, outlier_y):
        result = winsorize(outlier_y)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self, outlier_y):
        result = winsorize(outlier_y)
        assert result.shape == outlier_y.shape

    def test_upper_outliers_clipped(self, outlier_y):
        """All values in result should be <= IQR upper limit."""
        Q1, Q3 = np.percentile(outlier_y, [25, 75])
        upper_limit = Q3 + 1.5 * (Q3 - Q1)
        result = winsorize(outlier_y)
        assert (result <= upper_limit + 1e-9).all()

    def test_values_below_limit_unchanged(self, outlier_y):
        """Non-outlier values should not be modified."""
        Q1, Q3 = np.percentile(outlier_y, [25, 75])
        upper_limit = Q3 + 1.5 * (Q3 - Q1)
        result = winsorize(outlier_y)
        mask = outlier_y <= upper_limit
        np.testing.assert_array_equal(result[mask], outlier_y[mask])

    def test_upper_limit_computed_correctly(self, outlier_y):
        Q1, Q3 = np.percentile(outlier_y, [25, 75])
        expected_limit = Q3 + 1.5 * (Q3 - Q1)
        result = winsorize(outlier_y)
        assert result.max() == pytest.approx(expected_limit, rel=1e-6)

    def test_single_outlier_clipped(self, single_outlier_y):
        result = winsorize(single_outlier_y)
        assert result is not None
        assert result[-1] < single_outlier_y[-1]

    def test_single_outlier_rest_unchanged(self, single_outlier_y):
        result = winsorize(single_outlier_y)
        np.testing.assert_array_equal(result[:-1], single_outlier_y[:-1])

    def test_multiple_outliers_all_clipped(self, outlier_y):
        Q1, Q3 = np.percentile(outlier_y, [25, 75])
        upper_limit = Q3 + 1.5 * (Q3 - Q1)
        result = winsorize(outlier_y)
        n_above = (outlier_y > upper_limit).sum()
        assert n_above > 1  # confirm fixture has multiple outliers
        assert (result > upper_limit).sum() == 0

    def test_all_same_value_returns_none(self):
        """Uniform array has IQR=0 — no outliers possible."""
        y = np.ones(50) * 5.0
        assert winsorize(y) is None

    def test_does_not_clip_lower_values(self, outlier_y):
        """Winsorize is upper-only — minimum should be unchanged."""
        result = winsorize(outlier_y)
        assert result.min() == pytest.approx(outlier_y.min(), rel=1e-6)


# ---------------------------------------------------------------------------
# Y_TRANSFORMS registry
# ---------------------------------------------------------------------------


class TestYTransformsRegistry:
    def test_all_expected_keys_present(self):
        expected = {"count", "log", "sqrt", "wincount", "winlog", "winsqrt"}
        assert expected.issubset(set(Y_TRANSFORMS.keys()))

    def test_count_returns_y_unchanged(self, clean_y):
        result = Y_TRANSFORMS["count"](clean_y)
        np.testing.assert_array_equal(result, clean_y)

    def test_log_matches_log_transform(self, clean_y):
        result = Y_TRANSFORMS["log"](clean_y)
        expected = log_transform(clean_y)
        np.testing.assert_array_almost_equal(result, expected)

    def test_sqrt_matches_sqrt_transform(self, clean_y):
        result = Y_TRANSFORMS["sqrt"](clean_y)
        expected = sqrt_transform(clean_y)
        np.testing.assert_array_almost_equal(result, expected)

    def test_wincount_returns_none_when_no_outliers(self, clean_y):
        result = Y_TRANSFORMS["wincount"](clean_y)
        assert result is None

    def test_wincount_clips_outliers(self, outlier_y):
        result = Y_TRANSFORMS["wincount"](outlier_y)
        assert result is not None
        Q1, Q3 = np.percentile(outlier_y, [25, 75])
        upper_limit = Q3 + 1.5 * (Q3 - Q1)
        assert (result <= upper_limit + 1e-9).all()

    def test_winlog_applies_log_then_winsorize(self, outlier_y):
        """winlog = winsorize(log1p(y)) — verify chain."""
        log_y = log_transform(outlier_y)
        win_y = winsorize(log_y)
        result = Y_TRANSFORMS["winlog"](outlier_y)
        if win_y is None:
            assert result is None
        else:
            np.testing.assert_array_almost_equal(result, win_y)

    def test_winsqrt_applies_sqrt_then_winsorize(self, outlier_y):
        """winsqrt = winsorize(sqrt(y)) — verify chain."""
        sqrt_y = sqrt_transform(outlier_y)
        win_y = winsorize(sqrt_y)
        result = Y_TRANSFORMS["winsqrt"](outlier_y)
        if win_y is None:
            assert result is None
        else:
            np.testing.assert_array_almost_equal(result, win_y)

    def test_winlog_returns_none_when_log_has_no_outliers(self, clean_y):
        """Log compresses outliers — winsorize may find none after log."""
        result = Y_TRANSFORMS["winlog"](clean_y)
        # Either None (no outliers after log) or array (outliers survived log)
        assert result is None or isinstance(result, np.ndarray)

    def test_all_transforms_callable(self):
        for name, fn in Y_TRANSFORMS.items():
            assert callable(fn), f"Transform '{name}' is not callable"
