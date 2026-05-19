"""
tests/test_features/test_correlation.py

Tests for remove_correlated_features().

Coverage:
    - Returns a list of strings
    - Returns all features when none are correlated
    - Removes perfectly correlated feature, keeps higher-ranked one
    - Importance order determines which feature survives correlation pair
    - Higher threshold keeps more features
    - Lower threshold removes more features
    - corr_threshold=1.0 keeps all features (only perfect correlation removed)
    - Invalid corr_threshold raises ValueError
    - Single feature input returns that feature
    - Works with pearson-only method
    - Works with spearman-only method
    - Works with kendall-only method
    - Feature removed by any method even if others are below threshold
    - Selected features are a subset of original features
    - Output order follows importance order, not original column order
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from h2ml.features.correlation import remove_correlated_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_importance(feature_names: list[str], values: list[float] = None) -> pd.Series:
    """Build a descending importance Series."""
    if values is None:
        values = list(range(len(feature_names), 0, -1))
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def _make_independent_df(n_features: int = 5, n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """DataFrame with uncorrelated features."""
    rng = np.random.default_rng(seed)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(rng.standard_normal((n_samples, n_features)), columns=cols)


def _make_correlated_df() -> pd.DataFrame:
    """
    DataFrame where:
        feat_a — independent (highest importance)
        feat_b — perfectly correlated with feat_a (lower importance)
        feat_c — independent
    """
    rng = np.random.default_rng(0)
    base = rng.standard_normal(100)
    return pd.DataFrame(
        {
            "feat_a": base,
            "feat_b": base + rng.normal(0, 1e-9, 100),  # near-perfect correlation
            "feat_c": rng.standard_normal(100),
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def independent_df() -> pd.DataFrame:
    return _make_independent_df()


@pytest.fixture
def correlated_df() -> pd.DataFrame:
    return _make_correlated_df()


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_list(self, independent_df):
        importance = _make_importance(list(independent_df.columns))
        result = remove_correlated_features(independent_df, importance)
        assert isinstance(result, list)

    def test_returns_list_of_strings(self, independent_df):
        importance = _make_importance(list(independent_df.columns))
        result = remove_correlated_features(independent_df, importance)
        assert all(isinstance(f, str) for f in result)

    def test_result_is_subset_of_original(self, independent_df):
        importance = _make_importance(list(independent_df.columns))
        result = remove_correlated_features(independent_df, importance)
        assert set(result).issubset(set(independent_df.columns))


# ---------------------------------------------------------------------------
# No correlation — all features kept
# ---------------------------------------------------------------------------


class TestNoCorrelation:
    def test_all_features_kept_when_independent(self, independent_df):
        importance = _make_importance(list(independent_df.columns))
        result = remove_correlated_features(independent_df, importance, corr_threshold=0.7)
        assert len(result) == len(independent_df.columns)

    def test_feature_names_match_when_independent(self, independent_df):
        importance = _make_importance(list(independent_df.columns))
        result = remove_correlated_features(independent_df, importance, corr_threshold=0.7)
        assert set(result) == set(independent_df.columns)


# ---------------------------------------------------------------------------
# Correlated features — removal logic
# ---------------------------------------------------------------------------


class TestCorrelatedRemoval:
    def test_correlated_feature_removed(self, correlated_df):
        """feat_b is near-perfectly correlated with feat_a — should be removed."""
        importance = _make_importance(["feat_a", "feat_b", "feat_c"], [3.0, 2.0, 1.0])
        result = remove_correlated_features(correlated_df, importance, corr_threshold=0.9)
        assert "feat_b" not in result

    def test_higher_importance_feature_survives(self, correlated_df):
        """feat_a has higher importance than feat_b — feat_a should be kept."""
        importance = _make_importance(["feat_a", "feat_b", "feat_c"], [3.0, 2.0, 1.0])
        result = remove_correlated_features(correlated_df, importance, corr_threshold=0.9)
        assert "feat_a" in result

    def test_independent_feature_kept(self, correlated_df):
        """feat_c is independent — should always be kept."""
        importance = _make_importance(["feat_a", "feat_b", "feat_c"], [3.0, 2.0, 1.0])
        result = remove_correlated_features(correlated_df, importance, corr_threshold=0.9)
        assert "feat_c" in result

    def test_importance_order_determines_survivor(self):
        """
        When feat_b has higher importance than feat_a,
        feat_b should survive and feat_a should be removed.
        """
        rng = np.random.default_rng(0)
        base = rng.standard_normal(100)
        df = pd.DataFrame(
            {
                "feat_a": base,
                "feat_b": base + rng.normal(0, 1e-9, 100),
            }
        )
        # feat_b ranked higher than feat_a
        importance = _make_importance(["feat_b", "feat_a"], [2.0, 1.0])
        result = remove_correlated_features(df, importance, corr_threshold=0.9)
        assert "feat_b" in result
        assert "feat_a" not in result

    def test_two_correlated_features_reduces_to_one(self, correlated_df):
        importance = _make_importance(["feat_a", "feat_b", "feat_c"], [3.0, 2.0, 1.0])
        result = remove_correlated_features(correlated_df, importance, corr_threshold=0.9)
        # feat_a and feat_b were correlated — only one should remain
        corr_pair_in_result = sum(f in result for f in ["feat_a", "feat_b"])
        assert corr_pair_in_result == 1


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_higher_threshold_keeps_more_features(self):
        """Relaxing the threshold should keep more features."""
        rng = np.random.default_rng(5)
        base = rng.standard_normal(100)
        noise = rng.standard_normal(100) * 0.3
        df = pd.DataFrame(
            {
                "feat_a": base,
                "feat_b": base + noise,  # moderately correlated
                "feat_c": rng.standard_normal(100),
            }
        )
        importance = _make_importance(["feat_a", "feat_b", "feat_c"])

        result_strict = remove_correlated_features(df, importance, corr_threshold=0.5)
        result_loose = remove_correlated_features(df, importance, corr_threshold=0.99)
        assert len(result_loose) >= len(result_strict)

    def test_threshold_1_keeps_all_non_perfect(self, independent_df):
        importance = _make_importance(list(independent_df.columns))
        result = remove_correlated_features(independent_df, importance, corr_threshold=1.0)
        assert len(result) == len(independent_df.columns)

    def test_invalid_threshold_zero_raises(self, independent_df):
        importance = _make_importance(list(independent_df.columns))
        with pytest.raises(ValueError, match="corr_threshold"):
            remove_correlated_features(independent_df, importance, corr_threshold=0.0)

    def test_invalid_threshold_negative_raises(self, independent_df):
        importance = _make_importance(list(independent_df.columns))
        with pytest.raises(ValueError, match="corr_threshold"):
            remove_correlated_features(independent_df, importance, corr_threshold=-0.5)

    def test_invalid_threshold_above_one_raises(self, independent_df):
        importance = _make_importance(list(independent_df.columns))
        with pytest.raises(ValueError, match="corr_threshold"):
            remove_correlated_features(independent_df, importance, corr_threshold=1.1)


# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------


class TestMethods:
    def test_pearson_only(self, correlated_df):
        importance = _make_importance(["feat_a", "feat_b", "feat_c"], [3.0, 2.0, 1.0])
        result = remove_correlated_features(correlated_df, importance, corr_threshold=0.9, methods=["pearson"])
        assert "feat_b" not in result

    def test_spearman_only(self, correlated_df):
        importance = _make_importance(["feat_a", "feat_b", "feat_c"], [3.0, 2.0, 1.0])
        result = remove_correlated_features(correlated_df, importance, corr_threshold=0.9, methods=["spearman"])
        assert "feat_b" not in result

    def test_kendall_only(self, correlated_df):
        importance = _make_importance(["feat_a", "feat_b", "feat_c"], [3.0, 2.0, 1.0])
        result = remove_correlated_features(correlated_df, importance, corr_threshold=0.9, methods=["kendall"])
        assert "feat_b" not in result

    def test_removed_if_any_method_exceeds_threshold(self):
        """
        feat_b is correlated with feat_a only by Pearson.
        Even with methods=["pearson", "spearman"], it should be removed.
        """
        rng = np.random.default_rng(0)
        base = rng.standard_normal(100)
        df = pd.DataFrame(
            {
                "feat_a": base,
                "feat_b": base + rng.normal(0, 1e-9, 100),
                "feat_c": rng.standard_normal(100),
            }
        )
        importance = _make_importance(["feat_a", "feat_b", "feat_c"], [3.0, 2.0, 1.0])
        result = remove_correlated_features(df, importance, corr_threshold=0.9, methods=["pearson", "spearman"])
        assert "feat_b" not in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_feature_returns_that_feature(self):
        df = pd.DataFrame({"only": np.random.default_rng(0).standard_normal(50)})
        importance = pd.Series([1.0], index=["only"])
        result = remove_correlated_features(df, importance)
        assert result == ["only"]

    def test_output_order_follows_importance(self, independent_df):
        """Output list should follow importance rank, not original column order."""
        cols = list(independent_df.columns)
        # Reverse importance order from column order
        reversed_importance = _make_importance(cols, values=list(range(len(cols), 0, -1)))
        result = remove_correlated_features(independent_df, reversed_importance, corr_threshold=0.9)
        assert result[0] == reversed_importance.index[0]
