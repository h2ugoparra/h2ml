"""
tests/test_features/test_store.py

Tests for PipelineData.

Coverage:
    Construction:
        - Valid construction succeeds
        - Mismatched X columns and feature_names raises ValueError
        - n_samples and n_features properties are correct

    Conversions:
        - to_frame() returns DataFrame with correct columns
        - to_frame() preserves values
        - from_frame() builds correct PipelineData
        - from_frame() round-trips back to identical DataFrame

    select():
        - Returns PipelineData with only selected features
        - Selected feature order follows input list, not original order
        - X columns match selected feature names
        - y is preserved unchanged
        - Missing feature name raises ValueError
        - Selecting all features returns full store

    __repr__:
        - Contains n_samples and n_features
        - Truncates long feature lists with ellipsis
        - Short feature lists shown in full
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from h2ml.features.feature_store import PipelineData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def feature_names() -> list[str]:
    return ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]


@pytest.fixture
def store(feature_names) -> PipelineData:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = rng.integers(0, 2, size=100)
    return PipelineData(X=X, feature_names=feature_names, y=y)


@pytest.fixture
def small_store() -> PipelineData:
    """Store with 2 features — for repr truncation tests."""
    X = np.ones((10, 2))
    y = np.zeros(10)
    return PipelineData(X=X, feature_names=["f1", "f2"], y=y)


@pytest.fixture
def correlated_store() -> PipelineData:
    """
    Store where feat_b is perfectly correlated with feat_a.
    Used to verify select() correctly isolates columns.
    """
    rng = np.random.default_rng(0)
    base = rng.standard_normal(50)
    X = np.column_stack(
        [
            base,  # feat_a
            base * 1.5,  # feat_b — perfectly correlated with feat_a
            rng.standard_normal(50),  # feat_c — independent
        ]
    )
    y = rng.integers(0, 2, size=50)
    return PipelineData(X=X, feature_names=["feat_a", "feat_b", "feat_c"], y=y)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_valid_construction(self, store):
        assert store.n_samples == 100
        assert store.n_features == 5

    def test_mismatched_columns_raises(self):
        X = np.ones((10, 3))
        with pytest.raises(ValueError, match="3 columns"):
            PipelineData(X=X, feature_names=["a", "b"], y=np.zeros(10))

    def test_mismatched_y_rows_raises(self):
        X = np.ones((10, 2))
        with pytest.raises(ValueError, match="y has 8 rows"):
            PipelineData(X=X, feature_names=["a", "b"], y=np.zeros(8))

    def test_mismatched_y_true_rows_raises(self):
        X = np.ones((10, 2))
        with pytest.raises(ValueError, match="y_true has 7 rows"):
            PipelineData(
                X=X, feature_names=["a", "b"], y=np.zeros(10), y_true=np.zeros(7)
            )

    def test_mismatched_coords_rows_raises(self):
        X = np.ones((10, 2))
        with pytest.raises(ValueError, match="coords has 6 rows"):
            PipelineData(
                X=X, feature_names=["a", "b"], y=np.zeros(10), coords=np.zeros((6, 2))
            )

    def test_n_samples_property(self, store):
        assert store.n_samples == store.X.shape[0]

    def test_n_features_property(self, store):
        assert store.n_features == store.X.shape[1]

    def test_feature_names_stored(self, store, feature_names):
        assert store.feature_names == feature_names


# ---------------------------------------------------------------------------
# to_frame()
# ---------------------------------------------------------------------------


class TestToFrame:
    def test_returns_dataframe(self, store):
        df = store.to_frame()
        assert isinstance(df, pd.DataFrame)

    def test_columns_match_feature_names(self, store, feature_names):
        df = store.to_frame()
        assert list(df.columns) == feature_names

    def test_shape_matches_X(self, store):
        df = store.to_frame()
        assert df.shape == store.X.shape

    def test_values_preserved(self, store):
        df = store.to_frame()
        np.testing.assert_array_equal(df.to_numpy(), store.X)


# ---------------------------------------------------------------------------
# from_frame()
# ---------------------------------------------------------------------------


class TestFromFrame:
    def test_builds_from_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        y = np.array([0, 1])
        store = PipelineData.from_frame(df, y)
        assert store.feature_names == ["a", "b"]
        assert store.n_samples == 2
        assert store.n_features == 2

    def test_round_trip_preserves_values(self):
        df = pd.DataFrame(
            np.random.default_rng(7).standard_normal((20, 4)),
            columns=["w", "x", "y", "z"],
        )
        y = np.zeros(20)
        store = PipelineData.from_frame(df, y)
        df_rt = store.to_frame()
        pd.testing.assert_frame_equal(df.reset_index(drop=True), df_rt)

    def test_y_stored_correctly(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        y = np.array([1, 0])
        store = PipelineData.from_frame(df, y)
        np.testing.assert_array_equal(store.y, y)


# ---------------------------------------------------------------------------
# select()
# ---------------------------------------------------------------------------


class TestSelect:
    def test_returns_feature_store(self, store):
        result = store.select(["feat_a", "feat_c"])
        assert isinstance(result, PipelineData)

    def test_selected_feature_names(self, store):
        result = store.select(["feat_a", "feat_c"])
        assert result.feature_names == ["feat_a", "feat_c"]

    def test_selected_n_features(self, store):
        result = store.select(["feat_a", "feat_b", "feat_c"])
        assert result.n_features == 3

    def test_selected_n_samples_unchanged(self, store):
        result = store.select(["feat_a"])
        assert result.n_samples == store.n_samples

    def test_x_columns_match_selection(self, store):
        result = store.select(["feat_b", "feat_d"])
        df = result.to_frame()
        assert list(df.columns) == ["feat_b", "feat_d"]

    def test_column_values_correct_after_select(self, store):
        """Values in selected columns must match original X columns."""
        result = store.select(["feat_c"])
        original_col_idx = store.feature_names.index("feat_c")
        np.testing.assert_array_equal(
            result.X[:, 0],
            store.X[:, original_col_idx],
        )

    def test_order_follows_input_list(self, store):
        """select(["feat_e", "feat_a"]) should return feat_e first."""
        result = store.select(["feat_e", "feat_a"])
        assert result.feature_names[0] == "feat_e"
        assert result.feature_names[1] == "feat_a"

    def test_y_preserved_after_select(self, store):
        result = store.select(["feat_a"])
        np.testing.assert_array_equal(result.y, store.y)

    def test_missing_feature_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.select(["feat_a", "nonexistent_feature"])

    def test_select_all_features_returns_full_store(self, store, feature_names):
        result = store.select(feature_names)
        assert result.feature_names == feature_names
        assert result.n_features == store.n_features

    def test_select_single_feature(self, store):
        result = store.select(["feat_b"])
        assert result.n_features == 1
        assert result.feature_names == ["feat_b"]

    def test_correlated_columns_isolated(self, correlated_store):
        """Selecting feat_a and feat_c should not include feat_b values."""
        result = correlated_store.select(["feat_a", "feat_c"])
        assert "feat_b" not in result.feature_names
        assert result.n_features == 2


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_n_samples(self, store):
        assert "n_samples=100" in repr(store)

    def test_repr_contains_n_features(self, store):
        assert "n_features=5" in repr(store)

    def test_repr_truncates_long_feature_list(self, store):
        assert "..." in repr(store)

    def test_repr_no_truncation_for_short_list(self, small_store):
        assert "..." not in repr(small_store)

    def test_repr_shows_first_three_features(self, store):
        r = repr(store)
        assert "feat_a" in r
        assert "feat_b" in r
        assert "feat_c" in r


# ---------------------------------------------------------------------------
# coords field
# ---------------------------------------------------------------------------


class TestCoords:
    def _make_coords(self, n: int = 100) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.uniform(0, 1, size=(n, 2))

    def test_coords_stored_correctly(self, store):
        coords = self._make_coords(store.n_samples)
        s = PipelineData(
            X=store.X, feature_names=store.feature_names, y=store.y, coords=coords
        )
        np.testing.assert_array_equal(s.coords, coords)

    def test_coords_none_by_default(self, store):
        assert store.coords is None

    def test_coords_wrong_ndim_raises(self, store):
        with pytest.raises(ValueError, match="shape"):
            PipelineData(
                X=store.X,
                feature_names=store.feature_names,
                y=store.y,
                coords=np.zeros(store.n_samples),  # 1-D
            )

    def test_coords_wrong_shape_raises(self, store):
        with pytest.raises(ValueError, match="shape"):
            PipelineData(
                X=store.X,
                feature_names=store.feature_names,
                y=store.y,
                coords=np.zeros((store.n_samples, 3)),  # (n, 3) not (n, 2)
            )

    def test_coords_row_mismatch_raises(self, store):
        with pytest.raises(ValueError, match="rows"):
            PipelineData(
                X=store.X,
                feature_names=store.feature_names,
                y=store.y,
                coords=np.zeros((store.n_samples + 1, 2)),
            )

    def test_select_preserves_coords(self, store):
        coords = self._make_coords(store.n_samples)
        s = PipelineData(
            X=store.X, feature_names=store.feature_names, y=store.y, coords=coords
        )
        reduced = s.select(["feat_a", "feat_c"])
        np.testing.assert_array_equal(reduced.coords, coords)

    def test_select_preserves_coords_none(self, store):
        reduced = store.select(["feat_a", "feat_b"])
        assert reduced.coords is None

    def test_from_frame_accepts_coords(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        y = np.array([0, 1])
        coords = np.array([[0.1, 0.2], [0.3, 0.4]])
        s = PipelineData.from_frame(df, y, coords=coords)
        np.testing.assert_array_equal(s.coords, coords)

    def test_from_frame_coords_none_by_default(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        s = PipelineData.from_frame(df, np.array([0, 1]))
        assert s.coords is None
