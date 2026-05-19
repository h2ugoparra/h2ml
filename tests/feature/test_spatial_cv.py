"""
tests/feature/test_spatial_cv.py

Tests for SpatialBlockSplitter.

Coverage:
    Interface:
        - get_n_splits() returns n_splits
        - split() yields exactly n_splits folds
        - split() signature accepts (X), (X, y), (X, y, groups)

    Index integrity:
        - Train and test indices are disjoint per fold
        - Union of all test indices covers the full dataset exactly once
        - All indices are valid (within [0, n_samples))

    Spatial integrity:
        - Samples in the same tight cluster always land in the same fold
        - North/south datasets produce spatially separated test folds (2-split case)

    Reproducibility:
        - Two splitters with identical args produce identical fold assignments

    Quantile binning robustness:
        - Works when 90% of points are clustered in one corner

    Validation:
        - coords.ndim != 2 raises ValueError
        - coords.shape[1] != 2 raises ValueError
        - coords row count != X row count raises ValueError
        - n_splits * n_blocks_per_fold > n_samples raises ValueError
"""

from __future__ import annotations

import numpy as np
import pytest

from h2ml.features.spatial_cv import SpatialBlockSplitter


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

N = 200  # enough samples for default n_splits=5, n_blocks_per_fold=5


@pytest.fixture
def uniform_coords() -> np.ndarray:
    """200 uniformly distributed points on a unit square."""
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=(N, 2))


@pytest.fixture
def splitter(uniform_coords) -> SpatialBlockSplitter:
    return SpatialBlockSplitter(uniform_coords, n_splits=5, n_blocks_per_fold=5)


@pytest.fixture
def X_dummy() -> np.ndarray:
    """Dummy feature matrix — content irrelevant for spatial splitting."""
    return np.zeros((N, 3))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class TestInterface:
    def test_get_n_splits_matches_constructor(self, splitter):
        assert splitter.get_n_splits() == 5

    def test_get_n_splits_ignores_X_y_groups(self, splitter, X_dummy):
        y = np.zeros(N)
        assert splitter.get_n_splits(X_dummy, y, groups=None) == 5

    def test_split_yields_n_splits_tuples(self, splitter, X_dummy):
        folds = list(splitter.split(X_dummy))
        assert len(folds) == 5

    def test_split_each_element_is_two_arrays(self, splitter, X_dummy):
        for train_idx, test_idx in splitter.split(X_dummy):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_split_accepts_X_only(self, splitter, X_dummy):
        folds = list(splitter.split(X_dummy))
        assert len(folds) == 5

    def test_split_accepts_X_and_y(self, splitter, X_dummy):
        y = np.zeros(N)
        folds = list(splitter.split(X_dummy, y))
        assert len(folds) == 5

    def test_split_accepts_X_y_groups(self, splitter, X_dummy):
        y = np.zeros(N)
        folds = list(splitter.split(X_dummy, y, groups=None))
        assert len(folds) == 5


# ---------------------------------------------------------------------------
# Index integrity
# ---------------------------------------------------------------------------


class TestIndexIntegrity:
    def test_train_test_disjoint_per_fold(self, splitter, X_dummy):
        for train_idx, test_idx in splitter.split(X_dummy):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_all_test_indices_cover_dataset_exactly_once(self, splitter, X_dummy):
        all_test = np.concatenate([test for _, test in splitter.split(X_dummy)])
        assert sorted(all_test) == list(range(N))

    def test_all_indices_within_valid_range(self, splitter, X_dummy):
        for train_idx, test_idx in splitter.split(X_dummy):
            assert train_idx.min() >= 0
            assert train_idx.max() < N
            assert test_idx.min() >= 0
            assert test_idx.max() < N

    def test_train_plus_test_equals_n_samples(self, splitter, X_dummy):
        for train_idx, test_idx in splitter.split(X_dummy):
            assert len(train_idx) + len(test_idx) == N


# ---------------------------------------------------------------------------
# Spatial integrity
# ---------------------------------------------------------------------------


class TestSpatialIntegrity:
    def test_tight_cluster_members_share_fold(self):
        """Samples within the same tiny cluster must all be assigned to the same fold."""
        rng = np.random.default_rng(1)
        # 4 clusters, 20 samples each — clusters far apart so each maps to one block
        cluster_centers = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
        coords = np.vstack([center + rng.normal(0, 0.005, size=(20, 2)) for center in cluster_centers])  # shape (80, 2)
        splitter = SpatialBlockSplitter(coords, n_splits=4, n_blocks_per_fold=1)
        fold_assignments = splitter._fold_of_sample

        for cluster_idx in range(4):
            start = cluster_idx * 20
            cluster_folds = fold_assignments[start : start + 20]
            # All 20 members of this cluster should share exactly one fold value
            assert len(np.unique(cluster_folds)) == 1, (
                f"Cluster {cluster_idx} spans multiple folds: {np.unique(cluster_folds)}"
            )

    def test_north_south_separation_2fold(self):
        """
        With n_splits=2 and a dataset split into a clear north and south half,
        each test fold should be entirely north or entirely south.
        """
        n = 100
        # North half: lat in [0.6, 1.0]; south half: lat in [0.0, 0.4]
        north = np.column_stack(
            [
                np.linspace(0.6, 1.0, n),
                np.zeros(n),
            ]
        )
        south = np.column_stack(
            [
                np.linspace(0.0, 0.4, n),
                np.zeros(n),
            ]
        )
        coords = np.vstack([north, south])  # (200, 2)
        X_dummy = np.zeros((200, 1))

        splitter = SpatialBlockSplitter(coords, n_splits=2, n_blocks_per_fold=1)

        for train_idx, test_idx in splitter.split(X_dummy):
            # Test samples should be either all from north (idx < 100) or all south (idx >= 100)
            north_in_test = np.sum(test_idx < n)
            south_in_test = np.sum(test_idx >= n)
            # One group should dominate (at least 90% from one hemisphere)
            total = len(test_idx)
            assert max(north_in_test, south_in_test) / total >= 0.9, (
                f"Test fold is not spatially homogeneous: {north_in_test} north, {south_in_test} south out of {total}"
            )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_two_splitters_same_args_identical_folds(self, uniform_coords, X_dummy):
        s1 = SpatialBlockSplitter(uniform_coords, n_splits=5, n_blocks_per_fold=5)
        s2 = SpatialBlockSplitter(uniform_coords, n_splits=5, n_blocks_per_fold=5)
        for (tr1, te1), (tr2, te2) in zip(s1.split(X_dummy), s2.split(X_dummy)):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_repeated_split_calls_identical(self, splitter, X_dummy):
        folds_a = list(splitter.split(X_dummy))
        folds_b = list(splitter.split(X_dummy))
        for (tr_a, te_a), (tr_b, te_b) in zip(folds_a, folds_b):
            np.testing.assert_array_equal(tr_a, tr_b)
            np.testing.assert_array_equal(te_a, te_b)


# ---------------------------------------------------------------------------
# Quantile binning robustness
# ---------------------------------------------------------------------------


class TestQuantileBinningRobustness:
    def test_skewed_distribution_completes_without_error(self):
        """90% of points in one corner should not crash the splitter."""
        rng = np.random.default_rng(7)
        n = 200
        # 180 points near (0, 0), 20 points scattered elsewhere
        coords = np.vstack(
            [
                rng.uniform(0.0, 0.05, size=(180, 2)),
                rng.uniform(0.0, 1.0, size=(20, 2)),
            ]
        )
        splitter = SpatialBlockSplitter(coords, n_splits=4, n_blocks_per_fold=3)
        X_dummy = np.zeros((n, 1))
        all_test = np.concatenate([te for _, te in splitter.split(X_dummy)])
        assert sorted(all_test) == list(range(n))

    def test_constant_axis_does_not_crash(self):
        """All points on the same latitude line — axis 0 is constant."""
        rng = np.random.default_rng(3)
        n = 100
        coords = np.column_stack(
            [
                np.zeros(n),  # constant lat
                rng.uniform(0, 1, n),  # varying lon
            ]
        )
        splitter = SpatialBlockSplitter(coords, n_splits=2, n_blocks_per_fold=2)
        X_dummy = np.zeros((n, 1))
        all_test = np.concatenate([te for _, te in splitter.split(X_dummy)])
        assert sorted(all_test) == list(range(n))


# ---------------------------------------------------------------------------
# n_blocks_per_fold effect
# ---------------------------------------------------------------------------


class TestNBlocksPerFold:
    def test_different_n_blocks_produces_different_assignments(self, uniform_coords):
        s_fine = SpatialBlockSplitter(uniform_coords, n_splits=5, n_blocks_per_fold=10)
        s_coarse = SpatialBlockSplitter(uniform_coords, n_splits=5, n_blocks_per_fold=2)
        # The fold assignments will generally differ with different granularity
        assert not np.array_equal(s_fine._fold_of_sample, s_coarse._fold_of_sample)

    def test_n_blocks_per_fold_1_valid(self, uniform_coords, X_dummy):
        splitter = SpatialBlockSplitter(uniform_coords, n_splits=5, n_blocks_per_fold=1)
        folds = list(splitter.split(X_dummy))
        assert len(folds) == 5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_1d_coords_raises(self):
        with pytest.raises(ValueError, match="shape"):
            SpatialBlockSplitter(np.zeros(100), n_splits=2)

    def test_3d_coords_raises(self):
        with pytest.raises(ValueError, match="shape"):
            SpatialBlockSplitter(np.zeros((100, 3)), n_splits=2)

    def test_too_many_blocks_raises(self):
        coords = np.zeros((10, 2))
        with pytest.raises(ValueError, match="n_splits"):
            SpatialBlockSplitter(coords, n_splits=5, n_blocks_per_fold=5)  # 25 > 10

    def test_coords_row_mismatch_at_split_time_raises(self, uniform_coords):
        splitter = SpatialBlockSplitter(uniform_coords, n_splits=5, n_blocks_per_fold=5)
        X_wrong = np.zeros((50, 3))  # 50 rows vs 200 in coords
        with pytest.raises(ValueError, match="rows"):
            list(splitter.split(X_wrong))

    def test_valid_construction_does_not_raise(self):
        coords = np.zeros((100, 2))
        SpatialBlockSplitter(coords, n_splits=2, n_blocks_per_fold=2)  # 4 <= 100

    def test_n_splits_boundary_exact_fit(self):
        """n_splits * n_blocks_per_fold == n_samples should succeed."""
        n = 25
        coords = np.zeros((n, 2))
        coords[:, 0] = np.arange(n)
        # 5 * 5 == 25 — exact boundary, should not raise
        SpatialBlockSplitter(coords, n_splits=5, n_blocks_per_fold=5)
