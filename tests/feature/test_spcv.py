"""
tests/feature/test_spcv.py

Tests for SPCVSplitter.

Coverage:
    Interface:
        - get_n_splits() returns n_splits
        - split() yields exactly n_splits tuples of numpy arrays
        - split() accepts (X), (X, y), (X, y, groups)

    Index integrity:
        - Train and test indices are disjoint per fold
        - Union of all test indices covers the full dataset exactly once
        - All indices are within [0, n_samples)
        - len(train) + len(test) == n_samples per fold

    Stage 1 — AHC blocks:
        - block_id_ is set and has shape (n_samples,)
        - threshold=None triggers auto-heuristic (10th percentile)
        - Explicit threshold produces more/fewer blocks than auto
        - Threshold too large (single block) raises ValueError when n_blocks < n_splits
        - Tight spatial clusters stay within the same block

    Stage 2 — CE fold assignment:
        - n_splits folds are produced
        - Different random_states produce different fold assignments
        - All three clusterings (L, C, T) are consumed (smoke — no error)

    Reproducibility:
        - Two splitters with identical args produce identical folds
        - Repeated split() calls are identical

    Validation:
        - coords not 2-column raises ValueError
        - X row count != coords row count raises ValueError
        - y row count != coords row count raises ValueError
        - n_samples < n_splits raises ValueError
        - Row mismatch at split() time raises ValueError
"""
from __future__ import annotations

import numpy as np
import pytest

from h2ml.features.spatial_cv import SPCVSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 300


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def coords(rng) -> np.ndarray:
    return rng.uniform([36.0, -9.0], [44.0, 3.0], size=(N, 2))


@pytest.fixture
def X(rng) -> np.ndarray:
    return rng.standard_normal((N, 6))


@pytest.fixture
def y(rng) -> np.ndarray:
    return rng.standard_normal(N)


@pytest.fixture
def splitter(coords, X, y) -> SPCVSplitter:
    return SPCVSplitter(coords, X, y, n_splits=5)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TestInterface:

    def test_get_n_splits_matches_constructor(self, splitter):
        assert splitter.get_n_splits() == 5

    def test_get_n_splits_ignores_arguments(self, splitter, X, y):
        assert splitter.get_n_splits(X, y, groups=None) == 5

    def test_split_yields_n_splits_tuples(self, splitter, X):
        folds = list(splitter.split(X))
        assert len(folds) == 5

    def test_split_each_element_is_two_arrays(self, splitter, X, y):
        for train_idx, test_idx in splitter.split(X, y):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_split_accepts_X_only(self, splitter, X):
        assert len(list(splitter.split(X))) == 5

    def test_split_accepts_X_and_y(self, splitter, X, y):
        assert len(list(splitter.split(X, y))) == 5

    def test_split_accepts_X_y_groups(self, splitter, X, y):
        assert len(list(splitter.split(X, y, groups=None))) == 5


# ---------------------------------------------------------------------------
# Index integrity
# ---------------------------------------------------------------------------

class TestIndexIntegrity:

    def test_train_test_disjoint_per_fold(self, splitter, X, y):
        for train_idx, test_idx in splitter.split(X, y):
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_all_test_indices_cover_dataset_exactly_once(self, splitter, X, y):
        all_test = np.concatenate([te for _, te in splitter.split(X, y)])
        assert sorted(all_test) == list(range(N))

    def test_all_indices_within_valid_range(self, splitter, X, y):
        for train_idx, test_idx in splitter.split(X, y):
            assert train_idx.min() >= 0 and train_idx.max() < N
            assert test_idx.min() >= 0 and test_idx.max() < N

    def test_train_plus_test_equals_n_samples(self, splitter, X, y):
        for train_idx, test_idx in splitter.split(X, y):
            assert len(train_idx) + len(test_idx) == N


# ---------------------------------------------------------------------------
# Stage 1 — AHC blocks
# ---------------------------------------------------------------------------

class TestStage1Blocks:

    def test_block_id_attribute_set(self, splitter):
        assert hasattr(splitter, "block_id_")
        assert splitter.block_id_.shape == (N,)

    def test_block_id_values_are_non_negative_integers(self, splitter):
        assert splitter.block_id_.dtype in (np.int32, np.int64)
        assert splitter.block_id_.min() >= 0

    def test_threshold_none_uses_auto_heuristic(self, coords, X, y):
        # Should not raise — auto threshold based on 10th percentile
        sp = SPCVSplitter(coords, X, y, n_splits=5, threshold=None)
        assert sp.block_id_.shape == (N,)

    def test_small_threshold_produces_more_blocks(self, coords, X, y):
        sp_fine   = SPCVSplitter(coords, X, y, n_splits=5, threshold=0.5)
        sp_coarse = SPCVSplitter(coords, X, y, n_splits=5, threshold=5.0)
        assert np.unique(sp_fine.block_id_).size >= np.unique(sp_coarse.block_id_).size

    def test_threshold_too_large_raises(self, coords, X, y):
        # Threshold large enough to merge all samples into one block → fewer blocks than n_splits
        with pytest.raises(ValueError, match="block"):
            SPCVSplitter(coords, X, y, n_splits=5, threshold=1e9)

    def test_tight_clusters_share_block(self):
        """Samples within a tiny spatial cluster should all share the same block ID."""
        rng = np.random.default_rng(0)
        centers = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
        coords = np.vstack([center + rng.normal(0, 0.002, (25, 2)) for center in centers])
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)
        # threshold small enough to keep tight clusters intact
        sp = SPCVSplitter(coords, X, y, n_splits=4, threshold=0.05)
        for i in range(4):
            cluster_blocks = sp.block_id_[i * 25 : (i + 1) * 25]
            assert len(np.unique(cluster_blocks)) == 1, (
                f"Cluster {i} spans blocks: {np.unique(cluster_blocks)}"
            )


# ---------------------------------------------------------------------------
# Stage 2 — CE fold assignment
# ---------------------------------------------------------------------------

class TestStage2Folds:

    def test_produces_exactly_n_splits_fold_labels(self, splitter):
        fold_labels = np.unique(splitter._fold_of_sample)
        assert len(fold_labels) == 5

    def test_fold_labels_are_zero_based(self, splitter):
        assert set(splitter._fold_of_sample.tolist()) == set(range(5))

    def test_different_random_states_differ(self, coords, X, y):
        sp1 = SPCVSplitter(coords, X, y, n_splits=5, random_state=0)
        sp2 = SPCVSplitter(coords, X, y, n_splits=5, random_state=99)
        # With different seeds the fold assignments are very likely to differ
        assert not np.array_equal(sp1._fold_of_sample, sp2._fold_of_sample)

    def test_covariates_influence_fold_assignment(self, coords, y):
        """Two splitters with identical coords/y but different X should generally differ."""
        rng = np.random.default_rng(7)
        X1 = rng.standard_normal((N, 6))
        X2 = rng.standard_normal((N, 6)) * 100  # very different scale
        sp1 = SPCVSplitter(coords, X1, y, n_splits=5, random_state=42)
        sp2 = SPCVSplitter(coords, X2, y, n_splits=5, random_state=42)
        assert not np.array_equal(sp1._fold_of_sample, sp2._fold_of_sample)

    def test_target_influences_fold_assignment(self, coords, X):
        """Two splitters with identical coords/X but different y should generally differ."""
        rng = np.random.default_rng(8)
        y1 = rng.standard_normal(N)
        y2 = rng.standard_normal(N) * 100
        sp1 = SPCVSplitter(coords, X, y1, n_splits=5, random_state=42)
        sp2 = SPCVSplitter(coords, X, y2, n_splits=5, random_state=42)
        assert not np.array_equal(sp1._fold_of_sample, sp2._fold_of_sample)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_two_splitters_same_args_identical_folds(self, coords, X, y):
        sp1 = SPCVSplitter(coords, X, y, n_splits=5, random_state=42)
        sp2 = SPCVSplitter(coords, X, y, n_splits=5, random_state=42)
        for (tr1, te1), (tr2, te2) in zip(sp1.split(X, y), sp2.split(X, y)):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_repeated_split_calls_are_identical(self, splitter, X, y):
        folds_a = list(splitter.split(X, y))
        folds_b = list(splitter.split(X, y))
        for (tra, tea), (trb, teb) in zip(folds_a, folds_b):
            np.testing.assert_array_equal(tra, trb)
            np.testing.assert_array_equal(tea, teb)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_1d_coords_raises(self, X, y):
        with pytest.raises(ValueError, match="coords"):
            SPCVSplitter(np.zeros(N), X, y, n_splits=5)

    def test_3_column_coords_raises(self, X, y):
        with pytest.raises(ValueError, match="coords"):
            SPCVSplitter(np.zeros((N, 3)), X, y, n_splits=5)

    def test_X_row_mismatch_raises(self, coords, y):
        with pytest.raises(ValueError, match="X"):
            SPCVSplitter(coords, np.zeros((N + 10, 3)), y, n_splits=5)

    def test_y_row_mismatch_raises(self, coords, X):
        with pytest.raises(ValueError, match="y"):
            SPCVSplitter(coords, X, np.zeros(N + 10), n_splits=5)

    def test_n_samples_less_than_n_splits_raises(self):
        n = 3
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="n_samples"):
            SPCVSplitter(
                rng.standard_normal((n, 2)),
                rng.standard_normal((n, 4)),
                rng.standard_normal(n),
                n_splits=5,
            )

    def test_split_row_mismatch_raises(self, splitter):
        X_wrong = np.zeros((N + 10, 3))
        with pytest.raises(ValueError, match="rows"):
            list(splitter.split(X_wrong))

    def test_valid_construction_does_not_raise(self):
        rng = np.random.default_rng(0)
        n = 50
        SPCVSplitter(
            rng.standard_normal((n, 2)),
            rng.standard_normal((n, 4)),
            rng.standard_normal(n),
            n_splits=5,
        )
