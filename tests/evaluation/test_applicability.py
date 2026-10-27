"""
tests/evaluation/test_applicability.py

Tests for the area of applicability (Meyer & Pebesma 2021): dissimilarity index,
cross-validated training DI, upper-whisker threshold and feature weighting.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from h2ml.evaluation.applicability import build_applicability_domain

NAMES = ["a", "b", "c"]


def _train(n: int = 90, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, len(NAMES))), np.arange(n) % 3


class TestApplicabilityDomain:
    def test_training_rows_are_inside_with_zero_di(self):
        X, folds = _train()
        aoa = build_applicability_domain(X, NAMES, folds)
        np.testing.assert_allclose(aoa.dissimilarity(X), 0.0, atol=1e-12)
        assert aoa.inside(X).all()

    def test_far_point_is_outside(self):
        X, folds = _train()
        aoa = build_applicability_domain(X, NAMES, folds)
        far = np.array([[50.0, -50.0, 50.0]])
        assert aoa.dissimilarity(far)[0] > aoa.threshold
        assert not aoa.inside(far)[0]

    def test_mean_distance_and_di_definition(self):
        X, folds = _train()
        aoa = build_applicability_domain(X, NAMES, folds)
        Z = (X - X.mean(axis=0)) / X.std(axis=0) / 3  # equal weights normalised to 1/3
        assert aoa.mean_distance == pytest.approx(pdist(Z).mean())
        new = np.array([[0.3, -0.2, 1.1]])
        z_new = (new - X.mean(axis=0)) / X.std(axis=0) / 3
        expected = np.min(np.linalg.norm(Z - z_new, axis=1)) / aoa.mean_distance
        assert aoa.dissimilarity(new)[0] == pytest.approx(expected)

    def test_train_di_uses_rows_outside_own_fold_and_whisker_threshold(self):
        X, folds = _train()
        aoa = build_applicability_domain(X, NAMES, folds)
        Z = (X - X.mean(axis=0)) / X.std(axis=0) / 3
        expected = []
        for k in range(3):
            test, train = Z[folds == k], Z[folds != k]
            d = np.linalg.norm(test[:, None, :] - train[None, :, :], axis=2).min(axis=1)
            expected.append(d / aoa.mean_distance)
        expected = np.concatenate(expected)
        np.testing.assert_allclose(aoa.train_di, expected)
        q1, q3 = np.percentile(expected, [25, 75])
        assert aoa.threshold == pytest.approx(expected[expected <= q3 + 1.5 * (q3 - q1)].max())

    def test_zero_weight_feature_is_ignored(self):
        X, folds = _train()
        aoa = build_applicability_domain(X, NAMES, folds, weights=np.array([1.0, 2.0, 0.0]))
        np.testing.assert_allclose(aoa.weights, [1 / 3, 2 / 3, 0.0])
        point = X[:1].copy()
        moved = point.copy()
        moved[0, 2] += 100.0
        assert aoa.dissimilarity(moved)[0] == pytest.approx(aoa.dissimilarity(point)[0])

    def test_all_zero_weights_fall_back_to_equal(self):
        X, folds = _train()
        aoa = build_applicability_domain(X, NAMES, folds, weights=np.zeros(3))
        np.testing.assert_allclose(aoa.weights, 1 / 3)

    def test_rows_never_held_out_are_neighbours_only(self):
        X, folds = _train()
        folds = folds.copy()
        folds[:10] = -1
        aoa = build_applicability_domain(X, NAMES, folds)
        assert aoa.train_di.size == 80
        assert aoa.inside(X[:10]).all()  # still training rows → DI 0

    def test_single_fold_raises(self):
        X, _ = _train()
        with pytest.raises(ValueError, match="two CV folds"):
            build_applicability_domain(X, NAMES, np.zeros(len(X), dtype=int))

    def test_shape_mismatch_raises(self):
        X, folds = _train()
        with pytest.raises(ValueError, match="fold ids"):
            build_applicability_domain(X, NAMES, folds[:-1])
