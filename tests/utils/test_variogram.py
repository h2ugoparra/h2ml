"""
tests/utils/test_variogram.py

Tests for autocorrelation_range().

Coverage:
    - Returns a VariogramResult with finite, positive parameters
    - Large inputs are subsampled to max_points (bounds the O(n²) matrix)
    - The subsample is deterministic given random_state
    - The range is capped at the lags and status says how far to trust it
    - Haversine distances come out in km
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from h2ml.utils.variogram import EARTH_RADIUS_KM, VariogramResult, autocorrelation_range


def _structured_field(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates in [0, 10]² with a spatially structured value (gradient + noise)."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    values = coords[:, 0] + 0.5 * coords[:, 1] + rng.normal(0, 0.1, n)
    return coords, values


def test_returns_variogram_result():
    coords, values = _structured_field(200)
    result = autocorrelation_range(coords, values)
    assert isinstance(result, VariogramResult)
    assert np.isfinite(result.practical_range)
    assert result.practical_range > 0


def test_large_input_subsampled_runs():
    """n > max_points must not build the full n×n matrix — the subsample path runs."""
    coords, values = _structured_field(5_000)
    result = autocorrelation_range(coords, values, max_points=500)
    # n_pairs come from the subsample (500 points → 500*499/2 pairs at most), not n.
    assert result.n_pairs.sum() <= 500 * 499 // 2
    assert np.isfinite(result.practical_range)


def test_subsample_is_deterministic():
    coords, values = _structured_field(2_000)
    r1 = autocorrelation_range(coords, values, max_points=300, random_state=7)
    r2 = autocorrelation_range(coords, values, max_points=300, random_state=7)
    assert r1.practical_range == r2.practical_range
    assert np.array_equal(r1.n_pairs, r2.n_pairs)


def _exponential_field(
    seed: int, practical_range: float, n: int = 400, noise: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian field on [0, 1000]² with exponential correlation and a noise share of its variance."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1000, size=(n, 2))
    cov = np.exp(-3 * cdist(coords, coords) / practical_range) + 1e-8 * np.eye(n)
    field = np.linalg.cholesky(cov) @ rng.normal(size=n)
    return coords, np.sqrt(1 - noise) * field + np.sqrt(noise) * rng.normal(size=n)


class TestStatus:
    def test_recovers_a_resolved_range(self):
        coords, values = _exponential_field(0, 200)
        result = autocorrelation_range(coords, values)
        assert result.status == "ok"
        assert 100 < result.practical_range < 400

    def test_range_longer_than_the_lags_is_flagged_and_capped(self):
        coords, values = _structured_field(300)
        result = autocorrelation_range(coords, values, max_dist=5.0)
        assert result.status == "at_bound"
        assert result.practical_range == pytest.approx(5.0, rel=0.02)

    def test_gaussian_noise_has_no_structure(self):
        rng = np.random.default_rng(1)
        coords = rng.uniform(0, 1000, size=(400, 2))
        assert autocorrelation_range(coords, rng.normal(size=400)).status == "no_structure"

    def test_mostly_zero_noise_has_no_structure(self):
        statuses = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            coords = rng.uniform(0, 1000, size=(400, 2))
            values = np.where(rng.random(400) < 0.9, 0.0, rng.lognormal(1, 1.2, 400))
            statuses.append(autocorrelation_range(coords, values).status)
        assert statuses.count("no_structure") >= 8

    def test_clustered_catches_are_detected(self):
        statuses = []
        for seed in range(10):
            coords, field = _exponential_field(seed, 200, noise=0.0)
            rng = np.random.default_rng(seed)
            values = np.where(field > np.quantile(field, 0.7), rng.lognormal(1, 1.2, 400), 0.0)
            statuses.append(autocorrelation_range(coords, values).status)
        assert statuses.count("no_structure") <= 3

    def test_constant_values_have_no_structure(self):
        coords = np.random.default_rng(0).uniform(0, 10, size=(100, 2))
        assert autocorrelation_range(coords, np.zeros(100)).status == "no_structure"

    def test_status_is_deterministic(self):
        coords, values = _exponential_field(2, 200)
        assert autocorrelation_range(coords, values).status == autocorrelation_range(coords, values).status

    def test_saved_before_status_reads_unknown(self):
        result = autocorrelation_range(*_structured_field(100))
        del result.__dict__["status"]  # as unpickled from an older save
        assert result.status == "unknown"


class TestMetric:
    def test_haversine_gives_km(self):
        rng = np.random.default_rng(0)
        degrees = rng.uniform(-2, 2, size=(300, 2))  # near the equator, so degrees are nearly square
        values = degrees[:, 0] + rng.normal(0, 0.1, 300)
        km_per_degree = np.deg2rad(1.0) * EARTH_RADIUS_KM
        km = autocorrelation_range(degrees, values, metric="haversine")
        plain = autocorrelation_range(degrees, values)
        assert km.lag_distances.max() == pytest.approx(plain.lag_distances.max() * km_per_degree, rel=0.01)

    def test_unknown_metric_raises(self):
        coords, values = _structured_field(50)
        with pytest.raises(ValueError, match="metric"):
            autocorrelation_range(coords, values, metric="manhattan")
