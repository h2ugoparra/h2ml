"""
tests/utils/test_variogram.py

Tests for autocorrelation_range().

Coverage:
    - Returns a VariogramResult with finite, positive parameters
    - Large inputs are subsampled to max_points (bounds the O(n²) matrix)
    - The subsample is deterministic given random_state
"""

from __future__ import annotations

import numpy as np

from h2ml.utils.variogram import VariogramResult, autocorrelation_range


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
