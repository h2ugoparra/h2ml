"""
h2ml/utils/variogram.py

Empirical variogram + exponential model fitting to estimate the spatial
autocorrelation range of a variable (typically model residuals).

The range is the distance at which spatial autocorrelation effectively
vanishes — the practical cutoff for a buffer zone in spatial CV.

Usage:
    from h2ml.utils.variogram import autocorrelation_range, plot_variogram

    # From raw values + coordinates
    result = autocorrelation_range(coords, residuals)
    print(f"Practical range: {result.practical_range:.1f} km")

    # With a plot
    plot_variogram(coords, residuals)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Variogram model
# ---------------------------------------------------------------------------


def _exponential_model(h: np.ndarray, nugget: float, sill: float, a: float) -> np.ndarray:
    """
    γ(h) = nugget + sill × (1 − exp(−h / a))

    The model parameter `a` is the e-folding distance. The *practical range*
    where correlation drops to ~5% is 3a (the conventional definition for the
    exponential model).
    """
    return nugget + sill * (1.0 - np.exp(-h / a))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class VariogramResult:
    """Fitted variogram parameters and derived autocorrelation range."""

    nugget: float  # variance at h=0 (measurement noise / micro-scale variation)
    sill: float  # variance added by spatial structure (total sill = nugget + sill)
    scale: float  # exponential e-folding distance (model parameter `a`)
    practical_range: float  # 3 × scale — distance where correlation ≈ 5%
    lag_distances: np.ndarray  # bin centres used for fitting
    semivariances: np.ndarray  # empirical γ per bin
    n_pairs: np.ndarray  # number of pairs per bin


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def autocorrelation_range(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_dist: Optional[float] = None,
) -> VariogramResult:
    """
    Estimate the spatial autocorrelation range from an empirical variogram.

    Args:
        coords:    (n_samples, 2) coordinate array — same units as the desired
                   range output (degrees for lat/lon, km for projected).
        values:    (n_samples,) variable to analyse. Use model residuals
                   (y_pred − y_true) to measure unexplained spatial structure.
        n_lags:    Number of distance bins for the empirical variogram (default 15).
        max_dist:  Maximum lag distance to consider. Defaults to half the maximum
                   pairwise distance (standard geostatistics convention — the
                   variogram is unreliable beyond this point).

    Returns:
        VariogramResult with fitted parameters and practical_range.

    Raises:
        RuntimeError: If curve fitting fails to converge.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)

    dists = cdist(coords, coords)  # (n, n) pairwise distances
    upper = np.triu_indices_from(dists, k=1)
    h = dists[upper]
    dz = values[upper[0]] - values[upper[1]]
    gamma = 0.5 * dz**2  # semivariance for each pair

    if max_dist is None:
        max_dist = h.max() / 2.0

    mask = h <= max_dist
    h, gamma = h[mask], gamma[mask]

    # Bin into n_lags equal-width distance classes
    edges = np.linspace(0.0, max_dist, n_lags + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(h, edges[1:-1])  # 0 … n_lags-1

    emp_gamma = np.full(n_lags, np.nan)
    n_pairs = np.zeros(n_lags, dtype=int)
    for k in range(n_lags):
        sel = bin_idx == k
        n_pairs[k] = sel.sum()
        if n_pairs[k] >= 2:
            emp_gamma[k] = gamma[sel].mean()

    # Drop empty bins before fitting
    valid = ~np.isnan(emp_gamma)
    h_fit = centres[valid]
    g_fit = emp_gamma[valid]

    # Initial parameter guesses
    nugget0 = float(np.nanmin(emp_gamma))
    sill0 = float(np.nanmax(emp_gamma) - nugget0)
    a0 = float(max_dist / 3.0)
    p0 = [max(nugget0, 0.0), max(sill0, 1e-9), a0]

    try:
        popt, _ = curve_fit(
            _exponential_model,
            h_fit,
            g_fit,
            p0=p0,
            bounds=([0, 1e-9, 1e-9], [np.inf, np.inf, np.inf]),
            maxfev=10_000,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Variogram fitting failed to converge. "
            f"Try increasing n_lags or checking that values have spatial structure. "
            f"scipy message: {e}"
        ) from e

    nugget, sill, a = popt
    return VariogramResult(
        nugget=float(nugget),
        sill=float(sill),
        scale=float(a),
        practical_range=float(3.0 * a),
        lag_distances=centres,
        semivariances=emp_gamma,
        n_pairs=n_pairs,
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    units: str = "°",
    save_path=None,
) -> VariogramResult:
    """
    Plot the empirical variogram, fitted exponential model, and practical range.

    Args:
        coords:    (n_samples, 2) coordinate array.
        values:    (n_samples,) variable (use model residuals).
        n_lags:    Distance bins (default 15).
        max_dist:  Max lag. Defaults to half the max pairwise distance.
        units:     Distance unit label for the x-axis (default "°").
        save_path: Path to save the figure. If None, shows interactively.

    Returns:
        VariogramResult (same as autocorrelation_range).
    """
    import matplotlib.pyplot as plt

    vr = autocorrelation_range(coords, values, n_lags=n_lags, max_dist=max_dist)

    h_fit = np.linspace(0, vr.lag_distances.max(), 200)
    g_fit = _exponential_model(h_fit, vr.nugget, vr.sill, vr.scale)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Empirical variogram — size scaled by number of pairs (reliability)
    valid = ~np.isnan(vr.semivariances)
    sizes = 20 + 80 * vr.n_pairs[valid] / vr.n_pairs[valid].max()
    ax.scatter(
        vr.lag_distances[valid],
        vr.semivariances[valid],
        s=sizes,
        color="steelblue",
        zorder=3,
        label="Empirical γ(h)",
    )

    # Fitted model
    ax.plot(h_fit, g_fit, color="tomato", linewidth=2, label="Exponential fit")

    # Sill + nugget reference lines
    total_sill = vr.nugget + vr.sill
    ax.axhline(total_sill, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(vr.nugget, color="grey", linestyle=":", linewidth=1, alpha=0.6)

    # Practical range marker
    ax.axvline(
        vr.practical_range,
        color="tomato",
        linestyle="--",
        linewidth=1.5,
        label=f"Practical range = {vr.practical_range:.2f}{units}",
    )

    ax.set_xlabel(f"Lag distance ({units})")
    ax.set_ylabel("Semivariance γ(h)")
    ax.set_title(f"Variogram  —  nugget={vr.nugget:.3f}  sill={vr.sill:.3f}  range={vr.practical_range:.2f}{units}")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return vr
