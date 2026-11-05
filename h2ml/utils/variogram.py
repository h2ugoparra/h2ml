"""
h2ml/utils/variogram.py

Robust empirical variogram + exponential model fitting to estimate the spatial
autocorrelation range of a variable (typically model residuals).

The range is the distance at which spatial autocorrelation effectively
vanishes — the practical cutoff for a buffer zone in spatial CV.

Usage:
    from h2ml.utils.variogram import autocorrelation_range, plot_variogram

    # From raw values + coordinates
    result = autocorrelation_range(coords, residuals)
    print(f"Practical range: {result.practical_range:.1f} ({result.status})")

    # With a plot
    plot_variogram(coords, residuals)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances

EARTH_RADIUS_KM = 6371.0

# Share of the total sill that must be spatially structured for a range to mean anything.
MIN_STRUCTURED_SHARE = 0.20
# Scale fitted within this fraction of its upper limit counts as stuck there.
AT_BOUND_TOLERANCE = 0.99
# Permutation check for structure: shuffles, and the p-value at or below which it passes.
N_PERMUTATIONS = 99
STRUCTURE_ALPHA = 0.05

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
    """
    Fitted variogram parameters and derived autocorrelation range.

    Attributes:
        nugget:          Semivariance at lag h=0 — measurement noise / micro-scale
                         variation not resolved by the sampling.
        sill:            Semivariance added by spatial structure; the total sill
                         (the plateau the variogram reaches) is nugget + sill.
        scale:           Exponential e-folding distance (model parameter `a`) — the
                         lag at which ~63% of the structured variance is reached.
        practical_range: 3 × scale — the lag where correlation has decayed to ≈ 5%;
                         the effective autocorrelation range.
        lag_distances:   Bin centres (lag distances) used for fitting.
        semivariances:   Robust (Cressie–Hawkins) semivariance γ per bin.
        n_pairs:         Number of point pairs contributing to each bin.
        status:          How far to trust practical_range. "ok" — the variogram levels off
                         within the lags. "at_bound" — it doesn't, so the true range is at
                         least practical_range. "no_structure" — no resolved spatial
                         structure was found, so ignore the range; on small or very
                         zero-heavy samples, weak structure can land here too. "unknown" —
                         fitted before status existed.
    """

    nugget: float  # variance at h=0 (measurement noise / micro-scale variation)
    sill: float  # variance added by spatial structure (total sill = nugget + sill)
    scale: float  # exponential e-folding distance (model parameter `a`)
    practical_range: float  # 3 × scale — distance where correlation ≈ 5%
    lag_distances: np.ndarray  # bin centres used for fitting
    semivariances: np.ndarray  # empirical γ per bin
    n_pairs: np.ndarray  # number of pairs per bin
    status: str = "unknown"  # "ok", "at_bound", "no_structure"; "unknown" on older saves


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _pairwise_distances(coords: np.ndarray, metric: Literal["euclidean", "haversine"]) -> np.ndarray:
    """Pairwise distances: coordinate units for euclidean, km for haversine on (lat, lon) degrees."""
    if metric == "haversine":
        return pairwise_distances(np.deg2rad(coords), metric="haversine") * EARTH_RADIUS_KM
    return cdist(coords, coords)


def _robust_semivariance(dz: np.ndarray) -> float:
    """Cressie–Hawkins estimate, so a few huge residuals can't drive a lag's value."""
    n = dz.size
    return 0.5 * float(np.mean(np.sqrt(np.abs(dz)))) ** 4 / (0.457 + 0.494 / n + 0.045 / n**2)


def _nearby_pairs_agree(values: np.ndarray, i: np.ndarray, j: np.ndarray, random_state: int) -> bool:
    """
    Permutation check: do the pairs (i, j) differ less than shuffled values do?

    A fitted curve can look structured on pure noise when residuals are spiky, so this
    asks the data directly. Ranks make it respond to where large values occur, not to how
    large a few of them are.
    """
    if i.size == 0:
        return False
    rng = np.random.default_rng(random_state)
    ranks = rankdata(values)
    observed = np.mean((ranks[i] - ranks[j]) ** 2)
    n_as_close = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(ranks)
        n_as_close += np.mean((shuffled[i] - shuffled[j]) ** 2) <= observed
    return (n_as_close + 1) / (N_PERMUTATIONS + 1) <= STRUCTURE_ALPHA


def _fit_status(
    nugget: float, sill: float, scale: float, max_scale: float, first_lag: float, nearby_pairs_agree: bool
) -> str:
    """Classify a fit. Structure shorter than the first lag can't be told from the nugget."""
    if not nearby_pairs_agree or sill < MIN_STRUCTURED_SHARE * (nugget + sill) or 3.0 * scale < first_lag:
        return "no_structure"
    return "at_bound" if scale >= AT_BOUND_TOLERANCE * max_scale else "ok"


def autocorrelation_range(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    max_points: int = 3_000,
    random_state: int = 42,
    metric: Literal["euclidean", "haversine"] = "euclidean",
) -> VariogramResult:
    """
    Estimate the spatial autocorrelation range from an empirical variogram.

    Args:
        coords:    (n_samples, 2) coordinate array. With metric="euclidean" the range
                   comes out in these units; with "haversine" they are (lat, lon)
                   degrees and the range is in km.
        values:    (n_samples,) variable to analyse. Use model residuals
                   (y_pred − y_true) to measure unexplained spatial structure.
        n_lags:    Number of distance bins for the empirical variogram (default 15).
        max_dist:  Maximum lag distance to consider. Defaults to half the maximum
                   pairwise distance (standard geostatistics convention — the
                   variogram is unreliable beyond this point).
        max_points: Cap on the number of points used. The pairwise distance matrix
                   is O(n²) in memory, so for larger inputs a random subsample of
                   this size is drawn (the range estimate is stable under
                   subsampling). Default 3 000 (~72 MB matrix).
        random_state: Seed for the subsample draw and the permutation check.
        metric:    "euclidean" or "haversine", as in spatial_cv_metric.

    Returns:
        VariogramResult with fitted parameters and practical_range, which is capped at
        max_dist. Check status before using the range.

    Raises:
        ValueError: If metric is unknown.
        RuntimeError: If curve fitting fails to converge.
    """
    if metric not in ("euclidean", "haversine"):
        raise ValueError(f"metric must be 'euclidean' or 'haversine', got {metric!r}")
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)

    # Subsample before the O(n²) distance matrix to bound memory on large inputs.
    if coords.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[sel]
        values = values[sel]

    dists = _pairwise_distances(coords, metric)  # (n, n) pairwise distances
    upper = np.triu_indices_from(dists, k=1)
    h = dists[upper]

    if max_dist is None:
        max_dist = float(h.max()) / 2.0

    mask = h <= max_dist
    h, first, second = h[mask], upper[0][mask], upper[1][mask]
    dz = values[first] - values[second]

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
            emp_gamma[k] = _robust_semivariance(dz[sel])

    # Drop empty bins before fitting
    valid = ~np.isnan(emp_gamma)
    h_fit = centres[valid]
    g_fit = emp_gamma[valid]

    # Initial parameter guesses
    nugget0 = float(np.nanmin(emp_gamma))
    sill0 = float(np.nanmax(emp_gamma) - nugget0)
    max_scale = float(max_dist / 3.0)  # the data can't show a range beyond the longest lag
    a0 = max_scale / 2.0
    p0 = [max(nugget0, 0.0), max(sill0, 1e-9), a0]

    try:
        popt, _ = curve_fit(
            _exponential_model,
            h_fit,
            g_fit,
            p0=p0,
            bounds=([0, 1e-9, 1e-9], [np.inf, np.inf, max_scale]),
            maxfev=10_000,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Variogram fitting failed to converge. "
            f"Try increasing n_lags or checking that values have spatial structure. "
            f"scipy message: {e}"
        ) from e

    nugget, sill, a = popt
    short = h <= max_dist / 3.0
    return VariogramResult(
        nugget=float(nugget),
        sill=float(sill),
        scale=float(a),
        practical_range=float(3.0 * a),
        lag_distances=centres,
        semivariances=emp_gamma,
        n_pairs=n_pairs,
        status=_fit_status(
            float(nugget),
            float(sill),
            float(a),
            max_scale,
            float(h_fit[0]),
            _nearby_pairs_agree(values, first[short], second[short], random_state),
        ),
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    units: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    metric: Literal["euclidean", "haversine"] = "euclidean",
) -> VariogramResult:
    """
    Plot the empirical variogram, fitted exponential model, and practical range.

    Args:
        coords:    (n_samples, 2) coordinate array.
        values:    (n_samples,) variable (use model residuals).
        n_lags:    Distance bins (default 15).
        max_dist:  Max lag. Defaults to half the max pairwise distance.
        units:     Distance unit label for the x-axis. Defaults to "km" for haversine,
                   otherwise "°".
        save_path: Path to save the figure. If None, shows interactively.
        metric:    "euclidean" or "haversine", as in spatial_cv_metric.

    Returns:
        VariogramResult (same as autocorrelation_range).
    """
    import matplotlib.pyplot as plt

    if units is None:
        units = "km" if metric == "haversine" else "°"
    vr = autocorrelation_range(coords, values, n_lags=n_lags, max_dist=max_dist, metric=metric)

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
    ax.set_title(
        f"Variogram  —  nugget={vr.nugget:.3f}  sill={vr.sill:.3f}  "
        f"range={vr.practical_range:.2f}{units}  ({vr.status})"
    )
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return vr
