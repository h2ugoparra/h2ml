"""
h2ml/evaluation/conformal.py

Conformal calibration data structures and the space-time math behind them.

ConformalCalibration       — global nonconformity-score quantiles giving coverage-
                             guaranteed intervals (regression) and sets (classification).
LocalConformalCalibration  — block-local thresholds in combined space-time, with a
                             three-level (compound cell -> spatial block -> global) fallback.

These types are built from out-of-fold residuals by the factory helpers in
pipeline/final_model.py and consumed by FinalModel / DeltaFinalModel at inference. They
depend only on numpy, pandas, scikit-learn, and core types (no other h2ml package), so they
sit in the evaluation layer alongside metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from h2ml.core.base import TaskType

_SEASON_NAMES = {0: "DJF", 1: "MAM", 2: "JJA", 3: "SON"}


def _bin_label(time_bin: int, resolution: Optional[str]) -> str:
    """Human-readable label for an integer time bin (season name or month number)."""
    if resolution == "season":
        return _SEASON_NAMES.get(time_bin, str(time_bin))
    return str(time_bin)  # month resolution → "1".."12"


@dataclass
class ConformalCalibration:
    """
    Nonconformity scores from out-of-fold CV predictions, used to construct
    finite-sample coverage-guaranteed prediction intervals (regression) or
    prediction sets (classification).

    Scores are pre-sorted ascending. The threshold at level 1-alpha is the
    ceil((1-alpha)(n+1))/n quantile, which guarantees marginal coverage ≥ 1-alpha.

    Attributes:
        scores:    Sorted nonconformity scores from calibration folds.
        n:         Number of calibration samples (len(scores)).
        task_type: TaskType of the calibrated model.
    """

    scores: np.ndarray
    n: int
    task_type: TaskType

    def threshold(self, alpha: float) -> float:
        """
        Return the nonconformity threshold that gives ≥ 1-alpha coverage.

        Args:
            alpha: Miscoverage level (e.g. 0.10 for 90% coverage).
        """
        return _conformal_quantile(self.scores, alpha)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the conformal quantile giving ≥ 1-alpha coverage.

    scores must be sorted ascending — all call sites pre-sort. Indexes the sorted
    array directly with linear interpolation, which is numerically identical to
    np.quantile(method="linear") but skips its internal partition. This matters
    because LocalConformalCalibration calls this once per (block, time-bin) cell.
    """
    n = len(scores)
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    pos = level * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(scores[lo])
    return float(scores[lo] + (pos - lo) * (scores[hi] - scores[lo]))


def _time_bin(times: np.ndarray, resolution: str) -> np.ndarray:
    """Convert dates to integer time bins.

    resolution="month"  → 1–12
    resolution="season" → 0=DJF, 1=MAM, 2=JJA, 3=SON
    """
    t = np.asarray(times, dtype="datetime64[D]")
    month = (t.astype("datetime64[M]").astype(int) % 12) + 1
    if resolution == "month":
        return month.astype(np.int8)
    return ((month % 12) // 3).astype(np.int8)


def _encode_times(times: np.ndarray) -> np.ndarray:
    """
    Encode a date array to (sin_doy, cos_doy, year) columns.

    sin/cos capture seasonal position circularly (Dec is adjacent to Jan).
    year is passed raw so StandardScaler normalises it relative to the training
    period, capturing inter-annual drift proportionally.

    Args:
        times: (n,) array of datetime64[D] or YYYY-MM-DD strings.

    Returns:
        (n, 3) float array — columns [sin_doy, cos_doy, year].
    """
    t = np.asarray(times, dtype="datetime64[D]")
    year = (t.astype("datetime64[Y]").astype(int) + 1970).astype(float)
    doy = (t - t.astype("datetime64[Y]")).astype(int) + 1
    tau = 2.0 * np.pi * doy / 365.25
    return np.column_stack([np.sin(tau), np.cos(tau), year])


def _build_context(
    coords: Optional[np.ndarray],
    times: Optional[np.ndarray],
) -> np.ndarray:
    """
    Stack available spatial and temporal context into one matrix.

    Args:
        coords: (n_samples, 2) spatial coordinates, or None.
        times:  (n_samples,) datetime-like temporal context, or None.

    Returns:
        (n_samples, D) float matrix concatenating the provided parts. Coords
        contribute 2 columns; times are encoded via _encode_times.

    Raises:
        ValueError: If both coords and times are None.
    """
    parts = []
    if coords is not None:
        parts.append(np.asarray(coords, dtype=float))
    if times is not None:
        parts.append(_encode_times(times))
    if not parts:
        raise ValueError("At least one of coords or times must be provided.")
    return np.hstack(parts)


@dataclass
class LocalConformalCalibration:
    """
    Block-local conformal calibration in combined space-time.

    Threshold lookup uses three levels of fallback:
      1. Compound cell (spatial block × time bin) — finest granularity.
      2. Spatial block (all time bins pooled) — if compound cell has < min_compound_n samples.
      3. Global fallback — if spatial block has < min_block_n samples.

    At inference each query point is assigned to the spatial block of its nearest OOF
    training sample via k-NN in a jointly normalised context space. The time bin is
    derived from the query point's own date (not the nearest OOF sample's date), so
    the seasonal threshold reflects when the prediction is being made.

    Nearest-OOF-sample lookup (not centroid) is used so that non-convex or elongated
    AHC clusters (SPCVSplitter) are handled correctly.

    Attributes:
        scores_by_block:     Sorted nonconformity scores per spatial block.
                             Index i matches oof_block_indices == i.
        oof_context_scaled:  (n_oof, D) OOF context already normalised by context_mean/std.
                             Used as the k-NN index at inference.
        oof_block_indices:   (n_oof,) int index into scores_by_block per OOF sample.
        context_mean:        (D,) scaler mean — applied to normalise query context.
        context_std:         (D,) scaler std.
        fallback_scores:     Global sorted scores — level-3 fallback.
        metric:              "euclidean" or "haversine". Haversine only applies when
                             has_coords=True, has_times=False, and D==2.
        min_block_n:         Minimum OOF samples a spatial block needs to use its own
                             scores (level 2). Smaller blocks fall to global (level 3).
        has_coords:          Whether spatial coordinates were included at calibration time.
                             Coords passed at inference are silently ignored if False.
        has_times:           Whether temporal context was included at calibration time.
                             Times passed at inference are silently ignored if False.
        compound_scores:     {(block_idx, time_bin): sorted np.ndarray} — level-1 lookup.
                             None when times were absent at calibration.
        time_bin_resolution: "month" (bins 1–12) or "season" (0=DJF…3=SON).
                             None when times were absent at calibration.
        min_compound_n:      Minimum OOF samples a compound cell needs to be used (level 1).
                             Lower than min_block_n because compound cells are smaller by
                             construction (n_blocks × n_bins cells share the same total count).
    """

    scores_by_block: list[np.ndarray]
    oof_context_scaled: np.ndarray
    oof_block_indices: np.ndarray
    context_mean: np.ndarray
    context_std: np.ndarray
    fallback_scores: np.ndarray
    metric: str
    min_block_n: int = 20
    has_coords: bool = True  # whether coords were included at calibration time
    has_times: bool = False  # whether times were included at calibration time
    compound_scores: Optional[dict] = field(default=None)
    # {(block_idx, time_bin): sorted np.ndarray} — None when times absent at calibration
    time_bin_resolution: Optional[str] = field(default=None)
    # "month" or "season"; None when times absent at calibration
    min_compound_n: int = 5
    # Lower threshold for compound (spatial × time) cells — these are smaller by
    # construction (n_blocks × n_bins cells sharing the same total sample count),
    # so a tighter minimum than min_block_n is appropriate.

    def threshold(
        self,
        alpha: float,
        coords: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return per-sample nonconformity thresholds.

        Only the context dimensions present at calibration time are used —
        extra dimensions silently dropped so scaler shapes always match (e.g.
        times passed to a model built without store.times are ignored).

        When times is provided and time_bin_resolution is set, each sample's
        threshold is looked up from the compound (spatial block × time bin) cell
        first, falling back through spatial block and global as needed.

        Args:
            alpha:  Miscoverage level (e.g. 0.10 → 90% coverage).
            coords: (n_samples, 2) spatial coordinates. Degrees when metric="haversine".
            times:  (n_samples,) datetime64[D] or YYYY-MM-DD strings.

        Returns:
            (n_samples,) float array of per-sample thresholds.
        """
        _coords = coords if self.has_coords else None
        _times = times if self.has_times else None
        context = _build_context(_coords, _times)

        # Determine which columns of the stored scaler to use.
        # The scaler was fitted on [coords (2 cols), times (3 cols)] in that order.
        # When only a subset is available at inference, project to matching columns.
        n_coord_cols = 2 if self.has_coords else 0
        if self.has_times and _times is None:
            # Times expected but not provided — use spatial subspace only for k-NN.
            col_slice = slice(0, n_coord_cols)
        elif self.has_coords and _coords is None:
            # Coords expected but not provided — use temporal subspace only.
            col_slice = slice(n_coord_cols, None)
        else:
            col_slice = slice(None)

        use_haversine = self.metric == "haversine" and not self.has_times and _times is None and context.shape[1] == 2
        if use_haversine:
            context_scaled = np.deg2rad(context)
            oof_ctx = self.oof_context_scaled
        else:
            mean = self.context_mean[col_slice]
            std = self.context_std[col_slice]
            context_scaled = (context - mean) / std
            oof_ctx = self.oof_context_scaled[:, col_slice]

        block_indices = self._nearest_block(context_scaled, oof_ctx, use_haversine)

        time_bins = None
        if self.has_times and times is not None and self.time_bin_resolution:
            time_bins = _time_bin(times, self.time_bin_resolution)

        # A sample's threshold depends only on (block_idx, time_bin) for a fixed
        # alpha, so memoise per cell — a large prediction grid has far more samples
        # than distinct (block, bin) cells, turning n quantile calls into a handful.
        cell_cache: dict = {}
        out = np.empty(len(block_indices), dtype=float)
        for i, bi in enumerate(block_indices):
            tb = None if time_bins is None else int(time_bins[i])
            key = (int(bi), tb)
            q = cell_cache.get(key)
            if q is None:
                q = self._block_threshold(int(bi), alpha, time_bin=tb)
                cell_cache[key] = q
            out[i] = q
        return out

    def summary(self, alpha: float = 0.10, max_blocks: Optional[int] = None) -> pd.DataFrame:
        """Per-block (and per-time-bin) calibration thresholds at the given alpha.

        Returns one 'spatial' row per block (all time bins pooled) plus, when compound
        cells exist, one 'compound' row per populated (block, time_bin) cell. The bin
        labels follow self.time_bin_resolution (season → DJF/MAM/JJA/SON, month → 1–12),
        so no resolution argument is needed.

        Columns:
            block:    Spatial block index (into scores_by_block).
            level:    "spatial" or "compound".
            time_bin: Integer time bin for compound rows; <NA> for spatial rows.
            bin_name: "all" for spatial rows; season/month label for compound rows.
            n:        Number of calibration samples in the cell.
            q:        Conformal quantile at alpha (the threshold this cell would apply).
            used:     Which fallback level a query resolving here actually hits —
                      "compound" (n ≥ min_compound_n), "block" (block n ≥ min_block_n),
                      or "global" (falls through to fallback_scores).

        Args:
            alpha:      Miscoverage level (0.10 → 90% coverage).
            max_blocks: Cap the number of blocks reported (None = all).
        """
        n_blocks = len(self.scores_by_block)
        if max_blocks is not None:
            n_blocks = min(n_blocks, max_blocks)

        rows = []
        for block_idx in range(n_blocks):
            block_scores = self.scores_by_block[block_idx]
            block_n = len(block_scores)
            block_ok = block_n >= self.min_block_n
            rows.append(
                {
                    "block": block_idx,
                    "level": "spatial",
                    "time_bin": pd.NA,
                    "bin_name": "all",
                    "n": block_n,
                    "q": _conformal_quantile(block_scores, alpha),
                    "used": "block" if block_ok else "global",
                }
            )

            if self.compound_scores is None:
                continue
            cells = sorted(tb for (b, tb) in self.compound_scores if b == block_idx)
            for tb in cells:
                cs = self.compound_scores[(block_idx, tb)]
                if len(cs) >= self.min_compound_n:
                    used = "compound"
                elif block_ok:
                    used = "block"
                else:
                    used = "global"
                rows.append(
                    {
                        "block": block_idx,
                        "level": "compound",
                        "time_bin": tb,
                        "bin_name": _bin_label(tb, self.time_bin_resolution),
                        "n": len(cs),
                        "q": _conformal_quantile(cs, alpha),
                        "used": used,
                    }
                )

        df = pd.DataFrame(rows, columns=["block", "level", "time_bin", "bin_name", "n", "q", "used"])
        df["time_bin"] = df["time_bin"].astype("Int64")
        return df

    def _nearest_block(self, context_scaled: np.ndarray, oof_ctx: np.ndarray, use_haversine: bool) -> np.ndarray:
        """Return block index of the nearest OOF training sample per query point."""
        from sklearn.neighbors import NearestNeighbors

        if use_haversine:
            nn = NearestNeighbors(n_neighbors=1, metric="haversine", algorithm="ball_tree")
        else:
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(oof_ctx)
        nearest = nn.kneighbors(context_scaled, return_distance=False).ravel()
        return self.oof_block_indices[nearest]

    def _block_threshold(self, block_idx: int, alpha: float, time_bin: Optional[int] = None) -> float:
        """Return threshold for one sample using the three-level fallback.

        Level 1: compound cell (block_idx, time_bin) if ≥ min_compound_n samples.
        Level 2: spatial block scores if ≥ min_block_n samples.
        Level 3: global fallback_scores.
        """
        # Level 1 — compound (spatial block × time bin)
        if self.compound_scores is not None and time_bin is not None:
            cs = self.compound_scores.get((block_idx, time_bin))
            if cs is not None and len(cs) >= self.min_compound_n:
                return _conformal_quantile(cs, alpha)
        # Level 2 — spatial block only
        scores = self.scores_by_block[block_idx]
        if len(scores) >= self.min_block_n:
            return _conformal_quantile(scores, alpha)
        # Level 3 — global fallback
        return _conformal_quantile(self.fallback_scores, alpha)
