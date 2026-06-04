"""
h2ml/pipeline/final_model.py

FinalModel — deployment artifact produced after a full pipeline run.

Responsibilities
----------------
- Fit the best model (with best_params) on the full training dataset
- Handle per-sample scaling when the model requires it
- Expose predict() / predict_proba() for inference on new data
- Save / load via joblib

Separation of concerns
-----------------------
PipelineResult  — evaluation artifact (CV metrics, SHAP, stage selection)
FinalModel      — deployment artifact (fitted estimator ready for prediction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from h2ml.pipeline.pipeline import PipelineResult

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from h2ml.core.base import TaskType
from h2ml.preprocessing.transforms import INVERSE_TRANSFORMS


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


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
    """Return the conformal quantile giving ≥ 1-alpha coverage."""
    n = len(scores)
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level))


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
        elif not self.has_coords and _coords is None:
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

        return np.array(
            [
                self._block_threshold(
                    bi,
                    alpha,
                    time_bin=None if time_bins is None else int(time_bins[i]),
                )
                for i, bi in enumerate(block_indices)
            ]
        )

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


@dataclass
class FinalModel:
    """
    Fitted model ready for inference on new data.

    Attributes:
        estimator:        Sklearn-compatible estimator fitted on the full training set.
        feature_names:    Ordered list of features the model was trained on.
        task_type:        TaskType.CLASSIFICATION or TaskType.REGRESSION.
        requires_scaling: Whether StandardScaler was applied before fitting.
        scaler:           Fitted StandardScaler (None when requires_scaling is False).
        best_model_name:  Name of the model as registered in the h2ml registry.
        best_params:      Hyperparameters used for the final fit (None = defaults).
        conformal:        Global conformal calibrator built from out-of-fold residuals;
                          powers predict_interval / predict_set. None when no calibration
                          was available (e.g. partial pipeline run).
        y_transform:      Name of the y-transform applied during training (e.g. "log"),
                          or None. Predictions/intervals are in the transformed space;
                          the caller inverts them. See preprocessing/transforms.py.
        local_conformal:  Space-time block-local conformal calibrator. Used instead of
                          `conformal` when coords/times are passed to predict_interval/
                          predict_set, giving per-sample thresholds. None unless the run
                          had coordinates and a spatial splitter.
        variogram:        Fitted VariogramResult describing the spatial autocorrelation
                          range of the model's residuals (diagnostic only; not used in
                          prediction). None when coords were absent or fitting failed.

    Example:
        >>> final = result.build_final_model()
        >>> final.predict(X_new_df)
        >>> final.save("models/final_model.pkl")
        >>> final = FinalModel.load("models/final_model.pkl")
    """

    estimator: Any
    feature_names: list[str]
    task_type: TaskType
    requires_scaling: bool = False
    scaler: Optional[Any] = None
    best_model_name: Optional[str] = None
    best_params: Optional[dict] = field(default=None)
    conformal: Optional[ConformalCalibration] = field(default=None)
    y_transform: Optional[str] = None
    local_conformal: Optional[LocalConformalCalibration] = field(default=None)
    variogram: Optional[Any] = field(default=None)  # VariogramResult — imported lazily

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict on new data.

        Args:
            X: DataFrame (columns are aligned by name) or ndarray
                (columns must match feature_names order).

        Returns:
            1-D array of predictions.
        """
        return self.estimator.predict(self._prepare(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Args:
            X: DataFrame (columns aligned by name) or ndarray (columns must
                match feature_names order).

        Returns:
            Binary:     1-D array of positive-class probabilities.
            Multiclass: 2-D array of shape (n_samples, n_classes).

        Raises:
            ValueError: If the model was trained for a regression task.
        """
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks.")
        proba = self.estimator.predict_proba(self._prepare(X))
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def predict_interval(
        self,
        X: pd.DataFrame | np.ndarray,
        alpha: float = 0.10,
        coords: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Conformal prediction interval for each sample (regression only).

        The interval is centred on the point estimate and has width 2q, where q
        is the conformal threshold calibrated from out-of-fold residuals.
        Coverage is guaranteed to be ≥ 1-alpha in expectation.

        When coords and/or times are provided and the model has a
        LocalConformalCalibration, q varies per sample based on the nearest
        spatial block's residual distribution, giving wider intervals in
        high-error regions and narrower ones where the model is well calibrated.

        Note: if the pipeline used a y-transform, the interval is in the
        transformed space. Apply the inverse transform to the bounds if needed.

        Args:
            X:      Input features (DataFrame or ndarray).
            alpha:  Miscoverage level. Default 0.10 → 90% prediction intervals.
            coords: (n_samples, 2) spatial coordinates. Enables local calibration.
            times:  (n_samples,) datetime64[D] or YYYY-MM-DD strings. Enables
                    temporal calibration.

        Returns:
            (lower, upper) as a pair of 1-D arrays.

        Raises:
            ValueError: If the model is not a regression model, or if no
                conformal calibration is attached (rebuild via
                result.build_final_model() after a full pipeline run).
        """
        if self.task_type != TaskType.REGRESSION:
            raise ValueError("predict_interval is only available for regression tasks.")
        if self.conformal is None:
            raise ValueError(
                "No conformal calibration available. "
                "Rebuild FinalModel via result.build_final_model() after a full pipeline run."
            )
        y_hat = self.predict(X)
        if (coords is not None or times is not None) and self.local_conformal is not None:
            q = self.local_conformal.threshold(alpha, coords=coords, times=times)
        else:
            q = self.conformal.threshold(alpha)
        return y_hat - q, y_hat + q

    def predict_set(
        self,
        X: pd.DataFrame | np.ndarray,
        alpha: float = 0.10,
        coords: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
    ) -> list[np.ndarray]:
        """
        Conformal prediction set for each sample (classification only).

        Each prediction set contains the classes that are plausible given the
        coverage guarantee. A singleton set means the model is confident; a
        larger set means it is uncertain.

        Coverage guarantee: the true label is in the prediction set with
        probability ≥ 1-alpha.

        Nonconformity score per class k: 1 - p_k. Class k is included in the
        set when its score does not exceed the calibrated threshold q.

        When coords and/or times are provided and the model has a
        LocalConformalCalibration, q varies per sample.

        Args:
            X:      Input features (DataFrame or ndarray).
            alpha:  Miscoverage level. Default 0.10 → 90% coverage sets.
            coords: (n_samples, 2) spatial coordinates. Enables local calibration.
            times:  (n_samples,) datetime64[D] or YYYY-MM-DD strings.

        Returns:
            List of arrays of class labels, one per sample.

        Raises:
            ValueError: If the model is not a classification model, or if no
                conformal calibration is attached (rebuild via
                result.build_final_model() after a full pipeline run).
        """
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_set is only available for classification tasks.")
        if self.conformal is None:
            raise ValueError(
                "No conformal calibration available. "
                "Rebuild FinalModel via result.build_final_model() after a full pipeline run."
            )
        p = self.predict_proba(X)  # 1-D for binary, 2-D for multiclass
        classes = self.estimator.classes_

        if (coords is not None or times is not None) and self.local_conformal is not None:
            q_arr = self.local_conformal.threshold(alpha, coords=coords, times=times)
        else:
            q_arr = np.full(len(p) if p.ndim == 1 else p.shape[0], self.conformal.threshold(alpha))

        sets = []
        if p.ndim == 1:
            for pi, q in zip(p, q_arr):
                labels = []
                if pi <= q:
                    labels.append(classes[0])
                if 1 - pi <= q:
                    labels.append(classes[1])
                sets.append(np.array(labels))
        else:
            for row, q in zip(p, q_arr):
                labels = [classes[k] for k in range(len(classes)) if 1 - row[k] <= q]
                sets.append(np.array(labels))
        return sets

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist to *path* (single .pkl file via joblib). Parent directory is created if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FinalModel":
        """
        Reload a FinalModel saved with :meth:`save`.

        WARNING: only load files from trusted sources — joblib uses pickle,
        which executes arbitrary code on deserialisation.
        """
        return joblib.load(Path(path))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Align columns and apply scaling if required."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].to_numpy()
        X = np.asarray(X)
        if self.requires_scaling and self.scaler is not None:
            X = self.scaler.transform(X)
        return X  # type: ignore

    def __repr__(self) -> str:
        return (
            f"FinalModel("
            f"model={self.best_model_name!r}, "
            f"n_features={len(self.feature_names)}, "
            f"task={self.task_type.value!r})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_conformal_calibration(result, classes=None) -> Optional[ConformalCalibration]:
    """
    Build a ConformalCalibration from the out-of-fold predictions in best_cv_result.

    Regression:          nonconformity score = |y_test - y_pred_test|
    Binary classification:   nonconformity score = 1 - p(true_class)
    Multiclass classification: nonconformity score = 1 - p(true_class),
        looked up via the classes array from estimator.classes_.

    Returns None when no CV result is available or when the class mapping
    cannot be resolved for multiclass folds.
    """
    cv = result.best_cv_result
    if cv is None or not cv.folds:
        return None

    task_type = cv.task_type

    if task_type == TaskType.REGRESSION:
        scores = np.concatenate([np.abs(f.y_test - f.y_pred_test) for f in cv.folds])
    else:
        sample_scores = []
        for f in cv.folds:
            fold_scores = _classification_scores(f, classes)
            if fold_scores is None:
                return None
            sample_scores.append(fold_scores)
        scores = np.concatenate(sample_scores)

    return ConformalCalibration(
        scores=np.sort(scores),
        n=len(scores),
        task_type=task_type,
    )


def _classification_scores(f: Any, classes: Optional[Any]) -> Optional[np.ndarray]:
    """
    Compute per-sample nonconformity scores for one classification fold.

    Binary:     score = 1 - p(true_class)
    Multiclass: score = 1 - p(true_class), looked up via classes array.

    Returns None when the class mapping cannot be resolved (multiclass only).
    """
    from loguru import logger

    if f.y_prob_test.ndim == 1:
        return np.where(f.y_test == 1, 1.0 - f.y_prob_test, f.y_prob_test)

    if classes is None:
        logger.warning(
            "Conformal calibration skipped: multiclass folds require "
            "estimator.classes_ to map labels to probability columns."
        )
        return None
    label_to_idx = {c: i for i, c in enumerate(classes)}
    try:
        col_idx = np.array([label_to_idx[label] for label in f.y_test])
    except KeyError as e:
        logger.warning(
            f"Conformal calibration skipped: y_test contains label {e} not found in estimator.classes_."
        )
        return None
    return 1.0 - f.y_prob_test[np.arange(len(col_idx)), col_idx]


def _build_local_conformal(
    result: "PipelineResult",
    coords: Optional[np.ndarray],
    times: Optional[np.ndarray],
    metric: str,
    min_block_n: int = 20,
    min_compound_n: int = 5,
    classes: Optional[Any] = None,
    time_bin_resolution: str = "month",
) -> Optional[LocalConformalCalibration]:
    """
    Build a LocalConformalCalibration from OOF residuals.

    Residuals are partitioned by spatial block (from splitter.block_id_) and,
    when times are provided, additionally by time bin to form compound
    (block × time_bin) cells.

    Requires result.splitter to have a block_id_ attribute — set by both
    SpatialBlockSplitter and SPCVSplitter. Returns None when the splitter is
    absent (e.g. after PipelineResult.load()), is a plain KFold, or when fewer
    than 2 spatial blocks are represented in the OOF data.

    Args:
        result:              PipelineResult with splitter and best_cv_result.
        coords:              (n_samples, 2) spatial coordinates, or None.
        times:               (n_samples,) dates as datetime64[D] or strings, or None.
        metric:              "euclidean" or "haversine" for k-NN lookup.
        min_block_n:         Minimum OOF samples for a spatial block to use its own
                             scores (level 2 fallback).
        min_compound_n:      Minimum OOF samples for a compound cell to be used
                             (level 1). Lower than min_block_n because compound cells
                             are smaller by construction.
        classes:             estimator.classes_ for multiclass classification scoring.
        time_bin_resolution: "month" (1–12) or "season" (0–3). Only used when
                             times is not None.

    Block assignment at inference uses nearest-OOF-sample lookup in a jointly
    normalised context space [lat, lon, sin_doy, cos_doy, year], so the
    threshold adapts to both location and season. When only spatial context is
    supplied with metric="haversine", the k-NN runs in haversine space instead.
    """
    from loguru import logger

    splitter = result.splitter
    if splitter is None or not hasattr(splitter, "block_id_"):
        return None
    cv = result.best_cv_result
    if cv is None or not cv.folds:
        return None

    block_id_all: np.ndarray = splitter.block_id_
    task_type = cv.task_type

    sample_scores: dict[int, float] = {}
    for f in cv.folds:
        if task_type == TaskType.REGRESSION:
            fold_scores: Optional[np.ndarray] = np.abs(f.y_test - f.y_pred_test)
        else:
            fold_scores = _classification_scores(f, classes)
            if fold_scores is None:
                return None
        for idx, score in zip(f.test_idx, fold_scores):
            sample_scores[int(idx)] = float(score)

    if not sample_scores:
        return None

    indices = np.array(list(sample_scores.keys()))
    scores_arr = np.array([sample_scores[i] for i in indices])
    block_ids = block_id_all[indices]

    unique_blocks = np.unique(block_ids)
    block_to_idx = {int(bid): i for i, bid in enumerate(unique_blocks)}
    scores_by_block = [np.sort(scores_arr[block_ids == bid]) for bid in unique_blocks]
    oof_block_indices = np.array([block_to_idx[int(b)] for b in block_ids])

    oof_coords = coords[indices] if coords is not None else None
    oof_times = times[indices] if times is not None else None

    use_haversine = metric == "haversine" and times is None and coords is not None

    if use_haversine:
        assert oof_coords is not None  # use_haversine implies coords is not None
        oof_context_scaled = np.deg2rad(oof_coords)
        context_mean = np.zeros(2)
        context_std = np.ones(2)
    else:
        from sklearn.preprocessing import StandardScaler

        oof_context = _build_context(oof_coords, oof_times)
        scaler = StandardScaler().fit(oof_context)
        oof_context_scaled = scaler.transform(oof_context)
        context_mean = scaler.mean_
        context_std = scaler.scale_

    n_blocks = len(unique_blocks)
    if n_blocks < 2:
        logger.warning("Local conformal calibration skipped: fewer than 2 spatial blocks found in OOF data.")
        return None

    # Build compound (spatial block, time bin) score distributions when times are available
    compound: Optional[dict] = None
    _resolution: Optional[str] = None
    if oof_times is not None:
        bins = _time_bin(oof_times, time_bin_resolution)
        raw_compound: dict = {}
        for bi, tb, score in zip(oof_block_indices, bins, scores_arr):
            raw_compound.setdefault((int(bi), int(tb)), []).append(float(score))
        compound = {k: np.sort(v) for k, v in raw_compound.items()}
        _resolution = time_bin_resolution

    return LocalConformalCalibration(
        scores_by_block=scores_by_block,
        oof_context_scaled=oof_context_scaled,
        oof_block_indices=oof_block_indices,
        context_mean=context_mean,
        context_std=context_std,
        fallback_scores=np.sort(scores_arr),
        metric=metric,
        min_block_n=min_block_n,
        min_compound_n=min_compound_n,
        has_coords=coords is not None,
        has_times=times is not None,
        compound_scores=compound,
        time_bin_resolution=_resolution,
    )


def build_final_model(result: "PipelineResult") -> FinalModel:
    """
    Fit the overall best model on the full training dataset and return a
    FinalModel ready for inference.

    Picks the correct feature store (reduced or full) via best_feature_stage,
    applies StandardScaler when the model requires it, and fits with
    best_params when step 4 ran.

    Args:
        result: PipelineResult from H2MLPipeline.run().

    Returns:
        FinalModel instance.

    Raises:
        ValueError: If best_model_name is no longer present in the registry
            (e.g. the model was removed after the pipeline ran).
    """
    from h2ml.utils.registry import CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY

    assert result.step1_cv_result is not None
    task_type = result.step1_cv_result[0].task_type

    feature_stage = result.best_feature_stage or result.best_stage
    store = result.features_reduced if feature_stage == "reduced" else result.features

    registry = CLASSIFIER_REGISTRY if task_type == TaskType.CLASSIFICATION else REGRESSOR_REGISTRY
    assert result.best_model_name is not None
    entry = registry.get(result.best_model_name)
    if entry is None:
        raise ValueError(
            f"Model '{result.best_model_name}' not found in registry. Was it removed after the pipeline ran?"
        )

    # Instantiate with best_params when available, otherwise default
    if result.best_params:
        estimator = entry.model_cls(**result.best_params)
    else:
        estimator = entry.model_cls(**entry.default_kwargs)

    assert store is not None
    X = store.X
    scaler = None
    if entry.requires_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    estimator.fit(X, store.y)

    classes = getattr(estimator, "classes_", None)

    local_conformal: Optional[LocalConformalCalibration] = None
    variogram = None
    if result.splitter is not None and (store.coords is not None or store.times is not None):
        local_conformal = _build_local_conformal(
            result,
            coords=store.coords,
            times=store.times,
            metric=result.spatial_cv_metric,
            time_bin_resolution=result.time_bin_resolution,
            classes=classes,
        )
    if store.coords is not None:
        oof_preds = result.oof_predictions
        oof_labels = result.oof_labels
        if oof_preds is not None and oof_labels is not None and oof_preds.ndim == 1:
            valid = np.isfinite(oof_preds) & np.isfinite(oof_labels)
            if valid.sum() >= 10:
                from h2ml.utils.variogram import autocorrelation_range

                try:
                    variogram = autocorrelation_range(
                        store.coords[valid],
                        np.abs(oof_preds[valid] - oof_labels[valid]),
                    )
                except Exception:
                    pass

    return FinalModel(
        estimator=estimator,
        feature_names=store.feature_names,
        task_type=task_type,
        requires_scaling=entry.requires_scaling,
        scaler=scaler,
        best_model_name=result.best_model_name,
        best_params=result.best_params,
        conformal=_build_conformal_calibration(result, classes=classes),
        y_transform=result.y_transform,
        local_conformal=local_conformal,
        variogram=variogram,
    )


# ---------------------------------------------------------------------------
# Delta model — two-component presence × abundance
# ---------------------------------------------------------------------------


def _build_delta_conformal(
    clf_result: "PipelineResult",
    reg_result: "PipelineResult",
    reg_final: FinalModel,
    X_full: "pd.DataFrame | np.ndarray",
    y_full: np.ndarray,
    positive_indices: np.ndarray,
) -> Optional[ConformalCalibration]:
    """
    Build a ConformalCalibration for the combined delta output using OOF predictions.

    P_oof (all N samples) comes from clf_result.best_cv_result.
    count_oof_pos (positive samples, already inverse-transformed by CV) comes from
    reg_result.best_cv_result. For zero-count samples the regressor is called as a
    regular predictor (not OOF) and the inverse transform is applied manually.

    Pass X_full as a DataFrame (with all feature column names) when the clf and reg
    were trained on different feature sets — reg_final.predict selects its own columns.

    Nonconformity score: |y_full - (P_oof × count_hat)|

    Args:
        clf_result:       Fitted classifier PipelineResult (supplies P_oof).
        reg_result:       Fitted regressor PipelineResult (supplies positive-sample
                          count OOF predictions, already inverse-transformed).
        reg_final:        Final regressor used to score zero-count samples.
        X_full:           Features for all N samples; pass a DataFrame when clf and
                          reg use different feature sets.
        y_full:           (N,) observed counts in the original scale.
        positive_indices: Indices of presence (count > 0) samples within X_full.

    Returns:
        ConformalCalibration over the combined delta output, or None when the
        required OOF predictions are unavailable (a warning is logged).
    """
    from loguru import logger

    clf_cv = clf_result.best_cv_result
    reg_cv = reg_result.best_cv_result

    if clf_cv is None or not clf_cv.folds:
        logger.warning("Delta conformal calibration skipped: no classifier CV folds available.")
        return None
    if reg_cv is None or not reg_cv.folds:
        logger.warning("Delta conformal calibration skipped: no regressor CV folds available.")
        return None

    n_full = len(y_full)

    P_oof = clf_cv.oof_predictions  # shape (n_full,) — 1D probabilities for binary
    if P_oof is None:
        logger.warning("Delta conformal calibration skipped: classifier OOF predictions unavailable.")
        return None
    if len(P_oof) != n_full:
        logger.warning(
            f"Delta conformal calibration skipped: classifier OOF length {len(P_oof)} "
            f"does not match y_full length {n_full}. Pass the same X/y the classifier was trained on."
        )
        return None

    # count_oof_pos: shape (n_pos,) — regressor OOF predictions, already inverse-transformed
    # (CV applies inverse_fn to y_pred_test before storing in FoldResult)
    count_oof_pos = reg_cv.oof_predictions
    if count_oof_pos is None:
        logger.warning("Delta conformal calibration skipped: regressor OOF predictions unavailable.")
        return None

    count_hat_full = np.full(n_full, np.nan)
    count_hat_full[positive_indices] = count_oof_pos

    # For zero-count samples, call the final regressor (not OOF) and inverse-transform manually
    zero_mask = np.ones(n_full, dtype=bool)
    zero_mask[positive_indices] = False
    zero_indices = np.where(zero_mask)[0]
    if zero_indices.size > 0:
        # Slice preserving type: DataFrame slicing keeps column names so _prepare
        # can select the reg's feature subset when clf and reg have different features.
        X_zero = X_full.iloc[zero_indices] if isinstance(X_full, pd.DataFrame) else X_full[zero_indices]
        zero_preds = reg_final.predict(X_zero)  # transform space
        if reg_final.y_transform is not None:
            inverse_fn = INVERSE_TRANSFORMS.get(reg_final.y_transform)
            if inverse_fn is not None:
                zero_preds = inverse_fn(zero_preds)
        count_hat_full[zero_indices] = zero_preds

    delta_oof = P_oof * count_hat_full
    valid = np.isfinite(delta_oof)
    n_excluded = int((~valid).sum())
    if n_excluded:
        logger.warning(
            f"Delta conformal calibration: {n_excluded} samples excluded due to NaN (likely from failed CV folds)."
        )

    scores = np.abs(y_full[valid] - delta_oof[valid])
    return ConformalCalibration(
        scores=np.sort(scores),
        n=int(valid.sum()),
        task_type=TaskType.REGRESSION,
    )


@dataclass
class DeltaFinalModel:
    """
    Two-component delta model combining a presence/absence classifier and a
    count/abundance regressor.

    Prediction: P(present) × E(count | present)
    Conformal interval: [max(0, delta − q), delta + q]
    where q is calibrated from combined out-of-fold residuals on the full delta output.

    The regressor's y-transform (if any) is inverted inside predict() so the output is
    always in the original count scale. This differs from FinalModel, where the caller
    handles inversion — the multiplication P × count requires the original scale.

    Attributes:
        clf:             Presence/absence classifier — supplies P(present).
        reg:             Count/abundance regressor — supplies E(count | present).
        conformal:       Conformal calibrator over the combined delta output (not the
                         components); powers predict_interval. None when no calibration
                         was built.
        local_conformal: Space-time block-local calibrator used when coords/times are
                         passed to predict_interval, giving per-sample thresholds. None
                         for non-spatial runs.
        variogram:       Fitted VariogramResult for the delta residuals (diagnostic
                         only; imported lazily). None when unavailable.

    Example:
        >>> positive_idx = np.where(y_all > 0)[0]
        >>> delta = build_delta_final_model(clf_result, reg_result, X_all, y_all, positive_idx)
        >>> delta.save("models/sparrow_delta-model")
        >>> delta = DeltaFinalModel.load("models/sparrow_delta-model")
        >>> lower, upper = delta.predict_interval(X_new, alpha=0.10)
    """

    clf: FinalModel  # presence/absence classifier — supplies P(present)
    reg: FinalModel  # count/abundance regressor — supplies E(count | present)
    # Conformal calibrator over the combined delta output (not the components);
    # powers predict_interval. None when no calibration was built.
    conformal: Optional[ConformalCalibration] = field(default=None)
    # Space-time block-local calibrator used when coords/times are passed to
    # predict_interval, giving per-sample thresholds. None for non-spatial runs.
    local_conformal: Optional[LocalConformalCalibration] = field(default=None)
    variogram: Optional[Any] = field(default=None)  # VariogramResult — imported lazily

    def predict(self, X: "pd.DataFrame | np.ndarray") -> np.ndarray:
        """
        Delta prediction: P(present) × E(count | present).

        The regressor's y-transform is inverted here so the result is in the
        original count scale. Pass a DataFrame to let each sub-model select its own
        features by name; pass an ndarray only when both models share the same
        feature set and column order.

        Args:
            X: Input features (DataFrame preferred; ndarray only when clf and reg
                share the same feature set and column order).

        Returns:
            1-D array of expected counts in the original scale.
        """
        p = self.clf.predict_proba(X)
        count = self.reg.predict(X)
        if self.reg.y_transform is not None:
            inverse_fn = INVERSE_TRANSFORMS.get(self.reg.y_transform)
            if inverse_fn is not None:
                count = inverse_fn(count)
        return p * count

    def predict_interval(
        self,
        X: "pd.DataFrame | np.ndarray",
        alpha: float = 0.10,
        coords: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Conformal prediction interval for each sample (count scale).

        Coverage is guaranteed for the full delta output, not each component
        separately. The lower bound is clipped at zero (counts are non-negative).

        When coords and/or times are provided and the model has a
        LocalConformalCalibration, q varies per sample.

        Args:
            X:      Input features.
            alpha:  Miscoverage level. Default 0.10 → 90% coverage.
            coords: (n_samples, 2) spatial coordinates. Enables local calibration.
            times:  (n_samples,) datetime64[D] or YYYY-MM-DD strings.

        Returns:
            (lower, upper) pair of 1-D arrays.

        Raises:
            ValueError: If no conformal calibration is attached (rebuild via
                build_delta_final_model() with positive_indices provided).
        """
        if self.conformal is None:
            raise ValueError(
                "No conformal calibration available. "
                "Rebuild via build_delta_final_model() with positive_indices provided."
            )
        delta = self.predict(X)
        if (coords is not None or times is not None) and self.local_conformal is not None:
            q = self.local_conformal.threshold(alpha, coords=coords, times=times)
        else:
            q = self.conformal.threshold(alpha)
        return np.maximum(0.0, delta - q), delta + q

    def save(self, path: "str | Path") -> None:
        """
        Persist to *path* (directory). Creates the directory if needed.
        Saves clf.pkl, reg.pkl, conformal.pkl, local_conformal.pkl, variogram.pkl.
        Reload with DeltaFinalModel.load(path).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path / "clf.pkl")
        joblib.dump(self.reg, path / "reg.pkl")
        if self.conformal is not None:
            joblib.dump(self.conformal, path / "conformal.pkl")
        if self.local_conformal is not None:
            joblib.dump(self.local_conformal, path / "local_conformal.pkl")
        if self.variogram is not None:
            joblib.dump(self.variogram, path / "variogram.pkl")

    @classmethod
    def load(cls, path: "str | Path") -> "DeltaFinalModel":
        """
        Reload a DeltaFinalModel saved with :meth:`save`.

        WARNING: only load files from trusted sources — joblib uses pickle,
        which executes arbitrary code on deserialisation.
        """
        path = Path(path)
        clf = joblib.load(path / "clf.pkl")
        reg = joblib.load(path / "reg.pkl")
        conformal_path = path / "conformal.pkl"
        conformal = joblib.load(conformal_path) if conformal_path.exists() else None
        lc_path = path / "local_conformal.pkl"
        local_conformal = joblib.load(lc_path) if lc_path.exists() else None
        vg_path = path / "variogram.pkl"
        variogram = joblib.load(vg_path) if vg_path.exists() else None
        return cls(clf=clf, reg=reg, conformal=conformal, local_conformal=local_conformal, variogram=variogram)

    def __repr__(self) -> str:
        return (
            f"DeltaFinalModel("
            f"clf={self.clf.best_model_name!r}, "
            f"reg={self.reg.best_model_name!r}, "
            f"calibrated={self.conformal is not None})"
        )


def build_delta_final_model(
    clf_result: "PipelineResult",
    reg_result: "PipelineResult",
    X_full: "pd.DataFrame | np.ndarray",
    y_full: np.ndarray,
    positive_indices: np.ndarray,
) -> DeltaFinalModel:
    """
    Fit a DeltaFinalModel from a classifier and a regressor PipelineResult.

    Builds a clf FinalModel (all N samples) and a reg FinalModel (positive samples),
    then constructs a ConformalCalibration using combined OOF delta predictions
    following Option B — combined residual conformalization on the full output.

    Args:
        clf_result:       PipelineResult from the presence/absence classifier,
                          trained on all N samples.
        reg_result:       PipelineResult from the count regressor, trained on
                          positive-count samples (X_full[positive_indices]).
        X_full:           Feature matrix for ALL N samples. Pass a pandas DataFrame
                          (with all feature column names) when clf and reg have different
                          feature sets — the regressor will select its own columns by name.
        y_full:           Raw count targets for ALL N samples (0 for absence,
                          positive integer for presence).
        positive_indices: Integer indices into X_full / y_full identifying the
                          positive-count rows (i.e. those the regressor was trained on).

    Returns:
        DeltaFinalModel with clf, reg, and a combined ConformalCalibration.

    Example
    -------
    >>> positive_idx = np.where(y_all > 0)[0]
    >>> X_df = pd.DataFrame(X_all, columns=feature_names)
    >>> delta = build_delta_final_model(clf_result, reg_result, X_df, y_all, positive_idx)
    >>> delta.save("models/sparrow_delta-model")
    """
    clf_final = build_final_model(clf_result)
    reg_final = build_final_model(reg_result)
    conformal = _build_delta_conformal(clf_result, reg_result, reg_final, X_full, y_full, positive_indices)

    # Local conformal: use clf_result's splitter (covers all N samples) and
    # build from the combined delta OOF residuals via the clf store's coords/times.
    local_conformal: Optional[LocalConformalCalibration] = None
    _clf_stage = clf_result.best_feature_stage or clf_result.best_stage
    clf_store = clf_result.features_reduced if _clf_stage == "reduced" else clf_result.features
    if clf_store is not None and clf_result.splitter is not None:
        if clf_store.coords is not None or clf_store.times is not None:
            local_conformal = _build_local_conformal(
                clf_result,
                coords=clf_store.coords,
                times=clf_store.times,
                metric=clf_result.spatial_cv_metric,
                time_bin_resolution=clf_result.time_bin_resolution,
            )

    variogram = clf_final.variogram  # reuse — same spatial residual structure

    return DeltaFinalModel(
        clf=clf_final,
        reg=reg_final,
        conformal=conformal,
        local_conformal=local_conformal,
        variogram=variogram,
    )
