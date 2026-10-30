"""
Geo-spatial prediction utilities bridging h2ml FinalModels and h2mare ParquetIndexer.

No optional extra is required — h2mare is a core dependency and brings in cartopy
and polars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import polars as pl
from loguru import logger

from h2ml.core.base import TaskType
from h2ml.pipeline.final_model import DeltaFinalModel, FinalModel
from h2ml.preprocessing.transforms import INVERSE_TRANSFORMS


def _aoa_series(model: FinalModel, X: np.ndarray, idx: pl.Series, n: int, col_name: str) -> tuple[pl.Series, ...]:
    """
    '{col_name}_di' (Float32 dissimilarity index) and '{col_name}_aoa' (Boolean, True
    inside the area of applicability) aligned to the original grid; () with a warning
    when the model has no applicability domain (e.g. built before it existed).
    """
    if getattr(model, "applicability", None) is None:
        logger.warning(
            f"'{col_name}': model has no area of applicability — rebuild it with "
            "result.build_final_model(). No AOA columns written."
        )
        return ()
    di, inside = model.area_of_applicability(X)
    return (
        pl.Series(f"{col_name}_di", [None] * n, dtype=pl.Float32).scatter(idx, di.astype(np.float32)),
        pl.Series(f"{col_name}_aoa", [None] * n, dtype=pl.Boolean).scatter(idx, inside),
    )


def _predict_single(
    df_original: pl.DataFrame,
    model: FinalModel,
    col_name: str,
    alpha: Optional[float] = None,
    local: bool = False,
    aoa: bool = False,
) -> tuple[pl.Series, ...]:
    """
    Generate predictions (and optional conformal intervals) for one target.

    Rows with any null feature value are excluded from inference and appear as
    null in the returned Series, preserving alignment with the original grid.

    When alpha is set and local=True, lon/lat/time columns are extracted from
    df_original and passed to LocalConformalCalibration.threshold() so that
    interval widths vary by location and season. Pass local=False to use the
    global conformal threshold regardless of available geo columns.

    Args:
        df_original: Full feature DataFrame with a row 'index' column.
        model:       Loaded FinalModel (handles scaling and task type internally).
        col_name:    Name for the prediction Series.
        alpha:       Miscoverage level for conformal intervals. When None or the
                     model has no calibration, only the point prediction is returned.
        local:       If True, use LocalConformalCalibration when available to produce
                     spatially/temporally varying interval widths.
                     If False (default), always use the global conformal threshold.
        aoa:         If True, append '{col_name}_di' and '{col_name}_aoa' area-of-
                     applicability columns (see FinalModel.area_of_applicability).

    Returns:
        (pred_series,) when alpha is None or the model is uncalibrated.
        (pred_series, lower_series, upper_series) for calibrated regression or
        calibrated binary classifiers with alpha set.
        With aoa=True, the DI and AOA series follow.

    Raises:
        ValueError: For multiclass classifiers — predictions are stored as a single
            Float32 column, which cannot hold a probability matrix.
    """
    geo_cols = [c for c in ["lon", "lat", "time"] if c in df_original.columns]
    df_clean = df_original.select(model.feature_names + ["index"] + geo_cols).drop_nulls()
    X = df_clean.select(model.feature_names).to_numpy()
    idx = df_clean["index"]
    n = len(df_original)

    # (lat, lon): the order local conformal was calibrated on (store.coords)
    coords = df_clean.select(["lat", "lon"]).to_numpy() if "lon" in geo_cols and "lat" in geo_cols else None
    times = df_clean["time"].cast(pl.Utf8).to_numpy() if "time" in geo_cols else None
    extra = _aoa_series(model, X, idx, n, col_name) if aoa else ()

    if model.task_type == TaskType.CLASSIFICATION:
        preds = model.predict_proba(X).astype(np.float32)
        if preds.ndim == 2:
            raise ValueError(
                f"'{col_name}': multiclass classifiers are not supported — predictions are "
                "stored as a single Float32 column, which cannot hold a probability matrix. "
                "Use a binary (presence/absence) model."
            )
        pred_series = pl.Series(col_name, [None] * n, dtype=pl.Float32).scatter(idx, preds)
        if alpha is not None and model.conformal is not None and preds.ndim == 1:
            # Binary only: mirror the regression formula into probability space.
            # q is in probability units (nonconformity = 1 - p(true_class) ∈ [0,1]),
            # so p ± q is a meaningful calibration band. Clipped to [0, 1].
            if local and model.local_conformal is not None and (coords is not None or times is not None):
                q = model.local_conformal.threshold(alpha, coords=coords, times=times)
            else:
                q = float(model.conformal.threshold(alpha))
            lower_series = pl.Series(f"{col_name}_pi_lower", [None] * n, dtype=pl.Float32).scatter(
                idx, np.clip(preds - q, 0.0, 1.0).astype(np.float32)
            )
            upper_series = pl.Series(f"{col_name}_pi_upper", [None] * n, dtype=pl.Float32).scatter(
                idx, np.clip(preds + q, 0.0, 1.0).astype(np.float32)
            )
            return (pred_series, lower_series, upper_series) + extra
        return (pred_series,) + extra

    # Regression: predictions are inverse-transformed to the original scale first;
    # q is calibrated on original-scale OOF residuals, so intervals are built
    # around the original-scale point estimate (never in transform space).
    raw = model.predict(X)
    inverse_fn = INVERSE_TRANSFORMS.get(model.y_transform) if model.y_transform else None
    preds = inverse_fn(raw).astype(np.float32) if inverse_fn else raw.astype(np.float32)
    pred_series = pl.Series(col_name, [None] * n, dtype=pl.Float32).scatter(idx, preds)

    if alpha is not None and model.conformal is not None:
        if local and model.local_conformal is not None and (coords is not None or times is not None):
            q = model.local_conformal.threshold(alpha, coords=coords, times=times)
        else:
            q = model.conformal.threshold(alpha)
        lower = np.maximum(0.0, preds - q).astype(np.float32)
        upper = (preds + q).astype(np.float32)
        lower_series = pl.Series(f"{col_name}_pi_lower", [None] * n, dtype=pl.Float32).scatter(idx, lower)
        upper_series = pl.Series(f"{col_name}_pi_upper", [None] * n, dtype=pl.Float32).scatter(idx, upper)
        return (pred_series, lower_series, upper_series) + extra

    return (pred_series,) + extra


def _predict_delta_single(
    df_original: pl.DataFrame,
    model: "DeltaFinalModel",
    col_name: str,
    alpha: Optional[float] = None,
    local: bool = False,
    aoa: bool = False,
) -> tuple[pl.Series, ...]:
    """
    Generate delta predictions (and optional conformal intervals) for one target.

    Rows where any feature needed by either sub-model is null are excluded.
    The regressor's y-transform is inverted before multiplying so the delta is
    in the original count scale. Interval semantics mirror _predict_single —
    local=True uses LocalConformalCalibration when available.

    Args:
        df_original: Full feature DataFrame with a row 'index' column.
        model:       Loaded DeltaFinalModel.
        col_name:    Name for the prediction Series.
        alpha:       Miscoverage level. When None or model has no calibration,
                     only the point prediction is returned.
        local:       If True, use LocalConformalCalibration when available.
                     If False (default), always use the global conformal threshold.
        aoa:         If True, append '{col_name}_aoa' (Boolean): True where the row is
                     inside the area of applicability of both sub-models.

    Returns:
        (pred_series,) or (pred_series, lower_series, upper_series), followed by the
        AOA series when aoa=True.
    """
    clf_features = model.clf.feature_names
    reg_features = model.reg.feature_names
    all_features = list(dict.fromkeys(clf_features + reg_features))

    geo_cols = [c for c in ["lon", "lat", "time"] if c in df_original.columns]
    df_clean = df_original.select(all_features + ["index"] + geo_cols).drop_nulls()
    idx = df_clean["index"]
    n = len(df_original)

    # (lat, lon): the order local conformal was calibrated on (store.coords)
    coords = df_clean.select(["lat", "lon"]).to_numpy() if "lon" in geo_cols and "lat" in geo_cols else None
    times = df_clean["time"].cast(pl.Utf8).to_numpy() if "time" in geo_cols else None

    X_clf = df_clean.select(clf_features).to_numpy()
    X_reg = df_clean.select(reg_features).to_numpy()

    p = model.clf.predict_proba(X_clf)
    count = model.reg.predict(X_reg)
    if model.reg.y_transform is not None:
        inverse_fn = INVERSE_TRANSFORMS.get(model.reg.y_transform)
        if inverse_fn is not None:
            count = inverse_fn(count)

    delta = (p * count).astype(np.float32)
    pred_series = pl.Series(col_name, [None] * n, dtype=pl.Float32).scatter(idx, delta)

    extra: tuple[pl.Series, ...] = ()
    if aoa:
        if model.clf.applicability is None or model.reg.applicability is None:
            logger.warning(
                f"'{col_name}': a sub-model has no area of applicability — rebuild both with "
                "result.build_final_model(). No AOA column written."
            )
        else:
            inside = model.clf.area_of_applicability(X_clf)[1] & model.reg.area_of_applicability(X_reg)[1]
            extra = (pl.Series(f"{col_name}_aoa", [None] * n, dtype=pl.Boolean).scatter(idx, inside),)

    if alpha is not None and model.conformal is not None:
        if local and model.local_conformal is not None and (coords is not None or times is not None):
            q = model.local_conformal.threshold(alpha, coords=coords, times=times)
        else:
            q = float(model.conformal.threshold(alpha))
        lower = np.maximum(0.0, delta - q).astype(np.float32)
        upper = (delta + q).astype(np.float32)
        lower_series = pl.Series(f"{col_name}_pi_lower", [None] * n, dtype=pl.Float32).scatter(idx, lower)
        upper_series = pl.Series(f"{col_name}_pi_upper", [None] * n, dtype=pl.Float32).scatter(idx, upper)
        return (pred_series, lower_series, upper_series) + extra

    return (pred_series,) + extra


def predict_for_year(
    target: str | list[str],
    year: int | str,
    root_dir: Path,
    input_parquet_dir: Path,
    schema: str,
    geo_extent: tuple[float, float, float, float],
    alpha: Optional[float] = None,
    local: bool = False,
    aoa: bool = False,
) -> pl.DataFrame:
    """
    Load pre-trained FinalModels and generate predictions for a full calendar year.

    Parquet files in input_parquet_dir must contain all feature columns used during
    model training. Output retains the original spatial grid, with null for rows
    that had missing feature values. Targets whose model file is missing or whose
    prediction fails are skipped with a warning.

    FinalModels are expected at: root_dir / "models" / "{target}_{schema}_final-model.pkl"

    Args:
        target:            Species name(s) to predict.
        year:              Calendar year.
        root_dir:          Root directory containing the "models" subdirectory.
        input_parquet_dir: Hive-partitioned parquet store (read via ParquetIndexer).
        schema:            Schema identifier used to locate model files.
        geo_extent:        Spatial bounding box as (xmin, ymin, xmax, ymax).
        alpha:             Miscoverage level for conformal intervals (e.g. 0.10 → 90%).
                           When set, calibrated models produce additional
                           '{target}_{schema}_pi_lower' and '_pi_upper' columns.
                           Regression: bounds in the original scale (after y-transform
                           inversion). Binary classifiers: probability-space bands
                           clip(p ± q, 0, 1). Uncalibrated models are unaffected;
                           multiclass classifiers are not supported (skipped with a
                           warning).
        local:             If True, use LocalConformalCalibration when the model has
                           one, producing interval widths that vary by location and
                           season. If False (default), use the global threshold
                           regardless of the scan's lon/lat/time columns.
        aoa:               If True, add '{target}_{schema}_di' (dissimilarity index) and
                           '{target}_{schema}_aoa' (True inside the area of applicability)
                           per target. Predictions outside the AOA are extrapolations the
                           CV estimate does not cover.

    Returns:
        DataFrame with columns ['index', 'time', 'lon', 'lat'] plus one Float32
        prediction column per succeeded target (and interval / AOA columns when
        alpha / aoa are set).

    Raises:
        ImportError: If h2mare is unavailable (indicates a broken install).
        ValueError: If root_dir or input_parquet_dir does not exist.
    """
    try:
        from h2mare.storage import ParquetIndexer
    except ImportError as e:
        raise ImportError(
            "predict_for_year requires h2mare, a core dependency of h2ml. "
            "Your install looks incomplete — reinstall with: uv add h2ml"
        ) from e

    logger.info(f"Starting predictions for year {year}")

    targets = [target] if isinstance(target, str) else list(target)
    year = str(year)

    if not root_dir.exists():
        raise ValueError(f"root_dir not found: {root_dir}")
    if not input_parquet_dir.exists():
        raise ValueError(f"input_parquet_dir not found: {input_parquet_dir}")

    model_dir = root_dir / "models"
    xmin, ymin, xmax, ymax = geo_extent

    df_original = (
        ParquetIndexer(input_parquet_dir)
        .scan(dates=(f"{year}-01-01", f"{year}-12-31"), bbox=(xmin, ymin, xmax, ymax))
        .with_row_index()
        .collect()
    )

    pred_columns: list[pl.Series] = []
    succeeded = 0
    for sp in targets:
        model_path = model_dir / f"{sp}_{schema}_final-model.pkl"
        try:
            model = FinalModel.load(model_path)
            series = _predict_single(df_original, model, col_name=f"{sp}_{schema}", alpha=alpha, local=local, aoa=aoa)
            pred_columns.extend(series)
            logger.info(f"Predicted: {sp}")
            succeeded += 1
        except FileNotFoundError:
            logger.warning(f"Model not found for '{sp}', skipping: {model_path}")
        except Exception as e:
            logger.warning(f"Prediction failed for '{sp}', skipping: {e}")

    if not pred_columns:
        logger.warning(f"No predictions generated for year {year}.")
        return df_original.select(["index", "time", "lon", "lat"])

    logger.success(f"Predictions for year {year} completed ({succeeded}/{len(targets)} targets).")
    return df_original.select(["index", "time", "lon", "lat"]).with_columns(pred_columns)


def predict_for_year_delta(
    target: str | list[str],
    year: int | str,
    root_dir: Path,
    input_parquet_dir: Path,
    schema: str,
    geo_extent: tuple[float, float, float, float],
    alpha: float = 0.10,
    local: bool = False,
    aoa: bool = False,
) -> pl.DataFrame:
    """
    Load DeltaFinalModels and generate delta predictions + conformal intervals for a calendar year.

    Delta model directories are expected at:
        root_dir / "models" / "{target}_{schema}_final-model/"

    Output columns per succeeded target:
        {target}_{schema}            — delta point prediction: P(present) × E(count | present)
        {target}_{schema}_pi_lower   — lower conformal bound (non-negative; Float32)
        {target}_{schema}_pi_upper   — upper conformal bound (Float32)

    Interval columns are omitted for models without a saved ConformalCalibration.
    Targets whose model directory is missing or whose prediction fails are skipped with a warning.

    Args:
        target:            Species name(s) to predict.
        year:              Calendar year.
        root_dir:          Root directory containing the "models" subdirectory.
        input_parquet_dir: Hive-partitioned parquet store (read via ParquetIndexer).
        schema:            Schema identifier used to locate model directories.
        geo_extent:        Spatial bounding box as (xmin, ymin, xmax, ymax).
        alpha:             Miscoverage level for conformal intervals (default 0.10 → 90%).
        local:             If True, use LocalConformalCalibration when the model has
                           one, producing interval widths that vary by location and
                           season. If False (default), use the global threshold.

    Returns:
        DataFrame with columns ['index', 'time', 'lon', 'lat'] plus prediction and
        interval columns per succeeded target.

    Raises:
        ImportError: If h2mare is unavailable (indicates a broken install).
        ValueError: If root_dir or input_parquet_dir does not exist.
    """
    try:
        from h2mare.storage import ParquetIndexer
    except ImportError as e:
        raise ImportError(
            "predict_for_year_delta requires h2mare, a core dependency of h2ml. "
            "Your install looks incomplete — reinstall with: uv add h2ml"
        ) from e

    from h2ml.pipeline.final_model import DeltaFinalModel

    logger.info(f"Starting delta predictions for year {year}")

    targets = [target] if isinstance(target, str) else list(target)
    year = str(year)

    if not root_dir.exists():
        raise ValueError(f"root_dir not found: {root_dir}")
    if not input_parquet_dir.exists():
        raise ValueError(f"input_parquet_dir not found: {input_parquet_dir}")

    model_dir = root_dir / "models"
    xmin, ymin, xmax, ymax = geo_extent

    df_original = (
        ParquetIndexer(input_parquet_dir)
        .scan(dates=(f"{year}-01-01", f"{year}-12-31"), bbox=(xmin, ymin, xmax, ymax))
        .with_row_index()
        .collect()
    )

    pred_columns: list[pl.Series] = []
    succeeded = 0
    for sp in targets:
        model_path = model_dir / f"{sp}_{schema}_final-model"
        try:
            model = DeltaFinalModel.load(model_path)
            col_name = f"{sp}_{schema}"
            series = _predict_delta_single(df_original, model, col_name=col_name, alpha=alpha, local=local, aoa=aoa)
            pred_columns.extend(series)
            interval_note = " (with intervals)" if len(series) == 3 else " (point only — no calibration)"
            logger.info(f"Predicted: {sp}{interval_note}")
            succeeded += 1
        except FileNotFoundError:
            logger.warning(f"Delta model not found for '{sp}', skipping: {model_path}")
        except Exception as e:
            logger.warning(f"Delta prediction failed for '{sp}', skipping: {e}")

    if not pred_columns:
        logger.warning(f"No delta predictions generated for year {year}.")
        return df_original.select(["index", "time", "lon", "lat"])

    logger.success(f"Delta predictions for year {year} completed ({succeeded}/{len(targets)} targets).")
    return df_original.select(["index", "time", "lon", "lat"]).with_columns(pred_columns)


def predict_map(
    model,
    indexer,
    dates: tuple,
    bbox: tuple,
    target_col: str,
    vminmax: Optional[tuple] = None,
    agg_by: Literal["month", "season"] = "month",
    save_path=None,
    mask_outside_aoa: bool = False,
) -> None:
    """
    Predict on a spatial-temporal grid and plot aggregated maps.

    Args:
        model:      FinalModel with feature_names and predict() method.
        indexer:    ParquetIndexer pointing to the feature store.
        dates:      Date range as (start, end) strings.
        bbox:       Bounding box as (xmin, ymin, xmax, ymax).
        target_col: Column name for predictions (used as plot legend title).
        vminmax:    Optional (vmin, vmax) for colormap clipping.
        agg_by:     Temporal aggregation level — 'month' or 'season'.
        save_path:  Path to save the plot; if None, calls plt.show().
        mask_outside_aoa: If True, drop predictions outside the model's area of
                    applicability before aggregating, so the map only shows cells
                    the model can speak for. Requires model.applicability.

    Raises:
        ImportError: If h2mare is unavailable (indicates a broken install).
    """
    try:
        from h2mare.storage import aggregate_by_space_time
        from h2mare.utils.plot import plot_maps
    except ImportError as e:
        raise ImportError(
            "predict_map requires h2mare (and the cartopy it pulls in), a core "
            "dependency of h2ml. Your install looks incomplete — reinstall with: uv add h2ml"
        ) from e

    df_orig = indexer.scan(dates=dates, bbox=bbox, columns=model.feature_names).with_row_index().collect()

    df_pred = df_orig.select(model.feature_names + ["index"]).drop_nulls()
    X = df_pred.drop("index").to_numpy()
    preds = model.predict(X).astype("float32")
    pred_idx = df_pred["index"]
    if mask_outside_aoa:
        inside = model.area_of_applicability(X)[1]
        logger.info(f"predict_map: {int((~inside).sum())}/{len(inside)} cells outside the AOA masked.")
        preds, pred_idx = preds[inside], pred_idx.filter(pl.Series(inside))

    full_series = pl.Series(target_col, [None] * len(df_orig), dtype=pl.Float32)
    full_series = full_series.scatter(pred_idx, preds)

    df_results = df_orig.select(["index", "time", "lon", "lat"]).with_columns(full_series)
    df_agg = aggregate_by_space_time(df_results, vars_name=target_col, agg_by=agg_by)
    df_plot = df_agg.collect() if isinstance(df_agg, pl.LazyFrame) else df_agg

    plot_maps(
        df_plot,
        var_name=target_col,
        agg_by=agg_by,
        legend_title=target_col,
        vminmax=vminmax,
        save_path=save_path,
    )
