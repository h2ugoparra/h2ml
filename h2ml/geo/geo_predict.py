"""
Geo-spatial prediction utilities bridging h2ml FinalModels and h2mare ParquetIndexer.

Requires the [geo] optional dependencies:
    uv pip install h2ml[geo]
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


def _predict_single(
    df_original: pl.DataFrame,
    model: FinalModel,
    col_name: str,
    alpha: Optional[float] = None,
    local: bool = False,
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
                     Multiclass classifiers are unaffected.
        local:       If True, use LocalConformalCalibration when available to produce
                     spatially/temporally varying interval widths.
                     If False (default), always use the global conformal threshold.

    Returns:
        (pred_series,) when alpha is None, uncalibrated, or multiclass classifier.
        (pred_series, lower_series, upper_series) for calibrated regression or
        calibrated binary classifiers with alpha set.
    """
    geo_cols = [c for c in ["lon", "lat", "time"] if c in df_original.columns]
    df_clean = df_original.select(model.feature_names + ["index"] + geo_cols).drop_nulls()
    X = df_clean.select(model.feature_names).to_numpy()
    idx = df_clean["index"]
    n = len(df_original)

    coords = df_clean.select(["lon", "lat"]).to_numpy() if "lon" in geo_cols and "lat" in geo_cols else None
    times = df_clean["time"].cast(pl.Utf8).to_numpy() if "time" in geo_cols else None

    if model.task_type == TaskType.CLASSIFICATION:
        preds = model.predict_proba(X).astype(np.float32)
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
            return pred_series, lower_series, upper_series
        return (pred_series,)

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
        return pred_series, lower_series, upper_series

    return (pred_series,)


def _predict_delta_single(
    df_original: pl.DataFrame,
    model: "DeltaFinalModel",
    col_name: str,
    alpha: Optional[float] = None,
    local: bool = False,
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

    Returns:
        (pred_series,) or (pred_series, lower_series, upper_series).
    """
    clf_features = model.clf.feature_names
    reg_features = model.reg.feature_names
    all_features = list(dict.fromkeys(clf_features + reg_features))

    geo_cols = [c for c in ["lon", "lat", "time"] if c in df_original.columns]
    df_clean = df_original.select(all_features + ["index"] + geo_cols).drop_nulls()
    idx = df_clean["index"]
    n = len(df_original)

    coords = df_clean.select(["lon", "lat"]).to_numpy() if "lon" in geo_cols and "lat" in geo_cols else None
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

    if alpha is not None and model.conformal is not None:
        if local and model.local_conformal is not None and (coords is not None or times is not None):
            q = model.local_conformal.threshold(alpha, coords=coords, times=times)
        else:
            q = float(model.conformal.threshold(alpha))
        lower = np.maximum(0.0, delta - q).astype(np.float32)
        upper = (delta + q).astype(np.float32)
        lower_series = pl.Series(f"{col_name}_pi_lower", [None] * n, dtype=pl.Float32).scatter(idx, lower)
        upper_series = pl.Series(f"{col_name}_pi_upper", [None] * n, dtype=pl.Float32).scatter(idx, upper)
        return pred_series, lower_series, upper_series

    return (pred_series,)


def predict_for_year(
    target: str | list[str],
    year: int | str,
    root_dir: Path,
    input_parquet_dir: Path,
    schema: str,
    geo_extent: tuple[float, float, float, float],
    alpha: Optional[float] = None,
    local: bool = False,
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
                           clip(p ± q, 0, 1). Multiclass and uncalibrated models
                           are unaffected.
        local:             If True, use LocalConformalCalibration when the model has
                           one, producing interval widths that vary by location and
                           season. If False (default), use the global threshold
                           regardless of the scan's lon/lat/time columns.

    Returns:
        DataFrame with columns ['index', 'time', 'lon', 'lat'] plus one Float32
        prediction column per succeeded target (and interval columns when alpha is set).

    Raises:
        ImportError: If the [geo] extras are not installed.
        ValueError: If root_dir or input_parquet_dir does not exist.
    """
    try:
        from h2mare.storage import ParquetIndexer
    except ImportError as e:
        raise ImportError("predict_for_year requires the [geo] extras. Install with: uv pip install h2ml[geo]") from e

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
            series = _predict_single(df_original, model, col_name=f"{sp}_{schema}", alpha=alpha, local=local)
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
        ImportError: If the [geo] extras are not installed.
        ValueError: If root_dir or input_parquet_dir does not exist.
    """
    try:
        from h2mare.storage import ParquetIndexer
    except ImportError as e:
        raise ImportError(
            "predict_for_year_delta requires the [geo] extras. Install with: uv pip install h2ml[geo]"
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
            series = _predict_delta_single(df_original, model, col_name=col_name, alpha=alpha, local=local)
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

    Raises:
        ImportError: If the [geo] extras are not installed.
    """
    try:
        from h2mare.storage import aggregate_by_space_time
        from h2mare.utils.plot import plot_maps
    except ImportError as e:
        raise ImportError("predict_map requires the [geo] extras. Install with: uv pip install h2ml[geo]") from e

    df_orig = indexer.scan(dates=dates, bbox=bbox, columns=model.feature_names).with_row_index().collect()

    df_pred = df_orig.select(model.feature_names + ["index"]).drop_nulls()
    preds = model.predict(df_pred.drop("index").to_numpy()).astype("float32")

    full_series = pl.Series(target_col, [None] * len(df_orig), dtype=pl.Float32)
    full_series = full_series.scatter(df_pred["index"], preds)

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
