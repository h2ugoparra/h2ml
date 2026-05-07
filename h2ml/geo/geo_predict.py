"""
Geo-spatial prediction utilities bridging h2ml FinalModels and h2mare ParquetIndexer.

Requires the [geo] optional dependencies:
    uv pip install h2ml[geo]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

import numpy as np
import polars as pl
from loguru import logger

from h2ml.pipeline.base import TaskType
from h2ml.pipeline.final_model import FinalModel
from h2ml.preprocessing.transforms import INVERSE_TRANSFORMS


def _predict_single(
    df_original: pl.DataFrame,
    model: FinalModel,
    col_name: str,
) -> pl.Series:
    """
    Generate a prediction Series for one target using pre-loaded feature data.

    Rows with any null feature value are excluded from inference and appear as
    null in the returned Series, preserving alignment with the original grid.

    Args:
        df_original: Full feature DataFrame with a row 'index' column.
        model:       Loaded FinalModel (handles scaling and task type internally).
        col_name:    Name for the returned Series.

    Returns:
        Float32 Series of length len(df_original), nulls where features were missing.
    """
    df_clean = df_original.select(model.feature_names + ["index"]).drop_nulls()
    X = df_clean.drop("index").to_numpy()

    if model.task_type == TaskType.CLASSIFICATION:
        preds = model.predict_proba(X)
    else:
        preds = model.predict(X)
        if model.y_transform is not None:
            inverse_fn = INVERSE_TRANSFORMS.get(model.y_transform)
            if inverse_fn is not None:
                preds = inverse_fn(preds)

    preds = preds.astype(np.float32)
    series = pl.Series(col_name, [None] * len(df_original), dtype=pl.Float32)
    return series.scatter(df_clean["index"], preds)


def predict_for_year(
    target: str | list[str],
    year: int | str,
    root_dir: Path,
    input_parquet_dir: Path,
    schema: str,
    geo_extent: tuple[float, float, float, float],
) -> pl.DataFrame:
    """
    Load pre-trained FinalModels and generate predictions for a full calendar year.

    Parquet files in input_parquet_dir must contain all feature columns used during
    model training. Output retains the original spatial grid, with null for rows
    that had missing feature values. Targets whose model file is missing or whose
    prediction fails are skipped with a warning.

    FinalModels are expected at: root_dir / schema / "models" / "{target}_final_model.pkl"

    Args:
        target:            Species name(s) to predict.
        year:              Calendar year.
        root_dir:          Root directory containing schema subdirectories.
        input_parquet_dir: Hive-partitioned parquet store (read via ParquetIndexer).
        schema:            Schema identifier used to locate model files.
        geo_extent:        Spatial bounding box as (xmin, ymin, xmax, ymax).

    Returns:
        DataFrame with columns ['index', 'time', 'lon', 'lat'] plus one Float32
        prediction column per succeeded target named '{target}_{schema}'.
    """
    try:
        from h2mare.storage import ParquetIndexer
    except ImportError as e:
        raise ImportError(
            "predict_for_year requires the [geo] extras. "
            "Install with: uv pip install h2ml[geo]"
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

    pred_columns = []
    for sp in targets:
        model_path = model_dir / f"{sp}_{schema}_final-model.pkl"
        try:
            model = FinalModel.load(model_path)
            series = _predict_single(df_original, model, col_name=f"{sp}_{schema}")
            pred_columns.append(series)
            logger.info(f"Predicted: {sp}")
        except FileNotFoundError:
            logger.warning(f"Model not found for '{sp}', skipping: {model_path}")
        except Exception as e:
            logger.warning(f"Prediction failed for '{sp}', skipping: {e}")

    if not pred_columns:
        logger.warning(f"No predictions generated for year {year}.")
        return df_original.select(["index", "time", "lon", "lat"])

    logger.success(
        f"Predictions for year {year} completed ({len(pred_columns)}/{len(targets)} targets)."
    )
    return df_original.select(["index", "time", "lon", "lat"]).with_columns(
        pred_columns
    )


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
    """
    try:
        from h2mare.storage import aggregate_by_space_time
        from h2mare.utils.plot import plot_maps
    except ImportError as e:
        raise ImportError(
            "predict_map requires the [geo] extras. "
            "Install with: uv pip install h2ml[geo]"
        ) from e

    df_orig = (
        indexer.scan(dates=dates, bbox=bbox, columns=model.feature_names)
        .with_row_index()
        .collect()
    )

    df_pred = df_orig.select(model.feature_names + ["index"]).drop_nulls()
    preds = model.predict(df_pred.drop("index").to_numpy()).astype("float32")

    full_series = pl.Series(target_col, [None] * len(df_orig), dtype=pl.Float32)
    full_series = full_series.scatter(df_pred["index"], preds)

    df_results = df_orig.select(["index", "time", "lon", "lat"]).with_columns(
        full_series
    )
    df_plot = aggregate_by_space_time(df_results, vars_name=target_col, agg_by=agg_by)

    plot_maps(
        df_plot,
        var_name=target_col,
        agg_by=agg_by,
        legend_title=target_col,
        vminmax=vminmax,
        save_path=save_path,
    )
