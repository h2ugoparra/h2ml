"""
h2ml/persistence/result_io.py

Save and load PipelineResult to/from a directory.

Layout
------
<root>/
    metadata.json           — scalars: best_model_name, best_model_value, best_model_std,
                              best_stage, best_feature_stage, best_params, y_transform
    step1_fold.parquet
    step1_agg.parquet
    step3_fold.parquet
    step3_agg.parquet
    step4_fold.parquet
    step4_agg.parquet
    features/               — raw PipelineData  (X, y, y_true, meta)
    features_reduced/       — reduced PipelineData
    selector.pkl            — fitted FeatureSelector (joblib)
    step1_cv_result.pkl    — list[CVResult]  (joblib)
    step3_cv_result.pkl    — list[CVResult]
    step4_cv_result.pkl     — CVResult
    holdout_agg.parquet     — holdout metrics (when a holdout was scored)
    holdout/                — raw holdout PipelineData
    holdout_idx.npy         — holdout row indices into the store passed to run()
    holdout_cv_result.pkl   — one-fold CVResult on the holdout
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from h2ml.core.feature_store import PipelineData
from h2ml.evaluation.metrics import rename_deprecated_columns, resolve_metric

# ---------------------------------------------------------------------------
# Field registry — keep in sync with PipelineResult
# ---------------------------------------------------------------------------

# Fields intentionally excluded from persistence (transient / derived / expensive).
_UNSAVED_FIELDS = frozenset(
    {
        "step3_reduced_stores",  # step-4 cache; rebuilt from selector on demand
        "splitter",  # rebuilt from coords on demand
        "_final_shap_cache",  # plot cache; not needed across sessions
    }
)

_SAVED_FIELDS = frozenset(
    {
        # metadata.json
        "best_model_name",
        "best_model_value",
        "best_model_std",
        "best_stage",
        "best_feature_stage",
        "best_params",
        "y_transform",
        "cv_type",
        "spatial_cv_metric",
        "time_bin_resolution",
        "cv_warnings",
        "class_weight_balanced",
        "best_requires_scaling",
        "metric",
        "holdout_strategy",
        "holdout_refit",
        # parquet
        "step1_fold_df",
        "step1_agg_df",
        "step3_fold_df",
        "step3_agg_df",
        "step4_fold_df",
        "step4_agg_df",
        "holdout_df",
        # feature stores
        "features",
        "features_reduced",
        "holdout_store",
        # npy
        "holdout_idx",
        # pickles
        "selector",
        "best_base_estimator",
        "step1_cv_result",
        "step3_cv_result",
        "step4_cv_result",
        "holdout_cv_result",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_result(result, path: str | Path) -> None:
    """
    Persist a PipelineResult to *path* (created if it does not exist).

    An existing path is only overwritten when it is an empty directory or a
    previous save (contains metadata.json) — anything else is refused so a
    mistyped path cannot wipe an unrelated directory tree.

    Args:
        result: PipelineResult instance.
        path:   Destination directory.

    Raises:
        ValueError: If *path* exists but is not an empty directory or a
            previously saved result.
    """
    root = Path(path)
    if root.exists():
        is_prior_result = root.is_dir() and (root / "metadata.json").exists()
        is_empty_dir = root.is_dir() and not any(root.iterdir())
        if not (is_prior_result or is_empty_dir):
            raise ValueError(
                f"save_result: refusing to overwrite '{root}' — it exists but is not an "
                "empty directory or a previously saved result (no metadata.json). "
                "Delete it manually or choose another path."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True)

    # Scalars
    _save_json(
        {
            "best_model_name": result.best_model_name,
            "best_model_value": result.best_model_value,
            "best_model_std": result.best_model_std,
            "best_stage": result.best_stage,
            "best_feature_stage": result.best_feature_stage,
            "best_params": result.best_params,
            "y_transform": result.y_transform,
            "cv_type": result.cv_type,
            "spatial_cv_metric": result.spatial_cv_metric,
            "time_bin_resolution": result.time_bin_resolution,
            "cv_warnings": result.cv_warnings,
            "class_weight_balanced": result.class_weight_balanced,
            "best_requires_scaling": result.best_requires_scaling,
            "metric": result.metric,
            "holdout_strategy": result.holdout_strategy,
            "holdout_refit": result.holdout_refit,
        },
        root / "metadata.json",
    )

    # DataFrames
    _df_map = {
        "step1_fold": result.step1_fold_df,
        "step1_agg": result.step1_agg_df,
        "step3_fold": result.step3_fold_df,
        "step3_agg": result.step3_agg_df,
        "step4_fold": result.step4_fold_df,
        "step4_agg": result.step4_agg_df,
        "holdout_agg": result.holdout_df,
    }
    for name, df in _df_map.items():
        if df is not None:
            df.to_parquet(root / f"{name}.parquet", index=False)

    # PipelineDatas
    if result.features is not None:
        _save_feature_store(result.features, root / "features")
    if result.features_reduced is not None:
        _save_feature_store(result.features_reduced, root / "features_reduced")
    if result.holdout_store is not None:
        _save_feature_store(result.holdout_store, root / "holdout")
    if result.holdout_idx is not None:
        np.save(root / "holdout_idx.npy", result.holdout_idx)

    # FeatureSelector + CV results
    if result.selector is not None:
        joblib.dump(result.selector, root / "selector.pkl")
    if result.best_base_estimator is not None:
        joblib.dump(result.best_base_estimator, root / "best_base_estimator.pkl")
    if result.step1_cv_result:
        joblib.dump(result.step1_cv_result, root / "step1_cv_result.pkl")
    if result.step3_cv_result:
        joblib.dump(result.step3_cv_result, root / "step3_cv_result.pkl")
    if result.step4_cv_result is not None:
        joblib.dump(result.step4_cv_result, root / "step4_cv_result.pkl")
    if result.holdout_cv_result is not None:
        joblib.dump(result.holdout_cv_result, root / "holdout_cv_result.pkl")

    # Guard: warn if PipelineResult has fields that aren't in _SAVED_FIELDS or
    # _UNSAVED_FIELDS — catches new fields added without updating this module.
    all_fields = {f.name for f in dataclasses.fields(result)}
    unaccounted = all_fields - _SAVED_FIELDS - _UNSAVED_FIELDS
    if unaccounted:
        logger.warning(
            f"save_result: PipelineResult fields {sorted(unaccounted)} are not persisted "
            "and will be lost on reload. Add them to result_io.py or to _UNSAVED_FIELDS."
        )


def load_result(path: str | Path):
    """
    Reload a PipelineResult saved by :func:`save_result`.

    Args:
        path: Directory previously written by save_result().

    Returns:
        PipelineResult with all available fields restored.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    from h2ml.pipeline.pipeline import (
        PipelineResult,
    )  # local import — avoids circular dep

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"No saved result found at '{root}'.")

    meta = _load_json(root / "metadata.json") or {}

    dfs = {
        "step1_fold_df": _load_parquet(root / "step1_fold.parquet"),
        "step1_agg_df": _load_parquet(root / "step1_agg.parquet"),
        "step3_fold_df": _load_parquet(root / "step3_fold.parquet"),
        "step3_agg_df": _load_parquet(root / "step3_agg.parquet"),
        "step4_fold_df": _load_parquet(root / "step4_fold.parquet"),
        "step4_agg_df": _load_parquet(root / "step4_agg.parquet"),
        "holdout_df": _load_parquet(root / "holdout_agg.parquet"),
    }
    # Results saved before a metric rename keep the old column names on disk.
    dfs = {k: None if df is None else rename_deprecated_columns(df) for k, df in dfs.items()}
    metric = meta.get("metric")
    holdout_idx_path = root / "holdout_idx.npy"

    return PipelineResult(
        features=_load_feature_store(root / "features"),
        features_reduced=_load_feature_store(root / "features_reduced"),
        selector=_load_pkl(root / "selector.pkl"),
        best_base_estimator=_load_pkl(root / "best_base_estimator.pkl"),
        step1_cv_result=_load_pkl(root / "step1_cv_result.pkl"),
        step3_cv_result=_load_pkl(root / "step3_cv_result.pkl"),
        step4_cv_result=_load_pkl(root / "step4_cv_result.pkl"),
        holdout_store=_load_feature_store(root / "holdout"),
        holdout_idx=np.load(holdout_idx_path) if holdout_idx_path.exists() else None,
        holdout_cv_result=_load_pkl(root / "holdout_cv_result.pkl"),
        holdout_strategy=meta.get("holdout_strategy"),
        holdout_refit=meta.get("holdout_refit", "dev"),
        best_model_name=meta.get("best_model_name"),
        best_model_value=meta.get("best_model_value"),
        best_model_std=meta.get("best_model_std"),
        best_stage=meta.get("best_stage"),
        best_feature_stage=meta.get("best_feature_stage"),
        best_params=meta.get("best_params"),
        y_transform=meta.get("y_transform"),
        cv_type=meta.get("cv_type", "random"),
        spatial_cv_metric=meta.get("spatial_cv_metric", "euclidean"),
        time_bin_resolution=meta.get("time_bin_resolution", "month"),
        cv_warnings=meta.get("cv_warnings", []),
        class_weight_balanced=meta.get("class_weight_balanced", False),
        best_requires_scaling=meta.get("best_requires_scaling", False),
        metric=None if metric is None else resolve_metric(metric),
        step3_reduced_stores=None,
        **dfs,
    )


# ---------------------------------------------------------------------------
# PipelineData helpers
# ---------------------------------------------------------------------------


def _save_feature_store(store: PipelineData, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "X.npy", store.X)
    np.save(path / "y.npy", store.y)
    if store.y_true is not None:
        np.save(path / "y_true.npy", store.y_true)
    if store.coords is not None:
        np.save(path / "coords.npy", store.coords)
    if store.times is not None:
        np.save(path / "times.npy", np.asarray(store.times, dtype="datetime64[D]"))
    _save_json(
        {"feature_names": store.feature_names, "y_transform": store.y_transform},
        path / "meta.json",
    )


def _load_feature_store(path: Path) -> Optional[PipelineData]:
    if not path.exists():
        return None
    meta = _load_json(path / "meta.json") or {}
    y_true_path = path / "y_true.npy"
    coords_path = path / "coords.npy"
    times_path = path / "times.npy"
    return PipelineData(
        X=np.load(path / "X.npy"),
        y=np.load(path / "y.npy"),
        y_true=np.load(y_true_path) if y_true_path.exists() else None,
        coords=np.load(coords_path) if coords_path.exists() else None,
        times=np.load(times_path) if times_path.exists() else None,
        feature_names=meta.get("feature_names", []),
        y_transform=meta.get("y_transform"),
    )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays from Optuna params."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_parquet(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_parquet(path) if path.exists() else None


def _load_pkl(path: Path):
    return joblib.load(path) if path.exists() else None
