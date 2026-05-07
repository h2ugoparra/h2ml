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
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shutil
from loguru import logger

from h2ml.features.feature_store import PipelineData


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
        "cv_warnings",
        "metric",
        # parquet
        "step1_fold_df",
        "step1_agg_df",
        "step3_fold_df",
        "step3_agg_df",
        "step4_fold_df",
        "step4_agg_df",
        # feature stores
        "features",
        "features_reduced",
        # pickles
        "selector",
        "step1_cv_result",
        "step3_cv_result",
        "step4_cv_result",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_result(result, path: str | Path) -> None:
    """
    Persist a PipelineResult to *path* (created if it does not exist).

    Args:
        result: PipelineResult instance.
        path:   Destination directory.
    """
    root = Path(path)
    if root.exists():
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
            "cv_warnings": result.cv_warnings,
            "metric": result.metric,
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
    }
    for name, df in _df_map.items():
        if df is not None:
            df.to_parquet(root / f"{name}.parquet", index=False)

    # PipelineDatas
    if result.features is not None:
        _save_feature_store(result.features, root / "features")
    if result.features_reduced is not None:
        _save_feature_store(result.features_reduced, root / "features_reduced")

    # FeatureSelector + CV results
    if result.selector is not None:
        joblib.dump(result.selector, root / "selector.pkl")
    if result.step1_cv_result:
        joblib.dump(result.step1_cv_result, root / "step1_cv_result.pkl")
    if result.step3_cv_result:
        joblib.dump(result.step3_cv_result, root / "step3_cv_result.pkl")
    if result.step4_cv_result is not None:
        joblib.dump(result.step4_cv_result, root / "step4_cv_result.pkl")

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
    }

    return PipelineResult(
        features=_load_feature_store(root / "features"),
        features_reduced=_load_feature_store(root / "features_reduced"),
        selector=_load_pkl(root / "selector.pkl"),
        step1_cv_result=_load_pkl(root / "step1_cv_result.pkl"),
        step3_cv_result=_load_pkl(root / "step3_cv_result.pkl"),
        step4_cv_result=_load_pkl(root / "step4_cv_result.pkl"),
        best_model_name=meta.get("best_model_name"),
        best_model_value=meta.get("best_model_value"),
        best_model_std=meta.get("best_model_std"),
        best_stage=meta.get("best_stage"),
        best_feature_stage=meta.get("best_feature_stage"),
        best_params=meta.get("best_params"),
        y_transform=meta.get("y_transform"),
        cv_type=meta.get("cv_type", "random"),
        cv_warnings=meta.get("cv_warnings", []),
        metric=meta.get("metric"),
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
    return PipelineData(
        X=np.load(path / "X.npy"),
        y=np.load(path / "y.npy"),
        y_true=np.load(y_true_path) if y_true_path.exists() else None,
        coords=np.load(coords_path) if coords_path.exists() else None,
        feature_names=meta.get("feature_names", []),
        y_transform=meta.get("y_transform"),
    )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays from Optuna params."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
