"""
h2ml/evaluation/compare.py

compare_results() — compare multiple PipelineResult objects in one DataFrame.

Each row represents one pipeline run. Useful for selecting the best configuration
across runs with different PipelineConfig settings, feature schemas, or datasets.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from h2ml.pipeline.pipeline import _MINIMIZE

if TYPE_CHECKING:
    from h2ml.pipeline.pipeline import PipelineResult


def compare_results(
    results: list["PipelineResult"],
    labels: Optional[list[str]] = None,
    metric: Optional[str] = None,
    n_folds: Optional[int] = None,
) -> pd.DataFrame:
    """
    Summarise multiple PipelineResult objects as a single comparison DataFrame.

    One row per result. Columns:

        Run                — label (from labels, or Run_0 / Run_1 / …)
        Metric             — short metric name used for Score_Mean (e.g. "AUC", "R2")
        Best_Model         — winning model class name
        Best_Stage         — default | reduced | optimized
        Y_Transform        — winning y-transform, or None
        Score_Mean         — mean CV score for the selected metric
        Score_Std          — fold std for the selected metric
        Conservative_Bound — variance-penalised score used for ranking. Direction is
                             derived automatically from the metric name via _MINIMIZE:
                               higher-is-better: Score_Mean - Score_Std / sqrt(n_folds)
                               lower-is-better:  Score_Mean + Score_Std / sqrt(n_folds)
                             A model with high mean but high variance is pulled toward
                             a worse value, so stable models are preferred over
                             optimistically-estimated ones.
        Brier_Mean         — Brier_Test_Mean from the winning stage's agg_df
                             (classification only; NaN otherwise)
        OOF_Brier          — Brier score computed on the assembled OOF predictions
                             (more robust than the fold-averaged Brier_Mean;
                             classification only; NaN otherwise)
        N_Features         — feature count in the winning feature stage
        Completed_Steps    — list of pipeline steps that finished

    Args:
        results: List of PipelineResult objects to compare.
        labels:  Optional run labels (same length as results).
                 Defaults to "Run_0", "Run_1", …
        metric:  Short metric name to use as the common comparison column
                 (e.g. "AUC", "R2", "RMSE"). When provided, Score_Mean and
                 Score_Std are read from the ``<metric>_Test_Mean/Std`` columns
                 of each result's winning-stage agg DataFrame, making
                 cross-run scores directly comparable even when runs were
                 optimised on different metrics. When omitted, Score_Mean falls
                 back to each result's ``best_model_value``. Minimisation
                 direction and sort order are derived automatically from
                 ``_MINIMIZE`` — no need to pass ``ascending`` manually.
        n_folds: Number of CV folds used in all runs. When provided it overrides
                 automatic inference from the fold DataFrames, which can fail if
                 results were loaded from disk or the fold DataFrames are absent.
                 Conservative_Bound cannot be computed without a fold count and
                 will fall back to Score_Mean if both this argument and
                 inference fail.

    Returns:
        DataFrame sorted by Conservative_Bound (falls back to Score_Mean when
        fold count cannot be inferred), one row per result.

    Example:
        >>> r1 = pipeline_a.run(store)
        >>> r2 = pipeline_b.run(store)
        >>> compare_results([r1, r2], labels=["baseline", "spatial_cv"], metric="AUC")
    """
    if labels is not None and len(labels) != len(results):
        raise ValueError(f"labels has {len(labels)} entries but results has {len(results)}.")

    if metric is None:
        unique_metrics = {r.metric for r in results if r.metric is not None}
        if len(unique_metrics) > 1:
            raise ValueError(
                f"compare_results: results use different metrics {sorted(unique_metrics)} — "
                "Score_Mean values are not directly comparable. "
                "Pass metric= to select a common comparison metric."
            )

    rows = []
    for i, result in enumerate(results):
        label = labels[i] if labels is not None else f"Run_{i}"
        fold_count = n_folds if n_folds is not None else _infer_n_folds(result)

        if metric is not None:
            score_mean, score_std = _get_score_for_metric(result, metric)
            display_metric = metric
        else:
            score_mean = result.best_model_value
            score_std = result.best_model_std
            display_metric = result.metric

        minimize = _MINIMIZE.get(display_metric, False) if display_metric else False

        conservative_bound: Optional[float] = None
        if score_mean is not None and score_std is not None and fold_count:
            penalty = score_std / np.sqrt(fold_count)
            conservative_bound = score_mean + penalty if minimize else score_mean - penalty

        rows.append(
            {
                "Run": label,
                "Metric": display_metric,
                "Best_Model": result.best_model_name,
                "Best_Stage": result.best_stage,
                "Y_Transform": result.y_transform,
                "Score_Mean": score_mean,
                "Score_Std": score_std,
                "Conservative_Bound": conservative_bound,
                "Brier_Mean": _get_brier_from_agg(result),
                "OOF_Brier": _get_oof_brier(result),
                "N_Features": _get_n_features(result),
                "Completed_Steps": result.completed_steps,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort direction derived from metric, not from an ascending argument
    if metric is not None:
        sort_ascending = _MINIMIZE.get(metric, False)
    else:
        unique_minimize = {_MINIMIZE.get(r.metric, False) for r in results if r.metric}
        sort_ascending = unique_minimize.pop() if len(unique_minimize) == 1 else False

    if df["Conservative_Bound"].isna().all():
        logger.warning(
            "compare_results: Conservative_Bound could not be computed for any result "
            "(fold count unknown and inference failed). Sorting by Score_Mean instead. "
            "Pass n_folds= to enable variance penalisation."
        )
        sort_col = "Score_Mean"
    else:
        sort_col = "Conservative_Bound"

    if sort_col in df.columns and df[sort_col].notna().any():
        df = df.sort_values(sort_col, ascending=sort_ascending, ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_score_for_metric(
    result: "PipelineResult",
    metric: str,
) -> tuple[Optional[float], Optional[float]]:
    """Return (mean, std) for *metric* from the winning stage's agg DataFrame."""
    mean_col = f"{metric}_Test_Mean"
    std_col = f"{metric}_Test_Std"

    stage = result.best_stage
    if stage == "optimized":
        agg_df = result.step4_agg_df
    elif stage == "reduced":
        agg_df = result.step3_agg_df
    else:
        agg_df = result.step1_agg_df

    if agg_df is None or mean_col not in agg_df.columns or agg_df.empty:
        logger.warning(
            f"compare_results: column '{mean_col}' not found in the winning-stage "
            f"agg DataFrame — Score_Mean will be NaN for this result."
        )
        return None, None

    mask = agg_df["Model"] == result.best_model_name
    if result.y_transform and "Y_transform" in agg_df.columns:
        mask = mask & (agg_df["Y_transform"] == result.y_transform)
    if stage == "reduced" and "Stage" in agg_df.columns:
        mask = mask & (agg_df["Stage"] == "reduced")

    subset = agg_df.loc[mask]
    if subset.empty:
        return None, None

    row = subset.iloc[0]
    mean_val = float(row[mean_col])
    std_val = float(row[std_col]) if std_col in subset.columns else None
    return mean_val, std_val


def _infer_n_folds(result: "PipelineResult") -> Optional[int]:
    """Infer fold count, preferring the stored splitter to avoid under-counting when folds fail."""
    if result.splitter is not None and hasattr(result.splitter, "n_splits"):
        return result.splitter.n_splits
    for df in (result.step1_fold_df, result.step3_fold_df, result.step4_fold_df):
        if df is not None and "Fold" in df.columns and not df.empty:
            return int(df["Fold"].max()) + 1
    return None


def _get_n_features(result: "PipelineResult") -> Optional[int]:
    """Return the feature count for the winning feature stage."""
    feature_stage = result.best_feature_stage or result.best_stage
    if feature_stage == "reduced" and result.features_reduced is not None:
        return result.features_reduced.n_features
    if result.features is not None:
        return result.features.n_features
    return None


def _get_brier_from_agg(result: "PipelineResult") -> Optional[float]:
    """
    Extract Brier_Test_Mean for the winning (model, stage, transform) from
    whichever agg_df corresponds to the best_stage.
    Returns None when Brier was not computed or the column is absent.
    """
    stage = result.best_stage

    if stage == "optimized":
        df = result.step4_agg_df
        if df is not None and "Brier_Test_Mean" in df.columns and not df.empty:
            return float(df.iloc[0]["Brier_Test_Mean"])
        return None

    df = result.step3_agg_df if stage == "reduced" else result.step1_agg_df
    if df is None or "Brier_Test_Mean" not in df.columns:
        return None

    mask = df["Model"] == result.best_model_name
    if result.y_transform and "Y_transform" in df.columns:
        mask = mask & (df["Y_transform"] == result.y_transform)
    if stage == "reduced" and "Stage" in df.columns:
        mask = mask & (df["Stage"] == "reduced")

    subset = df.loc[mask, "Brier_Test_Mean"]
    return float(subset.iloc[0]) if not subset.empty else None


def _get_oof_brier(result: "PipelineResult") -> Optional[float]:
    """
    Compute Brier score on assembled OOF predictions.
    Only meaningful for classification — returns None for regression or when
    OOF data is unavailable.
    """
    from sklearn.metrics import brier_score_loss
    from h2ml.pipeline.base import TaskType

    cv = result.best_cv_result
    if cv is None or cv.task_type != TaskType.CLASSIFICATION:
        return None

    oof_pred = result.oof_predictions
    oof_labels = result.oof_labels
    if oof_pred is None or oof_labels is None:
        return None
    # Drop missing positions (failed folds); use pd.isnull to handle both
    # float NaN (numeric labels) and None (string/object labels from oof_labels)
    valid = ~(np.isnan(oof_pred) if oof_pred.ndim == 1 else np.isnan(oof_pred).any(axis=1))
    valid &= ~pd.isnull(oof_labels)
    if not valid.any():
        return None
    y_true = oof_labels[valid]
    y_prob = oof_pred[valid]
    if y_prob.ndim == 2:
        # Multiclass Brier
        from sklearn.preprocessing import label_binarize

        classes = np.unique(y_true.astype(int))
        y_bin = label_binarize(y_true, classes=classes)
        return float(np.mean(np.sum((y_bin - y_prob) ** 2, axis=1)))
    return float(brier_score_loss(y_true, y_prob))
