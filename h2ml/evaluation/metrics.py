"""
Consumes CVResult objects and produces metrics DataFrames.
Responsibilities:
    - Compute per-fold metrics (classification or regression)
    - Aggregate fold metrics (mean, std across folds)
    - Attach domain metadata (species, setting, batch, schema)
    - Detect overfitting gap between train and test metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    # Classification
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
    # Regression
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from h2ml.core.base import TaskType
from h2ml.core.cv_result import CVResult, FoldResult

# ---------------------------------------------------------------------------
# Metric metadata — shared by PipelineConfig (selection/HPO) and compare_results
# ---------------------------------------------------------------------------

# Short metric name → full aggregated-DataFrame column name
METRIC_COL: dict[str, str] = {
    # Classification
    "AUC": "AUC_Test_Mean",
    "AUC_PR": "AUC_PR_Test_Mean",
    "LogLoss": "LogLoss_Test_Mean",
    "F1": "F1_Test_Mean",
    "Brier": "Brier_Test_Mean",
    # Regression
    "R2": "R2_Test_Mean",
    "MAE": "MAE_Test_Mean",
    "RMSE": "RMSE_Test_Mean",
}

# Whether each metric should be minimised (True) or maximised (False)
METRIC_MINIMIZE: dict[str, bool] = {
    "AUC": False,
    "AUC_PR": False,
    "F1": False,
    "Brier": True,
    "LogLoss": True,
    "R2": False,
    "MAE": True,
    "RMSE": True,
}

# ---------------------------------------------------------------------------
# Metadata container — replaces PipelineSpecs fields in results
# ---------------------------------------------------------------------------


@dataclass
class RunMetadata:
    """
    Domain-specific context attached to results after CV.
    Keeps metadata out of the CV engine.

    Attributes:
        schema:      Top-level folder / feature schema / experiment name.
        target:      Target column name.
        stage:       Pipeline stage that produced the row, set automatically:
                       - "default":   step 1 (all features, default params)
                       - "reduced":   step 3 (reduced features, default params)
                       - "optimized": step 4 (reduced or full features, optimized params)
        batch:       Optional batch/run identifier (run ID, date, or any grouping label).
        y_transform: Winning y-transform name, set by the transform sweep. None otherwise.
        notes:       Free text — useful for an experiments log.
    """

    # Experiment ID
    schema: Optional[str] = None
    target: Optional[str] = None

    # Pipeline stage — set automatically by pipeline._metadata_with()
    stage: str = "default"  # "default" | "reduced" | "optimized"

    # Optional grouping
    batch: Optional[str] = None  # run ID, date, or any grouping label
    y_transform: Optional[str] = None  # y-transform name, set by run_transform_sweep()
    notes: Optional[str] = None  # free text — useful for experiments log

    def to_dict(self) -> dict:
        """Non-None fields as a dict with capitalised keys, for DataFrame columns."""
        return {k.capitalize(): v for k, v in self.__dict__.items() if v is not None}


# ---------------------------------------------------------------------------
# Per-fold metric rows
# ---------------------------------------------------------------------------


def _classification_metrics(fold: FoldResult) -> dict:
    """Compute classification metrics for a single fold.

    Supports both binary and multiclass targets. Detection is based on the shape
    of y_prob_test: 1-D → binary, 2-D → multiclass (OvR weighted averages).

    Degenerate folds (only one class in y_test or y_train) arise with spatial CV
    on imbalanced targets. Rank-based metrics (AUC, AUC-PR) return NaN for those
    folds; other metrics use explicit class labels to avoid ValueError.
    """
    if fold.y_prob_test is None or fold.y_prob_train is None:
        raise ValueError(f"[{fold.model_name}] Fold {fold.fold_id} has no probabilities. Did you use a BaseClassifier?")

    is_multiclass = fold.y_prob_test.ndim == 2

    # All classes seen across both splits — needed to keep metric calls stable
    # even when one split is missing a class (common with spatial CV + imbalance).
    all_classes = sorted(np.unique(np.concatenate([fold.y_train, fold.y_test])).tolist())
    single_class_test = len(np.unique(fold.y_test)) < 2
    single_class_train = len(np.unique(fold.y_train)) < 2

    if is_multiclass:
        from sklearn.preprocessing import label_binarize

        y_test_bin = label_binarize(fold.y_test, classes=all_classes)
        y_train_bin = label_binarize(fold.y_train, classes=all_classes)
        auc_test = (
            np.nan
            if single_class_test
            else roc_auc_score(fold.y_test, fold.y_prob_test, multi_class="ovr", average="weighted")
        )
        auc_train = (
            np.nan
            if single_class_train
            else roc_auc_score(fold.y_train, fold.y_prob_train, multi_class="ovr", average="weighted")
        )
        auc_pr_test = (
            np.nan if single_class_test else average_precision_score(y_test_bin, fold.y_prob_test, average="weighted")
        )
        auc_pr_train = (
            np.nan
            if single_class_train
            else average_precision_score(y_train_bin, fold.y_prob_train, average="weighted")
        )
        logloss_test = log_loss(fold.y_test, fold.y_prob_test, labels=all_classes)
        logloss_train = log_loss(fold.y_train, fold.y_prob_train, labels=all_classes)
        f1_test = f1_score(fold.y_test, fold.y_pred_test, average="weighted", zero_division=0)
        f1_train = f1_score(fold.y_train, fold.y_pred_train, average="weighted", zero_division=0)
        # Multiclass Brier: mean over samples of sum-of-squared probability errors
        brier_test = float(np.mean(np.sum((y_test_bin - fold.y_prob_test) ** 2, axis=1)))
        brier_train = float(np.mean(np.sum((y_train_bin - fold.y_prob_train) ** 2, axis=1)))
    else:
        auc_test = np.nan if single_class_test else roc_auc_score(fold.y_test, fold.y_prob_test)
        auc_train = np.nan if single_class_train else roc_auc_score(fold.y_train, fold.y_prob_train)
        auc_pr_test = np.nan if single_class_test else average_precision_score(fold.y_test, fold.y_prob_test)
        auc_pr_train = np.nan if single_class_train else average_precision_score(fold.y_train, fold.y_prob_train)
        logloss_test = log_loss(fold.y_test, fold.y_prob_test, labels=all_classes)
        logloss_train = log_loss(fold.y_train, fold.y_prob_train, labels=all_classes)
        f1_test = f1_score(fold.y_test, fold.y_pred_test, zero_division=0, labels=all_classes)
        f1_train = f1_score(fold.y_train, fold.y_pred_train, zero_division=0, labels=all_classes)
        brier_test = float(brier_score_loss(fold.y_test, fold.y_prob_test))
        brier_train = float(brier_score_loss(fold.y_train, fold.y_prob_train))

    return {
        "AUC_Test": auc_test,
        "AUC_Train": auc_train,
        "AUC_PR_Test": auc_pr_test,
        "AUC_PR_Train": auc_pr_train,
        "LogLoss_Test": logloss_test,
        "LogLoss_Train": logloss_train,
        "F1_Test": f1_test,
        "F1_Train": f1_train,
        "Brier_Test": brier_test,
        "Brier_Train": brier_train,
        "Fit_Time": fold.fit_time,
    }


def _regression_metrics(fold: FoldResult) -> dict:
    """Compute regression metrics for a single fold."""
    mse_test = mean_squared_error(fold.y_test, fold.y_pred_test)
    mse_train = mean_squared_error(fold.y_train, fold.y_pred_train)
    return {
        "R2_Test": r2_score(fold.y_test, fold.y_pred_test),
        "R2_Train": r2_score(fold.y_train, fold.y_pred_train),
        "MAE_Test": mean_absolute_error(fold.y_test, fold.y_pred_test),
        "MAE_Train": mean_absolute_error(fold.y_train, fold.y_pred_train),
        "RMSE_Test": np.sqrt(mse_test),
        "RMSE_Train": np.sqrt(mse_train),
        "Fit_Time": fold.fit_time,
    }


# Dispatcher — routes to the right metric function by task type
_METRIC_FN = {
    TaskType.CLASSIFICATION: _classification_metrics,
    TaskType.REGRESSION: _regression_metrics,
}


# ---------------------------------------------------------------------------
# Overfitting gap
# ---------------------------------------------------------------------------

# Metric triples (test_col, train_col, minimize).
# Gap convention: positive always means the model does worse on test than train (overfitting).
#   maximize metrics (AUC, R2, F1):  gap = train - test   → positive = overfitting
#   minimize metrics (LogLoss, MAE):  gap = test  - train  → positive = overfitting
_OVERFIT_PAIRS = {
    TaskType.CLASSIFICATION: [
        ("AUC_Test", "AUC_Train", False),
        ("AUC_PR_Test", "AUC_PR_Train", False),
        ("F1_Test", "F1_Train", False),
        ("LogLoss_Test", "LogLoss_Train", True),
        ("Brier_Test", "Brier_Train", True),
    ],
    TaskType.REGRESSION: [
        ("R2_Test", "R2_Train", False),
        ("MAE_Test", "MAE_Train", True),
        ("RMSE_Test", "RMSE_Train", True),
    ],
}


def _add_overfit_gap(df: pd.DataFrame, task_type: TaskType) -> pd.DataFrame:
    """
    Adds Gap columns for key metrics. A positive gap always means the model
    performs worse on test than train, regardless of metric direction.
    """
    for test_col, train_col, minimize in _OVERFIT_PAIRS.get(task_type, []):
        if test_col in df.columns and train_col in df.columns:
            metric_name = test_col.replace("_Test", "")
            df[f"{metric_name}_Gap"] = df[test_col] - df[train_col] if minimize else df[train_col] - df[test_col]
    return df


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def compute_metrics(
    cv_result: CVResult,
    metadata: Optional[RunMetadata] = None,
) -> pd.DataFrame:
    """
    Compute per-fold metrics for a single CVResult.

    Args:
        cv_result: Output of CrossValidator.run() for one model.
        metadata:  Optional RunMetadata to attach to every row.

    Returns:
        DataFrame with one row per fold.

    Raises:
        ValueError: If cv_result.task_type is neither classification nor regression.
    """
    metric_fn = _METRIC_FN.get(cv_result.task_type)
    if metric_fn is None:
        raise ValueError(f"Unsupported task type: {cv_result.task_type}")

    rows = []
    for fold in cv_result.folds:
        row = {"Model": fold.model_name, "Fold": fold.fold_id}
        row.update(metric_fn(fold))
        if metadata:
            row.update(metadata.to_dict())
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _add_overfit_gap(df, cv_result.task_type)
    return df


def compute_metrics_all(
    cv_results: list[CVResult],
    metadata: Optional[RunMetadata] = None,
) -> pd.DataFrame:
    """
    Compute per-fold metrics for a list of CVResults (all models).

    Args:
        cv_results: Output of CrossValidator.run_all().
        metadata:   Optional RunMetadata to attach to every row.

    Returns:
        DataFrame with one row per fold per model.

    Raises:
        RuntimeError: If every model has empty folds (all CV runs failed).
    """
    frames = [compute_metrics(r, metadata) for r in cv_results if r.folds]
    if not frames:
        raise RuntimeError("All CV results are empty — every model/fold failed.")
    return pd.concat(frames, ignore_index=True)


def aggregate_metrics(fold_df: pd.DataFrame, group_by_stage: bool = True) -> pd.DataFrame:
    """
    Aggregate per-fold results into mean ± std per model.

    Args:
        fold_df: Output of compute_metrics() or compute_metrics_all().
        group_by_stage: If True (default), group by Stage alongside Model and other
                        metadata columns so each (model, stage) pair gets its own row.
                        Set False to collapse all stages per model — used in step 4
                        when the optimized run's folds should be averaged without a
                        Stage breakdown.

    Returns:
        DataFrame with one row per model, columns suffixed _Mean and _Std.
    """
    # Columns to group by — model name + any metadata fields present in the df
    group_cols = ["Model"]
    meta_fields = ["Schema", "Target", "Batch", "Y_transform", "Stage", "Notes"]

    if group_by_stage:
        group_cols += [c for c in meta_fields if c in fold_df.columns]
    else:
        # Exclude Stage — pool all folds across stages per model/transform
        group_cols += [c for c in meta_fields if c in fold_df.columns and c != "Stage"]

    numeric_cols = fold_df.select_dtypes(include=np.number).columns.tolist()
    # Exclude fold id from aggregation
    numeric_cols = [c for c in numeric_cols if c != "Fold"]

    # Single groupby pass computing both statistics, then flatten the MultiIndex
    # columns (col, "mean") → "col_Mean" / (col, "std") → "col_Std". std uses
    # pandas' default ddof=1, matching the previous .std() call.
    grouped = fold_df.groupby(group_cols)[numeric_cols].agg(["mean", "std"])
    grouped.columns = [f"{col}_{stat.capitalize()}" for col, stat in grouped.columns]
    agg_df = grouped.reset_index()

    # Interleave Mean/Std columns for readability: AUC_Test_Mean, AUC_Test_Std, ...
    metric_cols = []
    for col in numeric_cols:
        metric_cols += [f"{col}_Mean", f"{col}_Std"]

    ordered_cols = group_cols + [c for c in metric_cols if c in agg_df.columns]
    return agg_df[ordered_cols]


def select_best(
    agg_df: pd.DataFrame,
    metric: str = "AUC_Test_Mean",
    minimize: bool = False,
    n_folds: Optional[int] = None,
) -> dict:
    """
    Select the best model from an aggregated metrics DataFrame.

    When n_folds is provided, ranking uses a lower/upper confidence bound
    (LCB for maximize metrics, UCB for minimize metrics):

        maximize:  score = mean - std / sqrt(n_folds)   → idxmax
        minimize:  score = mean + std / sqrt(n_folds)   → idxmin

    This prefers models whose conservative-bound performance is highest,
    penalising fold variance in proportion to 1/sqrt(n_folds).
    When n_folds is None the raw mean is used (original behaviour).

    Args:
        agg_df:   Output of aggregate_metrics().
        metric:   Column name to rank by (must end in _Mean, e.g. AUC_Test_Mean).
        minimize: If True, select the row with the lowest value (e.g. RMSE, MAE).
        n_folds:  Number of CV folds used to produce agg_df. When provided,
                  selection uses confidence-bound scoring instead of raw mean.

    Returns:
        Dict with keys model_name, metric, value, row.

    Raises:
        ValueError: If metric is not a column of agg_df.
        RuntimeError: If all values in the metric column are NaN (every model
            failed all CV folds).
    """
    if metric not in agg_df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(agg_df.columns)}")

    std_col = metric.replace("_Mean", "_Std")
    if n_folds is not None and std_col in agg_df.columns:
        penalty = agg_df[std_col] / np.sqrt(n_folds)
        score = agg_df[metric] + penalty if minimize else agg_df[metric] - penalty
    else:
        score = agg_df[metric]

    if score.isna().all():
        raise RuntimeError(
            f"select_best: all values in column '{metric}' are NaN. "
            "Every model may have failed all CV folds. Check cv_warnings on the result."
        )
    idx = score.idxmin() if minimize else score.idxmax()
    best_row = agg_df.loc[idx]

    return {
        "model_name": best_row["Model"],
        "metric": metric,
        "value": best_row[metric],
        "row": best_row,
    }
