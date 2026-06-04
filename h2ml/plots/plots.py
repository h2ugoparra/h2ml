"""
h2ml/plots/plots.py

Visualization functions for PipelineResult, CVResult, and FeatureSelector.

Functions
---------
model_scores        — horizontal pointplot of per-fold metric scores
pipeline_scores     — model_scores across all three pipeline stages from a PipelineResult
cv_diagnostics      — classification or regression diagnostic panel
shap_importance     — horizontal bar chart of SHAP feature importance
shap_summary_plot   — SHAP beeswarm for the final best model (recomputes SHAP)
shap_dependence     — scatter + lowess for top-N features
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib  # noqa: F401
# matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import roc_curve, auc as sklearn_auc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_or_show(save_path: Optional[Path]) -> None:
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _auto_hue(df: pd.DataFrame, prefer: list[str] | None = None) -> Optional[str]:
    """Return the first candidate column present in df, or None."""
    candidates = prefer or ["Y_transform", "Stage"]
    for col in candidates:
        if col in df.columns and df[col].nunique() > 1:
            return col
    return None


# ---------------------------------------------------------------------------
# Model scores
# ---------------------------------------------------------------------------


def model_scores(
    fold_df: pd.DataFrame,
    metric_col: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Horizontal pointplot of per-fold metric scores across models.

    Useful for comparing step1 or step3 results. Hue defaults to
    'Y_transform' (if present and varied) or 'Stage'.

    Args:
        fold_df:    Per-fold metrics DataFrame (output of compute_metrics_all()).
        metric_col: Column to plot, e.g. 'AUC_Test' or 'R2_Test'.
        hue:        Column to use as colour grouping. Auto-detected if None.
        title:      Plot title. Defaults to metric_col.
        save_path:  Path to save figure. If None, shows the plot.
    """
    df = fold_df.copy()
    hue_col = hue or _auto_hue(df)

    n_models = df["Model"].nunique()
    fig_h = max(4, n_models * 0.7 + 2)
    plt.figure(figsize=(10, fig_h))

    sns.pointplot(
        data=df,
        y="Model",
        x=metric_col,
        hue=hue_col,
        dodge=0.2,  # type: ignore
        linestyles="none",
        capsize=0.15,
        errorbar="sd",
        err_kws={"linewidth": 1},
        native_scale=False,
    )

    plt.title(title or metric_col)
    plt.xlabel("")
    plt.ylabel("")
    if hue_col:
        plt.legend(
            title="",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=3,
            frameon=False,
        )
    _save_or_show(save_path)


def pipeline_scores(
    result,
    metric_col: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Compare model scores across all completed pipeline stages in one plot.

    Concats step1 (default), step3 (reduced), and step4 (optimized) fold
    DataFrames and calls model_scores with hue='Stage'.

    Args:
        result:     PipelineResult from H2MLPipeline.run().
        metric_col: Metric column to plot. Auto-detected from task type if None
                    ('AUC_Test' for classification, 'R2_Test' for regression).
        title:      Plot title.
        save_path:  Path to save figure. If None, shows the plot.

    Raises:
        ValueError: If the result has no fold DataFrames to plot.
    """
    frames = [df for df in [result.step1_fold_df, result.step3_fold_df, result.step4_fold_df] if df is not None]
    if not frames:
        raise ValueError("PipelineResult has no fold DataFrames to plot.")

    combined = pd.concat(frames, ignore_index=True)

    if metric_col is None:
        metric_col = "AUC_Test" if "AUC_Test" in combined.columns else "R2_Test"

    model_scores(
        combined,
        metric_col=metric_col,
        hue="Stage",
        title=title or metric_col,
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# CV diagnostics
# ---------------------------------------------------------------------------


def cv_diagnostics(
    cv_result,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Diagnostic panel pooling all folds from one or more CVResults.

    Accepts a single CVResult or a list of CVResults. When a list is passed
    (e.g. result.step1_cv_result), all folds from all models are pooled
    together into one diagnostic panel.

    Classification (3×2): predicted probability by class, log-loss by class,
    raw residuals vs predicted, residuals histogram, ROC curve, calibration.

    Regression (2×2): observed vs predicted, residuals vs fitted,
    residuals histogram, QQ plot.

    Args:
        cv_result: CVResult or list[CVResult].
        title:     Panel title. Defaults to model name (single) or 'All Models' (list).
        save_path: Path to save figure. If None, shows the plot.

    Raises:
        ValueError: If a list is passed and none of the CVResults have folds.
    """
    from h2ml.core.base import TaskType

    if isinstance(cv_result, list):
        results = [r for r in cv_result if r.folds]
        if not results:
            raise ValueError("No CVResults with folds to plot.")
        task_type = results[0].task_type
        all_folds = [f for r in results for f in r.folds]
        main_title = title or "All Models"
    else:
        task_type = cv_result.task_type
        all_folds = cv_result.folds
        main_title = title or cv_result.model_name

    y_true = np.concatenate([f.y_test for f in all_folds])
    y_pred = np.concatenate([f.y_pred_test for f in all_folds])

    if task_type == TaskType.CLASSIFICATION:
        y_prob = np.concatenate([f.y_prob_test for f in all_folds])
        _classification_diagnostics(y_true, y_prob, main_title, save_path)
    else:
        _regression_diagnostics(y_true, y_pred, main_title, save_path)


def _classification_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    save_path: Optional[Path],
) -> None:
    """3×2 classification diagnostic panel."""
    ll = -(
        y_true * np.log(np.clip(y_prob, 1e-15, 1 - 1e-15))
        + (1 - y_true) * np.log(np.clip(1 - y_prob, 1e-15, 1 - 1e-15))
    )
    residuals = y_true - y_prob
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = sklearn_auc(fpr, tpr)

    df_pred = pd.DataFrame({"y": y_true, "y_prob": y_prob})
    df_pred["bin"] = pd.qcut(df_pred["y_prob"], q=10, duplicates="drop")
    calib = df_pred.groupby("bin", observed=True).agg(y_mean=("y", "mean"), y_prob_mean=("y_prob", "mean"))

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title)

    # [0,0] Predicted probability by class
    sns.histplot(
        y_prob[y_true == 1],
        ax=axes[0, 0],
        label="Positive",
        color="steelblue",
        stat="density",
        kde=True,
        bins=50,
    )
    sns.histplot(
        y_prob[y_true == 0],
        ax=axes[0, 0],
        label="Negative",
        color="tomato",
        stat="density",
        kde=True,
        bins=50,
    )
    axes[0, 0].set_title("Predicted Probability by Class")
    axes[0, 0].set_xlabel("Predicted Probability")
    axes[0, 0].legend()

    # [0,1] Log-loss by class
    sns.histplot(
        ll[y_true == 1],
        ax=axes[0, 1],
        color="steelblue",
        label="Positive",
        kde=True,
        bins=50,
        alpha=0.5,
    )
    sns.histplot(
        ll[y_true == 0],
        ax=axes[0, 1],
        color="tomato",
        label="Negative",
        kde=True,
        bins=50,
        alpha=0.5,
    )
    axes[0, 1].set_title("Log Loss by Class")
    axes[0, 1].set_xlabel("Log Loss")
    axes[0, 1].legend()

    # [1,0] Residuals vs predicted probability
    axes[1, 0].scatter(y_prob, residuals, alpha=0.3, s=10)
    axes[1, 0].axhline(0, color="red", linestyle="--")
    axes[1, 0].set_title("Residuals vs Predicted Probability")
    axes[1, 0].set_xlabel("Predicted Probability")
    axes[1, 0].set_ylabel("Residuals")

    # [1,1] Residuals histogram
    sns.histplot(residuals, ax=axes[1, 1], kde=True, bins=50, color="steelblue", alpha=0.4)
    axes[1, 1].axvline(0, color="red", linestyle="--")
    axes[1, 1].set_title("Residuals Histogram")
    axes[1, 1].set_xlabel("Residuals")

    # [2,0] ROC curve
    axes[2, 0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[2, 0].plot([0, 1], [0, 1], "r--")
    axes[2, 0].set_xlabel("False Positive Rate")
    axes[2, 0].set_ylabel("True Positive Rate")
    axes[2, 0].set_title("ROC Curve")
    axes[2, 0].legend()

    # [2,1] Calibration plot
    axes[2, 1].plot([0, 1], [0, 1], "r--")
    sns.scatterplot(data=calib, ax=axes[2, 1], x="y_prob_mean", y="y_mean", s=80)
    axes[2, 1].set_title("Calibration Plot")
    axes[2, 1].set_xlabel("Mean Predicted Probability")
    axes[2, 1].set_ylabel("Observed Frequency")

    _save_or_show(save_path)


def _regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[Path],
) -> None:
    """2×2 regression diagnostic panel."""
    residuals = y_true - y_pred

    if np.allclose(residuals, 0):
        print(f"All residuals are ~0 for '{title}' — skipping diagnostics.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    # [0,0] Observed vs Predicted
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0, 0].scatter(y_true, y_pred, alpha=0.4, s=15)
    axes[0, 0].plot(lims, lims, "r--", lw=1.5)
    axes[0, 0].set_xlabel("Observed")
    axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].set_title("Observed vs Predicted")

    # [0,1] Residuals vs Fitted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.4, s=15)
    axes[0, 1].axhline(0, color="red", linestyle="--")
    axes[0, 1].set_xlabel("Fitted Values")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residuals vs Fitted")

    # [1,0] Residuals histogram
    sns.histplot(residuals, ax=axes[1, 0], kde=True, bins=30, color="steelblue")
    axes[1, 0].axvline(0, color="red", linestyle="--")
    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_title("Residuals Distribution")

    # [1,1] QQ plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ Plot of Residuals")

    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# SHAP plots
# ---------------------------------------------------------------------------


def shap_importance(
    selector,
    n_features: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Horizontal bar chart of mean absolute SHAP feature importance.

    Selected features are shown in blue; removed (correlated) features in grey.

    Args:
        selector:   Fitted FeatureSelector.
        n_features: Number of top features to display. Defaults to all.
        title:      Plot title.
        save_path:  Path to save figure. If None, shows the plot.
    """
    summary = selector.importance_summary()
    summary = summary.sort_values("importance", ascending=False)
    if n_features:
        summary = summary.head(n_features)

    summary = summary.sort_values("importance")  # ascending for horizontal bar
    colors = ["steelblue" if sel else "lightgrey" for sel in summary["selected"]]

    fig_h = max(4, len(summary) * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(summary["feature"], summary["importance"], color=colors)
    ax.margins(y=0.01)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title or "SHAP Feature Importance")

    # Legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(color="steelblue", label="Selected"),
        Patch(color="lightgrey", label="Removed (correlated)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)

    _save_or_show(save_path)


def shap_dependence(
    result,
    n_features: int = 6,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Scatter + lowess plots for the top-N most important features.

    Refits the overall best model and recomputes SHAP values.

    Args:
        result:     PipelineResult from H2MLPipeline.run().
        n_features: Number of top features to plot (default 6, arranged in 2×3 grid).
        title:      Figure title.
        save_path:  Path to save figure. If None, shows the plot.
    """
    shap_values, importance, store = _compute_final_shap(result)
    feat_idx = {name: i for i, name in enumerate(store.feature_names)}
    top_features = importance.index[:n_features].tolist()

    ncols = min(3, n_features)
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = list(np.array(axes).flat) if n_features > 1 else [axes]

    fig.suptitle(title or "SHAP Dependence Plots", fontsize=12, fontweight="bold")

    for ax, feat in zip(axes_flat, top_features):
        idx = feat_idx.get(feat)
        if idx is None:
            ax.set_visible(False)
            continue

        x_vals = store.X[:, idx]
        shap_vals = shap_values[:, idx]

        ax.scatter(x_vals, shap_vals, alpha=0.3, c="#aed6dc", s=20)
        try:
            sns.regplot(
                x=x_vals,
                y=shap_vals,
                lowess=True,
                scatter=False,
                color="#f47a60",
                line_kws={"linewidth": 2},
                ax=ax,
            )
        except RuntimeError:
            # statsmodels not installed — fall back to a degree-2 polynomial smooth
            sns.regplot(
                x=x_vals,
                y=shap_vals,
                order=2,
                scatter=False,
                color="#f47a60",
                line_kws={"linewidth": 2},
                ax=ax,
            )
        ax.set_xlabel(feat)
        ax.set_ylabel(f"SHAP ({feat})")

    # Hide unused axes
    for ax in axes_flat[len(top_features) :]:
        ax.set_visible(False)

    _save_or_show(save_path)


def _compute_final_shap(result) -> tuple:
    """
    Build the FinalModel and compute SHAP values from it.

    Passes scaled X to the SHAP explainer when the model requires scaling,
    but returns the original store so display values remain on the original scale.

    Result is cached on result._final_shap_cache so subsequent calls within the
    same session (e.g. shap_summary_plot followed by shap_dependence) skip the
    expensive refit + SHAP recomputation.

    Returns (shap_values, feature_importance, store).
    """
    if getattr(result, "_final_shap_cache", None) is not None:
        return result._final_shap_cache

    from h2ml.features.shap_importance import get_shap_values
    from h2ml.core.feature_store import PipelineData
    from h2ml.pipeline.final_model import build_final_model

    final = build_final_model(result)

    feature_stage = result.best_feature_stage or result.best_stage
    store = result.features_reduced if feature_stage == "reduced" else result.features

    # If the model was trained on scaled X, SHAP must also receive scaled X
    if final.requires_scaling and final.scaler is not None:
        shap_store = PipelineData(
            X=final.scaler.transform(store.X),
            feature_names=store.feature_names,
            y=store.y,
        )
    else:
        shap_store = store

    shap_values, feature_importance = get_shap_values(final.estimator, shap_store, final.task_type)

    result._final_shap_cache = (shap_values, feature_importance, store)
    return result._final_shap_cache


def shap_summary_plot(
    result,
    save_path: Optional[Path] = None,
) -> None:
    """
    SHAP beeswarm summary plot for the overall best model.

    Refits the best model (with optimized params when available) on the
    correct feature store and recomputes SHAP values.

    Args:
        result:    PipelineResult from H2MLPipeline.run().
        save_path: Path to save figure. If None, shows the plot.
    """
    import shap as _shap

    shap_values, _, store = _compute_final_shap(result)

    X_df = pd.DataFrame(store.X, columns=store.feature_names)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*NumPy global RNG.*")
        _shap.summary_plot(shap_values, X_df, show=False)
    plt.gcf().set_size_inches(10, 6)

    _save_or_show(save_path)


def plot_spatial_blocks(
    splitter,
    lon_col: int = 1,
    lat_col: int = 0,
    save_path: Optional[Path] = None,
) -> None:
    """
    Two-panel scatter plot of spatial CV block and fold assignments.

    Left panel  — each point coloured by its compact block ID.
                  Reveals the grid resolution used internally by the splitter.
    Right panel — each point coloured by the fold it belongs to (0 … n_splits-1).
                  Shows exactly which geographic region is held out per fold.

    Args:
        splitter:  A fitted SpatialBlockSplitter instance.
        lon_col:   Column index in splitter.coords for longitude (default 1).
        lat_col:   Column index in splitter.coords for latitude  (default 0).
        save_path: Path to save figure. If None, shows the plot interactively.

    Example:
        >>> coords = np.column_stack([lats, lons])
        >>> splitter = SpatialBlockSplitter(coords, n_splits=5, n_blocks_per_fold=5)
        >>> plot_spatial_blocks(splitter)
    """
    coords = splitter.coords
    block_id = splitter.block_id_
    fold_id = splitter._fold_of_sample

    lon = coords[:, lon_col]
    lat = coords[:, lat_col]

    n_blocks = int(block_id.max()) + 1
    n_folds = splitter.n_splits

    fig, (ax_block, ax_fold) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: blocks ---
    block_cmap = plt.get_cmap("tab20" if n_blocks <= 20 else "turbo", n_blocks)
    sc_block = ax_block.scatter(
        lon,
        lat,
        c=block_id,
        cmap=block_cmap,
        vmin=-0.5,
        vmax=n_blocks - 0.5,
        s=18,
        linewidths=0,
    )
    cb_block = fig.colorbar(sc_block, ax=ax_block, pad=0.02)
    cb_block.set_label("Block ID")
    cb_block.set_ticks(range(n_blocks))
    ax_block.set_title(f"Spatial blocks  (n={n_blocks})")
    ax_block.set_xlabel("Longitude" if lon_col == 1 else f"coords[:, {lon_col}]")
    ax_block.set_ylabel("Latitude" if lat_col == 0 else f"coords[:, {lat_col}]")
    ax_block.set_aspect("equal", adjustable="datalim")

    # --- Right: folds ---
    fold_cmap = plt.get_cmap("tab10", n_folds)
    sc_fold = ax_fold.scatter(
        lon,
        lat,
        c=fold_id,
        cmap=fold_cmap,
        vmin=-0.5,
        vmax=n_folds - 0.5,
        s=18,
        linewidths=0,
    )
    cb_fold = fig.colorbar(sc_fold, ax=ax_fold, pad=0.02)
    cb_fold.set_label("Fold (test set)")
    cb_fold.set_ticks(range(n_folds))
    n_blocks = int(block_id.max()) + 1
    if hasattr(splitter, "n_blocks_per_fold"):
        block_info = f"n_blocks_per_fold={splitter.n_blocks_per_fold}"
    else:
        block_info = f"n_blocks={n_blocks}"
    ax_fold.set_title(f"Fold assignments  (n_splits={n_folds}, {block_info})")
    ax_fold.set_xlabel("Longitude" if lon_col == 1 else f"coords[:, {lon_col}]")
    ax_fold.set_ylabel("Latitude" if lat_col == 0 else f"coords[:, {lat_col}]")
    ax_fold.set_aspect("equal", adjustable="datalim")

    _save_or_show(save_path)
