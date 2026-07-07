"""
h2ml/utils/eda.py

Exploratory data analysis helpers for inspecting a dataset before it enters the
pipeline. These functions print human-readable summaries to stdout for
interactive use (notebooks / REPL), mirroring the idiom of pandas' own
DataFrame.info().
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def profile_dataset(df: pd.DataFrame) -> None:
    """
    Print a one-screen profile of a DataFrame.

    Reports overall shape and memory footprint, then a per-column table of dtype,
    null count, null percentage, distinct-value count, and the first row's value as
    a sample.

    Args:
        df: DataFrame to profile.
    """
    print("=== DATASET PROFILE ===\n")
    print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    summary = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "nulls": df.isnull().sum(),
            "null_%": (df.isnull().mean() * 100).round(2),
            "unique": df.nunique(),
            "sample": df.iloc[0] if len(df) else pd.NA,
        }
    )

    print(summary.to_string())


def flag_outliers(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Summarise IQR-based outliers for each numeric column (Tukey's fences).

    A value is an outlier when it falls below Q1 - multiplier*IQR or above
    Q3 + multiplier*IQR. multiplier=1.5 is the conventional "mild outlier"
    threshold; 3.0 flags only extreme outliers. NaNs are ignored — quantiles skip
    them and NaN never satisfies the fence comparisons.

    Args:
        df:         DataFrame to scan. Only numeric columns are considered.
        multiplier: IQR multiplier setting the fence width. Default 1.5.

    Returns:
        One row per numeric column with lower/upper bounds, outlier count, and
        outlier percentage. Empty (with the expected columns) when df has no rows
        or no numeric columns.
    """
    columns = ["column", "lower_bound", "upper_bound", "outlier_count", "outlier_%"]
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=columns)

    results = []
    for col in df.select_dtypes(include=np.number).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        results.append(
            {
                "column": col,
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "outlier_count": outlier_count,
                "outlier_%": round(outlier_count / n * 100, 2),
            }
        )

    return pd.DataFrame(results, columns=columns)


def target_correlations(
    df: pd.DataFrame,
    target: str,
    n_features: int | None = None,
    sort_by: str = "pearson",
) -> pd.DataFrame:
    """
    Rank numeric features by their correlation with the target.

    Computes Pearson (linear), Spearman, and Kendall (both monotonic) coefficients
    between each numeric feature and the target, giving a quick read on the
    relationships to expect during modelling. Correlations use pairwise-complete
    observations, so NaNs are dropped per feature rather than poisoning the result.

    Args:
        df:         DataFrame containing the target and feature columns.
        target:     Name of the (numeric) target column.
        n_features: Number of top features to return. None returns all.
        sort_by:    Coefficient to rank by — "pearson", "spearman", or "kendall".
                    Rows are ordered by descending absolute value, so the strongest
                    relationships (positive or negative) come first; NaNs sink last.

    Returns:
        DataFrame with columns [feature, pearson, spearman, kendall], one row per
        numeric feature, sorted strongest-first. Empty (with those columns) when no
        numeric features are present.

    Raises:
        ValueError: If target is missing, non-numeric, or sort_by is invalid.
    """
    methods: list[Literal["pearson", "spearman", "kendall"]] = ["pearson", "spearman", "kendall"]
    if sort_by not in methods:
        raise ValueError(f"sort_by must be one of {methods}, got {sort_by!r}")
    if target not in df.columns:
        raise ValueError(f"target {target!r} not found in df columns.")

    numeric = df.select_dtypes(include=np.number)
    if target not in numeric.columns:
        raise ValueError(f"target {target!r} must be numeric to correlate against.")

    features = [c for c in numeric.columns if c != target]
    columns = ["feature", *methods]
    if not features:
        return pd.DataFrame(columns=columns)

    y = numeric[target]
    corr = pd.DataFrame([{"feature": col, **{m: numeric[col].corr(y, method=m) for m in methods}} for col in features])
    # Rank by |coefficient| so the strongest relationships (either sign) come first;
    # NaNs (e.g. a constant feature) sort last by default.
    corr = corr.sort_values(by=sort_by, key=lambda s: s.abs(), ascending=False, ignore_index=True).round(3)

    if n_features is not None:
        corr = corr.head(n_features)
    return corr


def check_pipeline_readiness(
    df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    n_splits: int = 5,
    max_classes: int = 2,
) -> pd.DataFrame:
    """
    Report issues that would stop (or degrade) an H2MLPipeline run, before you run it.

    Mirrors the checks in H2MLPipeline._validate_store, but reports them all at once
    instead of raising on the first, so you can fix everything in one pass. Numeric
    columns are what become the model matrix X, so the finite/variance checks target
    those.

    This is deliberately a blockers-only verdict — for descriptive per-column stats
    use profile_dataset, and for outliers use flag_outliers.

    Args:
        df:          DataFrame of features (and optionally the target).
        target:      Target column name. When given, target-specific checks run.
        task:        "classification" enables class-count / cardinality / imbalance
                     checks. None or "regression" skips them.
        n_splits:    Planned CV fold count, checked against the row count.
        max_classes: For classification, a numeric target with more than this many
                     distinct values is flagged as likely count/continuous data
                     mistakenly typed as classification (the pipeline would treat each
                     value as its own class). Default 2 — only a binary 0/1 target
                     passes; raise it if you genuinely have a numeric multiclass label.

    Returns:
        DataFrame [check, severity, columns, detail], one row per detected issue
        (severity "error" = will break the run, or "warning"). Empty when ready.
    """
    issues: list[tuple[str, str, str, str]] = []

    n = len(df)
    if n < n_splits:
        issues.append(("insufficient_rows", "error", "", f"{n} rows < n_splits={n_splits}: add data or lower n_splits"))

    all_cols = list(df.columns)
    dups = sorted({str(c) for c in all_cols if all_cols.count(c) > 1})
    if dups:
        issues.append(("duplicate_columns", "error", ", ".join(dups), "each column must have a unique name"))

    # Drop duplicate-named columns (already reported above) so per-column access
    # below returns a Series, not a DataFrame.
    numeric = df.select_dtypes(include=np.number).loc[:, lambda d: ~d.columns.duplicated()]
    feature_cols = [c for c in numeric.columns if c != target]

    nonfinite = {c: int((~np.isfinite(numeric[c])).sum()) for c in feature_cols}
    bad = {c: k for c, k in nonfinite.items() if k}
    if bad:
        detail = ", ".join(f"{c} ({k})" for c, k in list(bad.items())[:10])
        if len(bad) > 10:
            detail += f", … +{len(bad) - 10} more"
        issues.append(("nonfinite_features", "error", str(len(bad)), f"NaN/inf in numeric features: {detail}"))

    constant = [str(c) for c in feature_cols if numeric[c].nunique(dropna=True) <= 1]
    if constant:
        issues.append(
            ("constant_features", "error", ", ".join(constant), "zero variance breaks SHAP in step 2; drop them")
        )

    if target is not None:
        if target not in df.columns:
            issues.append(("missing_target", "error", str(target), "target column not found in df"))
        else:
            y = df[target]
            if pd.api.types.is_numeric_dtype(y):
                n_bad = int((~np.isfinite(y.to_numpy(dtype="float64"))).sum())
                if n_bad:
                    issues.append(("nonfinite_target", "error", str(target), f"{n_bad} NaN/inf target value(s)"))
            elif y.isnull().any():
                issues.append(("null_target", "error", str(target), f"{int(y.isnull().sum())} missing target value(s)"))
            if task == "classification":
                n_classes = int(y.nunique(dropna=True))
                if n_classes < 2:
                    issues.append(("single_class_target", "error", str(target), f"only {n_classes} class; need >= 2"))
                elif pd.api.types.is_numeric_dtype(y) and n_classes > max_classes:
                    # Likely counts/continuous mis-typed as classification. Flag this
                    # instead of class_imbalance, whose "handle_imbalance=True" advice
                    # would be misleading here (the real fix is regression/binarize).
                    issues.append(
                        (
                            "high_cardinality_target",
                            "warning",
                            str(target),
                            f"{n_classes} distinct numeric values (> max_classes={max_classes}) — looks "
                            "like counts/continuous, not a categorical label. Classification treats each "
                            "value as its own class; use task='regression', binarize (e.g. y > 0), or the "
                            "delta model.",
                        )
                    )
                else:
                    counts = y.value_counts()
                    ratio = float(counts.min() / counts.sum())
                    if ratio < 0.1:
                        issues.append(
                            (
                                "class_imbalance",
                                "warning",
                                str(target),
                                f"minority ratio={ratio:.3f}; try handle_imbalance=True",
                            )
                        )

    return pd.DataFrame(issues, columns=["check", "severity", "columns", "detail"])


def correlated_feature_pairs(
    df: pd.DataFrame,
    threshold: float = 0.7,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    target: str | None = None,
) -> pd.DataFrame:
    """
    List pairs of numeric features whose absolute correlation meets a threshold.

    Surfaces multicollinearity / redundancy *between features* — a preview of what
    the pipeline's step-2 correlation filter (corr_threshold, default 0.7) will prune.
    The target is excluded so this stays feature↔feature; for feature↔target
    relationships use target_correlations.

    Note: step 2 drops a feature when it exceeds corr_threshold in *any* of Pearson,
    Spearman, or Kendall; this helper inspects one method at a time for clarity.

    Args:
        df:        DataFrame of features (and optionally the target).
        threshold: Minimum |correlation| for a pair to be reported. Range [0, 1].
        method:    "pearson", "spearman", or "kendall".
        target:    Target column to exclude from the feature set (optional).

    Returns:
        DataFrame [feature_1, feature_2, corr] for pairs with |corr| >= threshold,
        strongest first. Empty (with those columns) when fewer than two numeric
        features are present or none clear the threshold.

    Raises:
        ValueError: If method is invalid or threshold is outside [0, 1].
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"method must be 'pearson', 'spearman', or 'kendall', got {method!r}")
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    numeric = df.select_dtypes(include=np.number)
    if target is not None and target in numeric.columns:
        numeric = numeric.drop(columns=[target])

    cols = ["feature_1", "feature_2", "corr"]
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=cols)

    corr = numeric.corr(method=method)
    # Keep the strict upper triangle so each pair appears once and the self-correlation
    # diagonal is dropped; .stack() then discards the NaN-masked lower triangle.
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().reset_index()
    pairs.columns = cols
    pairs = pairs[pairs["corr"].abs() >= threshold]
    pairs = pairs.sort_values("corr", key=lambda s: s.abs(), ascending=False, ignore_index=True)
    pairs["corr"] = pairs["corr"].round(3)
    return pairs
