# Exploratory Data Analysis

`h2ml.utils` ships a small set of EDA helpers for inspecting a dataset **before** it enters the pipeline. Each answers one question, and they don't overlap — run them in sequence to understand the data and catch problems early.

```python
from h2ml.utils import (
    profile_dataset,
    flag_outliers,
    target_correlations,
    check_pipeline_readiness,
    correlated_feature_pairs,
)
```

| Function | Question it answers |
|----------|---------------------|
| `profile_dataset` | What does this data look like? |
| `flag_outliers` | Which numeric columns have outliers? |
| `target_correlations` | Which features relate to the target? |
| `correlated_feature_pairs` | Which features are redundant with each other? |
| `check_pipeline_readiness` | Will `pipeline.run()` break, and why? |

All operate on a `pandas.DataFrame`. The four that return a DataFrame are sorted/structured for direct inspection; `profile_dataset` prints to stdout.

## Profiling

`profile_dataset(df)` prints shape, memory footprint, and a per-column table of dtype, null count/percentage, distinct-value count, and a sample value:

```python
profile_dataset(df)
```

## Outliers

`flag_outliers(df, multiplier=1.5)` returns IQR (Tukey-fence) outlier counts per numeric column. `multiplier=1.5` is the conventional mild-outlier threshold; raise it to `3.0` to flag only extreme outliers:

```python
flag_outliers(df, multiplier=1.5)
```

## Feature–target relationships

`target_correlations(df, target, n_features=None, sort_by="pearson")` ranks numeric features by their Pearson (linear), Spearman, and Kendall (monotonic) correlation with the target — strongest first, by absolute value. This previews the relationships the model will exploit:

```python
target_correlations(df, "abundance", n_features=10)          # top 10 by |Pearson|
target_correlations(df, "abundance", sort_by="spearman")      # rank by monotonic strength
```

## Feature–feature redundancy

`correlated_feature_pairs(df, threshold=0.7, method="pearson", target=...)` lists feature pairs whose absolute correlation meets the threshold — a preview of what the pipeline's [step-2 correlation filter](../pipeline.md) (`corr_threshold`, default `0.7`) will prune. The target is excluded so this stays feature↔feature:

```python
correlated_feature_pairs(df, threshold=0.8, target="abundance")
```

!!! note
    Step 2 drops a feature when it exceeds `corr_threshold` in **any** of Pearson, Spearman, or Kendall; this helper inspects one method at a time for clarity.

## Pipeline readiness

`check_pipeline_readiness(df, target=None, task=None, n_splits=5, max_classes=2)` mirrors the validation inside `H2MLPipeline._validate_store`, but reports **all** blocking issues at once instead of raising on the first. An empty result means the data is ready to run.

```python
check_pipeline_readiness(df, target="present", task="classification")
```

Each row is one issue with a `severity` of `error` (will break the run) or `warning`. Checks include:

| `check` | Severity | Meaning |
|---------|----------|---------|
| `insufficient_rows` | error | fewer rows than `n_splits` |
| `duplicate_columns` | error | non-unique column names |
| `nonfinite_features` | error | NaN/inf in numeric features |
| `constant_features` | error | zero-variance columns (break SHAP in step 2) |
| `missing_target` / `nonfinite_target` / `null_target` | error | target absent or non-finite |
| `single_class_target` | error | classification target with < 2 classes |
| `high_cardinality_target` | warning | numeric classification target with too many distinct values (see below) |
| `class_imbalance` | warning | minority class ratio < 10% |

### Counts mistyped as classification

A common mistake is passing a **count** target (`0, 1, 2, 3, …`) with `task="classification"`. The pipeline does **not** binarize it — it treats each distinct count as its own class, producing a high-cardinality, badly-imbalanced multiclass problem.

`check_pipeline_readiness` catches this: a numeric classification target with more than `max_classes` distinct values (default **2** — binary only) is flagged as `high_cardinality_target` *instead of* the misleading `class_imbalance` (whose `handle_imbalance=True` advice would not help):

```python
check_pipeline_readiness(df, target="counts", task="classification")
# high_cardinality_target  warning  ... use task='regression', binarize (y > 0), or the delta model.
```

Fixes, depending on intent:

- **Presence/absence** → binarize the target: `PipelineData.from_frame(X, y=(counts > 0).astype(int))`
- **Abundance** → use `task="regression"` (optionally with a [y-transform](transforms.md))
- **Both** → the delta model (`build_delta_final_model`): presence classifier × abundance regressor

Raise `max_classes` if you genuinely have a numeric multiclass label.
