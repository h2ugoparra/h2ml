# Visualization

All plot functions live in `h2ml.plots.plots`. Each accepts an optional `save_path`; omit it to call `plt.show()`.

## Model scores

```python
from h2ml.plots.plots import model_scores, pipeline_scores

# Per-fold scores of every step-1 candidate (takes a fold DataFrame and a column)
model_scores(result.step1_fold_df, "R2_Test", save_path="plots/step1_scores.png")

# All stages (default, reduced, optimized) overlaid, on the run's selection metric
pipeline_scores(result, save_path="plots/all_stages.png")
```

These plot per-fold scores. For regression, selection ranks on the pooled score
(`<metric>_Test_Pooled` in the agg frames), so the plotted fold spread can favour a
different candidate than the one that won.

## CV diagnostics

Takes a `CVResult` (or a list of them, pooled into one panel). Classification panel (predicted probability and log-loss by class, residuals, ROC, calibration) or regression panel (observed vs predicted, residuals vs fitted, residual histogram, QQ plot):

```python
from h2ml.plots.plots import cv_diagnostics

cv_diagnostics(result.best_cv_result, save_path="plots/diagnostics.png")
```

## SHAP importance

```python
from h2ml.plots.plots import shap_importance, shap_summary_plot, shap_dependence

# Horizontal bar chart — mean absolute SHAP per feature
shap_importance(result.selector, save_path="plots/shap_bar.png")

# Beeswarm — direction and magnitude per sample for the final best model
shap_summary_plot(result, save_path="plots/shap_beeswarm.png")

# Scatter + LOWESS with bootstrap CI band for the top-N most important features
shap_dependence(result, n_features=6, save_path="plots/shap_dependence.png")
```

## Spatial blocks and folds

A two-panel scatter showing each sample's block ID (left) and the fold it is held
out in (right). Call `.plot()` directly on a fitted splitter, or use the exported
`plot_spatial_blocks` function:

```python
from h2ml.plots import plot_spatial_blocks

# Shortcut method on the splitter (result.splitter is the fitted instance)
result.splitter.plot(save_path="plots/blocks.png")

# Equivalent explicit call
plot_spatial_blocks(result.splitter, save_path="plots/blocks.png")
```

`lon_col` / `lat_col` (default `1` / `0`) select which columns of the splitter's
coords are longitude and latitude. The defaults assume `[lat, lon]` coords; if you
built them as `np.column_stack([lon, lat])`, pass `lon_col=0, lat_col=1`.

Only available when `result.cv_type == "spatial"`.

`result.splitter` is not persisted, so it is `None` after `PipelineResult.load()`
(`build_final_model()` does not restore it). Rebuild it from the rows the CV ran on
(`result.features`, which excludes the holdout) and the same config — the folds are
deterministic, so they match the ones the pipeline used:

```python
from h2ml.core.base import TaskType
from h2ml.features.spatial_cv import build_splitter

dev = result.features
splitter = build_splitter(
    TaskType(config.task_type),
    n_splits=config.n_splits,
    random_state=config.random_state,
    shuffle=True,
    coords=dev.coords,
    X=dev.X,
    y=dev.y,
    spatial=config.spatial_cv,
)
splitter.plot()
```

With `spatial_cv_method="spcv"` and its target view on (`spcv_use_target=True`), the
pipeline built the splitter from the first store of the y-transform sweep, so pass that
store's `y` (e.g. `np.log1p(dev.y)` when `"log"` was listed first) to get the same folds.
