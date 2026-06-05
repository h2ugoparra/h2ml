# Visualization

All plot functions live in `h2ml.plots.plots`. Each accepts an optional `save_path`; omit it to call `plt.show()`.

## Model scores

```python
from h2ml.plots.plots import model_scores, pipeline_scores

# Step 1 scores only
model_scores(result, save_path="plots/step1_scores.png")

# All stages (default, reduced, optimized) overlaid
pipeline_scores(result, save_path="plots/all_stages.png")
```

## CV diagnostics

Classification panel (ROC, PR curve, calibration, confusion matrix) or regression panel (residuals, actual vs predicted, error distribution):

```python
from h2ml.plots.plots import cv_diagnostics

cv_diagnostics(result, save_path="plots/diagnostics.png")
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

## Spatial fold assignment

Visualise how samples are distributed across spatial CV folds:

```python
from h2ml.plots.plots import spatial_folds

spatial_folds(result, store, save_path="plots/folds.png")
```

Only available when `result.cv_type == "spatial"`.

## Spatial blocks

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
coords are longitude and latitude. `result.splitter` is `None` after
`PipelineResult.load()` — rebuild via `build_final_model()` first.
