# Y-Transforms

Y-transforms allow the pipeline to evaluate whether predicting in a transformed target space improves generalisation. Steps 1 and 3 sweep all (model × transform) combinations; the winning pair is selected once and used throughout.

## Usage

```python
result = pipeline.run(store, transforms=["log", "sqrt", "count", "wincount"])
print(result.y_transform)   # e.g. "log"
```

Only supported for regression (`task_type="regression"`).

## Available transforms

| Name | Transform | Inverse | Notes |
|------|-----------|---------|-------|
| `"count"` | identity | identity | Baseline — no transformation |
| `"log"` | `log1p(y)` | `expm1(y)` | Requires non-negative y |
| `"sqrt"` | `sqrt(y)` | `y²` | Requires non-negative y |
| `"wincount"` | winsorize(y) | identity | Caps y at the 99th percentile of its non-zero values |
| `"winlog"` | winsorize(log1p(y)) | `expm1(y)` | Log, then cap. Usually close to `"log"`, because the log already compresses the top |
| `"winsqrt"` | winsorize(sqrt(y)) | `y²` | Square root, then cap. Usually close to `"sqrt"` |

Winsorize-based transforms return `None` when nothing lies above the cap and are **silently skipped** by `build_transform_stores`. If all winsorized variants are skipped, only non-winsorized transforms are evaluated.

The cap ignores zeros, so on a zero-heavy target it stays in the tail instead of cutting into ordinary catches. A cap removes real signal when large values are genuine: it lowers predicted totals, so compare totals as well as RMSE before choosing a winsorized model.

### Winsorize limits are fitted per training fold

The upper limit is a statistic of the data, so computing it on the full `y` before splitting would let test-fold targets shape the training targets. To avoid that leak, the stores for winsorize-based transforms hold `y` **before clipping** (`y`, `log1p(y)`, or `sqrt(y)`), and every cross-validation fit — steps 1 and 3, the OOF SHAP pass in step 2, and the Optuna trials and final CV in step 4 — clips its training targets with a limit computed from that training fold only. Test-fold targets are never clipped. The final model (`result.build_final_model()`) is refit on `y` clipped with the limit from the full training set.

## Metrics in transformed space

CV metrics in steps 1 and 3 are computed in the **original scale** when `y_true` is provided — the inverse transform is applied to predictions before metric computation. This ensures `R2`, `MAE`, `RMSE`, and `PoissonDev` are comparable across transforms.

`metric="PoissonDev"` suits non-negative counts or rates such as CPUE. It scales errors by the predicted value, so a few huge catches don't dominate selection as they can under RMSE. It raises an error on negative targets, and predictions are floored just above zero, where the deviance is undefined.

## Predictions and transforms

When `result.y_transform` is set, `FinalModel.predict()` and `FinalModel.predict_interval()` both return the **original scale**, so don't apply the inverse transform to their output:

```python
y_hat = final.predict(X_new)
lower, upper = final.predict_interval(X_new, alpha=0.10)
```

To get the transformed space, call `final.estimator.predict()` directly.

## Predicted totals and `mean_correction`

A model fitted on `log1p(y)` or `sqrt(y)` estimates the mean of that scale, and inverting it doesn't give the mean of `y`. Predictions come out too low, sometimes badly on skewed, zero-heavy targets. When a log or sqrt-family model wins and its out-of-fold total is more than 20% from the observed total, `cv_warnings` gets a `[transform]` entry with the ratio.

If totals matter, for example when summing predictions over an area, set `mean_correction="ratio"`:

```python
config = PipelineConfig(task_type="regression", metric="RMSE", mean_correction="ratio")
result = H2MLPipeline(config).run(store, transforms=["count", "log"])
final = result.build_final_model()
final.mean_ratio        # observed OOF total / predicted OOF total, clamped to 0.1–10
```

`predict()` multiplies back-transformed predictions by `mean_ratio`, and conformal scores are built from the corrected predictions, so `predict_interval()` stays centred on `predict()` with matching widths. `DeltaFinalModel` corrects the count part only, using the regressor's own out-of-fold rows.

It applies to `log`, `sqrt`, `winlog` and `winsqrt`; other transforms keep a ratio of 1.

- **Rowwise error:** it usually gets worse, because one factor scales every prediction. Use it for totals, not for accuracy at single sites.
- **High values:** the largest values often stay under-predicted, while low predictions are pushed up.
- **Scores:** CV scores, selection and `holdout_df` don't include the correction, so compare totals yourself on `holdout_store` before relying on it.
