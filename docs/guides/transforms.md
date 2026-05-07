# Y-Transforms

Y-transforms allow the pipeline to evaluate whether predicting in a transformed target space improves generalisation. Steps 1 and 3 sweep all (model × transform) combinations; the winning pair is selected once and used throughout.

## Usage

```python
result = pipeline.run(store, transforms=["log", "sqrt", "count", "winlog"])
print(result.y_transform)   # e.g. "log"
```

Only supported for regression (`TaskType.REGRESSION`).

## Available transforms

| Name | Transform | Inverse | Notes |
|------|-----------|---------|-------|
| `"count"` | identity | identity | Baseline — no transformation |
| `"log"` | `log1p(y)` | `expm1(y)` | Requires non-negative y |
| `"sqrt"` | `sqrt(y)` | `y²` | Requires non-negative y |
| `"wincount"` | winsorize(y) | identity | Clips upper outliers to IQR upper limit |
| `"winlog"` | winsorize(log1p(y)) | `expm1(y)` | Winsorize then log |
| `"winsqrt"` | winsorize(sqrt(y)) | `y²` | Winsorize then sqrt |

Winsorize-based transforms return `None` when no upper outliers are found and are **silently skipped** by `build_transform_stores`. If all winsorized variants are skipped, only non-winsorized transforms are evaluated.

## Metrics in transformed space

CV metrics in steps 1 and 3 are computed in the **original scale** when `y_true` is provided — the inverse transform is applied to predictions before metric computation. This ensures `R2`, `MAE`, and `RMSE` are comparable across transforms.

## Conformal intervals and transforms

When `result.y_transform` is set, `FinalModel.predict_interval()` returns bounds in the **transformed space**. Apply the inverse to get original-scale intervals:

```python
from h2ml.preprocessing.transforms import INVERSE_TRANSFORMS

lower, upper = final.predict_interval(X_new, alpha=0.10)
inv = INVERSE_TRANSFORMS[result.y_transform]
lower_orig, upper_orig = inv(lower), inv(upper)
```
