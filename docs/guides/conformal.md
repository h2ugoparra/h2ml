# Conformal Prediction

`build_final_model()` automatically builds a conformal calibrator from out-of-fold CV predictions — no separate calibration set is required.

## Regression — prediction intervals

```python
final = result.build_final_model()

# 90% prediction interval (alpha=0.10 → 10% miscoverage)
lower, upper = final.predict_interval(X_new, alpha=0.10)
```

**Coverage guarantee:** the true value falls inside `[lower, upper]` with probability ≥ 1 − alpha, averaged over new draws from the training distribution.

## Classification — prediction sets

```python
sets = final.predict_set(X_new, alpha=0.10)
# sets[i] == [1]    → confident: class 1
# sets[i] == [0]    → confident: class 0
# sets[i] == [0, 1] → uncertain: either class is plausible
```

A singleton set means the model is confident; a larger set means it is uncertain.

## How it works

**Nonconformity scores** are computed per sample from the OOF folds:

- Regression: `|y_true − y_pred|`
- Binary classification: `1 − p(true class)`
- Multiclass: `1 − p(true class)` (looked up via `estimator.classes_`)

The scores are sorted ascending and stored. At inference time the threshold `q` is the `⌈(1−alpha)(n+1)/n⌉` quantile, which guarantees marginal coverage ≥ 1−alpha.

Interval: `ŷ ± q` (regression) — constant width.
Prediction set: all classes with nonconformity score ≤ `q` (classification).

## Limitations

- **Constant-width intervals (regression):** the same `q` is applied to every sample. Regions of the input space with higher inherent variance get the same interval as low-variance regions. For heteroscedastic data this means intervals are too wide in easy regions and too narrow in hard ones.
- **Marginal coverage only:** the guarantee holds on average over the training distribution. Out-of-distribution inputs (e.g., spatial extrapolation beyond the training extent) may not achieve nominal coverage.
- **Transformed targets:** if `result.y_transform` is set, intervals are in the transformed space. Apply the inverse transform to the bounds manually — see [Y-Transforms](transforms.md).
