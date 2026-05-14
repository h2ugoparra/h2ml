# Conformal Prediction

`build_final_model()` automatically builds a conformal calibrator from out-of-fold CV predictions — no separate calibration set is required.

A conformal calibrator is essentially a recorded history of how wrong the model has been, used to answer one question at prediction time:
*"how much buffer do I need around my estimate to be right X% of the time?"*

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

## Geo prediction — conformal columns

`predict_for_year` accepts an `alpha` argument that adds conformal columns to the output parquet for any calibrated model.

### Regression

Calibrated regression models produce two extra columns per target:

```python
df = predict_for_year(target="sparrow", year=2023, ..., alpha=0.10)
# df columns: sparrow_v1, sparrow_v1_pi_lower, sparrow_v1_pi_upper
```

The bounds are per-sample and in the original (inverse-transformed) count scale.

### Binary classifiers

Calibrated binary classifiers produce the same `_pi_lower` / `_pi_upper` column names, mirroring the regression formula applied to probability space:

```
pi_lower = clip(p − q, 0, 1)
pi_upper = clip(p + q, 0, 1)
```

Where `p` is the per-sample predicted probability and `q = conformal.threshold(alpha)`. This works because nonconformity scores for binary classification are `1 − p(true class)`, which live on [0, 1] — the same units as `p` — so `q` and `p` are directly comparable.

```python
df = predict_for_year(target="sparrow", year=2023, ..., alpha=0.10)
# df columns: sparrow_v1           (per-sample predicted probability p)
#             sparrow_v1_pi_lower  (per-sample: clip(p − q, 0, 1))
#             sparrow_v1_pi_upper  (per-sample: clip(p + q, 0, 1))
```

The bounds vary spatially — high-confidence pixels (p near 0 or 1) produce narrow intervals; uncertain pixels (p near 0.5) produce wide intervals that span the decision boundary.

!!! note "Interpretation differs from regression"
    Regression `_pi_lower`/`_pi_upper` have a formal conformal coverage guarantee on the outcome. For binary classifiers the bounds are a calibration band in probability space — they convey *"given past calibration errors of size q, the predicted probability could plausibly be off by this much"*, not a coverage guarantee on the class label.

Multiclass classifiers and uncalibrated models always return point predictions only.

## How it works

The calibrator does not touch the model itself. It sits on top of any model and wraps its point predictions in a statistically honest band.
The model could be a random forest, a neural network, a delta model — the calibrator doesn't care. It only cares about the distribution of past errors.

**Nonconformity scores** are computed per sample from the OOF folds:

- Regression: `|y_true − y_pred|`
- Binary classification: `1 − p(true class)`
- Multiclass: `1 − p(true class)` (looked up via `estimator.classes_`)

The scores are sorted ascending and stored — one score per training sample, not per fold. **n** is the total number of held-out samples across all folds (e.g. 5-fold CV on 1000 samples gives n = 1000).

```python
scores = [0.1, 0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.2, 1.8, 3.1, ...]
           ↑ model was almost right    ↑ typical error     ↑ model was badly wrong
```

At inference time, find the threshold `q` — the score that was exceeded only `alpha × 100%` of the time in the calibration data:

```python
q = ⌈(1−alpha)(n+1)/n⌉ quantile of scores
```

Then apply it:

- **Regression:** `interval = [ŷ − q,  ŷ + q]` — constant width
- **Classification:** prediction set = all classes with nonconformity score ≤ `q`

The logic: if the model's errors in the past rarely exceeded `q`, then adding `q` as a buffer around a new prediction will catch the true value most of the time.

## Limitations

- **Constant-width intervals (regression):** the same `q` is applied to every sample. Regions of the input space with higher inherent variance get the same interval as low-variance regions. For heteroscedastic data this means intervals are too wide in easy regions and too narrow in hard ones.
- **Marginal coverage only:** the guarantee holds on average over the training distribution. Out-of-distribution inputs (e.g., spatial extrapolation beyond the training extent) may not achieve nominal coverage.
- **Transformed targets:** if `result.y_transform` is set, intervals are in the transformed space. Apply the inverse transform to the bounds manually — see [Y-Transforms](transforms.md).

## Interpretation

Think of it like a fishing net.

You're trying to catch the true value (the real count of animals at that pixel-day) inside a net (the interval [pi_lower, pi_upper]). Alpha
controls how tight you make the net.

- alpha = 0.10 → the net is sized so it catches the true value 9 times out of 10. One time in ten, the value slips through.
- alpha = 0.05 → catches it 19 times out of 20. Rarely misses, but the net has to be wider.
- alpha = 0.20 → catches it 4 times out of 5. Misses more often, but the net is narrower and more informative.

So alpha is the miss rate you're willing to accept. The lower the alpha, the wider the interval, the fewer misses.

The guarantee is marginal: if you make predictions across thousands of pixel-days, roughly (1 - alpha) × 100% of the true values will fall inside their interval. It doesn't mean any single interval is guaranteed — it's a statement about the long-run hit rate across all predictions.

In practice:

```python
alpha = 0.10   # "I want my intervals to cover the truth 90% of the time"
alpha = 0.20   # "Narrower intervals are more useful to me, I accept more misses"
```

There's no universally "correct" alpha — it depends on the cost of being wrong. For species distribution maps used in conservation decisions, 0.10 (90% coverage) is a common starting point.
