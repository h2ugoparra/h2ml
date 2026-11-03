# Conformal Prediction

`build_final_model()` automatically builds a conformal calibrator from out-of-fold CV predictions — no separate calibration set is required.

!!! note "OOF vs holdout calibration"
    The OOF predictions come from the same folds the pipeline selected the model, features
    and hyperparameters on, so coverage can fall slightly below the nominal 1 − alpha.
    With `PipelineConfig(holdout_frac=...)` the calibrator uses the residuals on rows no
    selection step saw instead (split conformal), which restores the guarantee for the
    model fitted on the dev rows (`holdout_refit="dev"`). Local calibration is then
    skipped — see [Outer holdout](../validation.md#outer-holdout).

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

---

## Spatio-temporal local calibration

When `store.coords` (and optionally `store.times`) are provided, `build_final_model()` also builds a `LocalConformalCalibration` alongside the global one. This enables **spatially and temporally varying interval widths** — regions where the model is historically less accurate get wider intervals.

### How it works

OOF residuals are partitioned into a two-level structure:

1. **Spatial blocks** — from the CV splitter's `block_id_` (quantile-grid cells or AHC clusters). Each block gets its own sorted score distribution.
2. **Compound cells `(block, time_bin)`** — when `store.times` is provided, residuals within each block are further split by month or season.

At inference, threshold lookup uses three levels of fallback:

| Level | Cell | Used when |
|-------|------|-----------|
| 1 | Compound `(block × time_bin)` | ≥ `min_compound_n` samples (default 5) |
| 2 | Spatial block | ≥ `min_block_n` samples (default 20) |
| 3 | Global fallback | all other cases |

Block assignment at inference uses **nearest-OOF-sample k-NN** (not centroid), so non-convex AHC clusters are handled correctly. The time bin is derived from the **query point's own date**, not the nearest OOF sample's date.

### Activation

Pass `coords` and optionally `times` when constructing `PipelineData`:

```python
store = PipelineData.from_frame(
    df=features,
    y=counts,
    coords=np.column_stack([lats, lons]),  # (n_samples, 2) — activates spatial blocks
    times=date_array,                      # (n_samples,) YYYY-MM-DD — activates compound cells
)
result = H2MLPipeline(PipelineConfig(
    spatial_cv_method="block",    # or "spcv"
    time_bin_resolution="season", # "month" (1–12) or "season" (DJF/MAM/JJA/SON)
)).run(store)

final = result.build_final_model()
print(final.local_conformal)          # LocalConformalCalibration if built, None otherwise
print(final.local_conformal.has_times)
print(final.local_conformal.time_bin_resolution)
```

`local_conformal` is `None` when:

- No spatial CV was used (`store.coords` was `None`)
- `result.splitter` is absent (e.g. after `PipelineResult.load()` — rebuild with `build_final_model()` before saving)
- Fewer than 2 spatial blocks were found in the OOF data
- A holdout was used (`holdout_frac > 0`) — the global holdout calibrator applies instead

### Local intervals at inference

Pass `coords` and/or `times` to `predict_interval` / `predict_set` to activate local thresholds:

```python
# Spatially varying widths
lower, upper = final.predict_interval(X_new, alpha=0.10, coords=test_coords)

# Spatially + temporally varying widths
lower, upper = final.predict_interval(
    X_new, alpha=0.10,
    coords=test_coords,
    times=test_dates,   # YYYY-MM-DD strings or datetime64[D]
)

# Global threshold (original behaviour) — omit coords/times
lower, upper = final.predict_interval(X_new, alpha=0.10)
```

Extra context dimensions not present at calibration time are silently ignored — passing `times` to a model built without `store.times` is safe and falls back to spatial-only calibration.

### Diagnostics

```python
lc = final.local_conformal

# One row per spatial block, plus one per populated (block, time_bin) compound cell.
# Bin labels follow the model's time_bin_resolution (season → DJF/MAM/JJA/SON, month → 1–12).
print(lc.summary(alpha=0.10))

# Which cells fall through to the global fallback (too few samples)?
print(lc.summary().query("used == 'global'"))

# Limit the report to the first few blocks:
print(lc.summary(max_blocks=5))
```

`summary()` returns a DataFrame with columns `block`, `level` (`spatial`/`compound`),
`time_bin`, `bin_name`, `n` (samples), `q` (conformal quantile at `alpha`), and `used` —
the fallback level a query resolving to that cell actually hits (`compound`, `block`, or
`global`). The `used` column makes the sparsity problem below directly visible.

### Tuning

**Too few samples per compound cell** — with many spatial blocks and monthly bins, average cell size can be very small. Prefer seasonal bins (`time_bin_resolution="season"`, 4 bins) over monthly (12 bins), or lower `min_compound_n` directly on the built model:

```python
final.local_conformal.min_compound_n = 3  # runtime change, no rebuild needed
```

Note: at α=0.10, compound cells need ≥ 10 samples before the conformal quantile formula can return anything other than the cell maximum. Very small cells are conservative (always return the worst observed residual).

**SPCVSplitter** produces more and smaller blocks than the block splitter (AHC cluster count is controlled by `ahc_threshold`). With many small blocks, most compound cells will have few samples and fall back to spatial level. Increasing `ahc_threshold` reduces the block count and increases cell density.

---

## Geo prediction — conformal columns

`predict_for_year` and `predict_for_year_delta` accept `alpha` to add conformal bound columns. By default they use the global conformal threshold (constant-width intervals). Pass `local=True` to use a `LocalConformalCalibration` when the model has one — the `lon`, `lat`, and `time` columns from the scan are then used so interval widths vary across space and time.

```python
# Global constant-width intervals (default)
df = predict_for_year(target="sparrow", year=2023, ..., alpha=0.10)

# Spatio-temporally varying intervals (requires a LocalConformalCalibration)
df = predict_for_year(target="sparrow", year=2023, ..., alpha=0.10, local=True)
```

### Regression

```python
df = predict_for_year(target="sparrow", year=2023, ..., alpha=0.10)
# columns: sparrow_v1, sparrow_v1_pi_lower, sparrow_v1_pi_upper
```

Bounds are in the original (inverse-transformed) count scale, because `q` is calibrated on original-scale residuals. They are symmetric around the prediction, with or without a y-transform, apart from the clip at zero:

```
pi_lower = max(0, ŷ − q)
pi_upper = ŷ + q
```

When `local=True`, `q` is a per-sample value from `LocalConformalCalibration.threshold()`. The default is `local=False`, which uses the global constant `q`.

### Binary classifiers

```python
df = predict_for_year(target="sparrow", year=2023, ..., alpha=0.10)
# columns: sparrow_v1, sparrow_v1_pi_lower, sparrow_v1_pi_upper
```

Calibration bands in probability space:

```
pi_lower = clip(p − q, 0, 1)
pi_upper = clip(p + q, 0, 1)
```

!!! note "Interpretation differs from regression"
    Regression `_pi_lower`/`_pi_upper` have a formal conformal coverage guarantee on the outcome. For binary classifiers the bounds are a calibration band in probability space — they convey *"given past calibration errors of size q, the predicted probability could plausibly be off by this much"*, not a coverage guarantee on the class label.

Multiclass classifiers and uncalibrated models always return point predictions only.

---

## Delta model — presence × abundance

`DeltaFinalModel` combines a presence/absence classifier and a count regressor. Prediction is `P(present) × E(count | present)`, always in the original count scale.

```python
from h2ml.pipeline.final_model import build_delta_final_model, DeltaFinalModel

positive_idx = np.where(y_all > 0)[0]
X_df = pd.DataFrame(X_all, columns=feature_names)

delta = build_delta_final_model(
    clf_result=clf_result,
    reg_result=reg_result,
    X_full=X_df,
    y_full=y_all,
    positive_indices=positive_idx,
)
```

Point predictions and conformal intervals:

```python
preds = delta.predict(X_new)
lower, upper = delta.predict_interval(X_new, alpha=0.10)

# Local (spatio-temporal) intervals
lower, upper = delta.predict_interval(X_new, alpha=0.10, coords=coords, times=dates)
```

**Save / load** — saves to a directory (`clf.pkl`, `reg.pkl`, `conformal.pkl`, `local_conformal.pkl`, `variogram.pkl`):

```python
delta.save("models/sparrow_delta-model")
delta = DeltaFinalModel.load("models/sparrow_delta-model")
```

`delta.variogram` is fitted on the signed delta residuals (P × count minus observed), the ones the intervals are calibrated on, using the classifier's coordinates. With many absences these residuals carry presence noise, so structure in abundance alone can read `"no_structure"` there while `delta.reg.variogram` still shows it. See the [variogram diagnostic](#variogram-diagnostic).

---

## Variogram diagnostic

When `store.coords` is provided, `build_final_model()` also fits an exponential variogram to the OOF residuals and stores it as `FinalModel.variogram`. This is a diagnostic — it has no effect on predictions.

```python
final = result.build_final_model()
vr = final.variogram    # VariogramResult or None

if vr:
    print(f"Practical range: {vr.practical_range:.1f}")  # distance where correlation ≈ 5%
    print(f"Nugget: {vr.nugget:.3f}, Sill: {vr.sill:.3f}")

from h2ml.utils.variogram import plot_variogram
residuals = result.oof_predictions - result.oof_labels   # signed, as the fit uses
plot_variogram(result.features.coords, residuals, metric=result.spatial_cv_metric)
```

`result.features` holds the rows the CV ran on (the dev rows with a holdout), so its coordinates line up with the OOF residuals.

Residuals are signed (predicted minus observed), and distances follow `spatial_cv_metric`: coordinate units for `"euclidean"`, km for `"haversine"`. For `plot_variogram` and `autocorrelation_range`, pass `metric="haversine"` yourself.

Check `vr.status` before using the range:

| Status | Meaning |
|---|---|
| `"ok"` | The variogram levels off within the lags, so the range is an estimate |
| `"at_bound"` | It doesn't level off, so the true range is at least `practical_range`, which is capped at the longest lag |
| `"no_structure"` | No spatial structure found: nearby residuals are no more alike than shuffled ones, or structure is under 20% of the variance or shorter than the first lag. Ignore the range |
| `"unknown"` | Saved before `status` existed |

With `"ok"`, a short range relative to your spatial block size confirms the spatial CV is well-separated, and a long one suggests residual autocorrelation the model hasn't captured. `"at_bound"` means the same as a long range. `"no_structure"` isn't proof of independence: on small samples or very zero-heavy targets, weak structure can go undetected.

---

## How it works

The calibrator does not touch the model itself. It sits on top of any model and wraps its point predictions in a statistically honest band.

**Nonconformity scores** are computed per sample from the OOF folds:

- Regression: `|y_true − y_pred|`
- Binary classification: `1 − p(true class)`
- Multiclass: `1 − p(true class)` (looked up via `estimator.classes_`)

With a holdout they come from the holdout rows instead, and with `mean_correction="ratio"` they are measured against the corrected predictions (`|y − mean_ratio × ŷ|`), so intervals stay centred on `predict()`.

The scores are sorted ascending. **n** is the total number of held-out samples across all folds (e.g. 5-fold CV on 1000 samples gives n = 1000), or the number of holdout rows.

```python
scores = [0.1, 0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.2, 1.8, 3.1, ...]
           ↑ model was almost right    ↑ typical error     ↑ model was badly wrong
```

The threshold `q` at level α:

```python
q = ⌈(1−alpha)(n+1)/n⌉ quantile of scores
```

Applied as:

- **Regression:** `interval = [ŷ − q,  ŷ + q]` — symmetric around the point estimate
- **Classification:** prediction set = all classes with nonconformity score ≤ `q`

With local calibration, `q` is looked up per sample from the compound (block × time_bin) cell or its spatial/global fallback.

---

## Limitations

- **Symmetric intervals (regression):** `ŷ ± q` — same offset above and below, which can produce negative lower bounds for count data. Local calibration changes *which* `q` is used per sample but keeps the symmetric form. Conformalized Quantile Regression (CQR) is the principled fix for asymmetric intervals and is forward-compatible with `LocalConformalCalibration` — only the score-generation step would change.
- **Marginal coverage only:** the guarantee holds on average over the training distribution. Out-of-distribution inputs (spatial extrapolation, future years with no training data in nearby blocks) may not achieve nominal coverage.
- **Temporal block sparsity:** with SPCVSplitter producing many AHC blocks and monthly bins, compound cells can be very small. Use `time_bin_resolution="season"` and check the diagnostic output to ensure cells have enough samples.
- **Transformed targets:** `predict()`, `predict_interval()`, `predict_for_year` and `DeltaFinalModel` all return the original scale. Intervals built from original-scale residuals are symmetric, so they can't follow the skew of a log or sqrt target.

---

## Interpretation

Think of it like a fishing net.

You're trying to catch the true value inside a net `[pi_lower, pi_upper]`. Alpha controls how tight the net is.

- alpha = 0.10 → catches the true value 9 times out of 10
- alpha = 0.05 → catches it 19 times out of 20 (wider net)
- alpha = 0.20 → catches it 4 times out of 5 (narrower net, more misses)

The guarantee is marginal: across thousands of predictions, roughly (1 − alpha) × 100% of the true values will fall inside their interval.

```python
alpha = 0.10   # "I want intervals to cover the truth 90% of the time"
alpha = 0.20   # "Narrower intervals are more useful; I accept more misses"
```
