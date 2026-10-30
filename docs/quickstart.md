# Quick Start

## Classification

Build the data container from a DataFrame or from NumPy arrays:

=== "DataFrame"

    ```python
    from h2ml import PipelineData

    store = PipelineData.from_frame(
        X,                           # pd.DataFrame with features
        y=y_arr,                     # np.ndarray (n_samples,)
    )
    ```

=== "NumPy arrays"

    ```python
    from h2ml import PipelineData

    store = PipelineData(
        X=X_arr,                     # np.ndarray (n_samples, n_features)
        feature_names=feature_cols,  # list[str]
        y=y_arr,                     # np.ndarray (n_samples,)
    )
    ```

Then configure and run the pipeline:

```python
from h2ml import H2MLPipeline, PipelineConfig

pipeline = H2MLPipeline(config=PipelineConfig(
    task_type="classification",   # or "regression"; the TaskType enum is also accepted
    metric="AUC",
    n_splits=5,
    n_trials=50,
    verbose=True,
))
result = pipeline.run(store)

print(result.summary())
print(result.best_model_name, result.best_stage)
```

## Regression with y-transform sweep

Step 1 evaluates all models × all transforms jointly. The winning (model, transform) pair is selected once, and steps 2–4 use the winning transform.

```python
pipeline = H2MLPipeline(config=PipelineConfig(
    task_type="regression",
    metric="R2",
    verbose=True,
))
result = pipeline.run(store, transforms=["log", "sqrt", "count", "winlog"])
print(result.y_transform)   # winning transform name
```

Available transforms: `"count"` (identity), `"log"`, `"sqrt"`, `"wincount"`, `"winlog"`, `"winsqrt"`. Winsorize-based variants are silently skipped when no upper outliers are present. The `log` and `sqrt` families need non-negative targets.

## Cross-validation

The pipeline picks the splitter from your data, so none of the examples above set one:

| `store.coords` | Task | Splitter | `result.cv_type` |
|---|---|---|---|
| Not set | Classification | Stratified k-fold (class balance kept in each fold) | `"random"` |
| Not set | Regression | Shuffled k-fold | `"random"` |
| Set | Either | Spatial block or SPCV | `"spatial"` |

`n_splits` sets the number of folds in steps 1 and 3:

```python
pipeline = H2MLPipeline(config=PipelineConfig(
    task_type="classification",
    metric="AUC",
    n_splits=10,
))
result = pipeline.run(store)
print(result.cv_type)  # "random"
```

Random folds assume rows are independent. With spatially autocorrelated data, nearby rows land on both sides of a split and scores come out too optimistic, so pass an `(n_samples, 2)` coordinate array to hold out whole regions instead. With `spatial_cv_metric="haversine"`, the columns must be latitude then longitude, in decimal degrees. With `"euclidean"`, the default, use projected coordinates such as metres, because distances in degrees are distorted over large areas.

```python
store = PipelineData(X=X_arr, feature_names=feature_cols, y=y_arr, coords=coords_arr)

pipeline = H2MLPipeline(config=PipelineConfig(
    task_type="classification",
    spatial_cv_method="block",   # or "spcv" (the default)
    spatial_cv_metric="haversine",
))
result = pipeline.run(store)
print(result.cv_type)  # "spatial"
```

See [Spatial CV](spatial_cv.md) for full parameter details.

## Partial runs

```python
# Step 1 only — quick model screening
result = pipeline.run_step1_only(store)

# Steps 1–2 — inspect SHAP importances before continuing
result = pipeline.run_step1_to_step2(store)
print(result.selector.importance_summary())
print(result.features_reduced.feature_names)

# Steps 1–3 — model and stage selection without HPO
result = pipeline.run_step1_to_step3(store)
result.save("runs/experiment_01")

# Resume from step 3 (uses a saved or existing result)
result = pipeline.run_from_step3(result)

# Re-run HPO only on a saved result
from h2ml import PipelineResult
result = PipelineResult.load("runs/experiment_01")
result = pipeline.run_step4_only(result)
```

## Check the result

A winner can be statistically tied with other candidates, or rank lower on another metric. Check both before relying on it:

```python
print(result.cv_warnings)       # failed folds, metric-suitability and ranking flags
result.ranking_robustness()     # paired-fold ties and per-metric ranks
```

See [Selection robustness](pipeline.md#selection-robustness) for how to read them.

## Compare selection metrics

The metric decides the winner, so a different metric can pick a different model. `multi_metric` runs the pipeline once per metric and reuses the CV work they share:

```python
from h2ml import multi_metric

pipeline = H2MLPipeline(config=PipelineConfig(task_type="classification"))
mm = multi_metric(pipeline, store, metrics=["Brier", "LogLoss", "AUC"])

print(mm.runs_df)                # winner and score per metric
print(mm.summary())              # how often the metrics agree

final = mm.results["Brier"].build_final_model()   # each result is a normal PipelineResult
```

Leave out `metrics=` to run every metric for the task. For regression, pass `transforms=` to include the y-transform sweep.

If `Top_Selection_Fraction` is 1, the choice doesn't depend on the metric. If the metrics disagree, choose the one that matches how predictions will be used: Brier or LogLoss for probabilities, AUC or AUCPR for ranking, F1 for thresholded decisions. Scores can only be compared within a metric.

See [Comparing selection metrics](pipeline.md#comparing-selection-metrics) for class weighting, shared tuning and what runs once.

## Deploy the final model

```python
from h2ml.pipeline.final_model import FinalModel

final = result.build_final_model()
final.predict(X_new)            # in the original y scale, also with a y-transform
final.predict_proba(X_new)      # classification only; 1-D positive-class probabilities when binary

final.save("models/final.pkl")
final = FinalModel.load("models/final.pkl")
```

`build_final_model()` also calibrates conformal prediction, so the model can report its uncertainty:

```python
lower, upper = final.predict_interval(X_new, alpha=0.10)   # regression
sets = final.predict_set(X_new, alpha=0.10)                # classification
```

See [Conformal Prediction](guides/conformal.md) for what the coverage means and when it weakens.

## Attach experiment metadata

```python
from h2ml.evaluation.metrics import RunMetadata

config = PipelineConfig(task_type="regression", metric="R2")
pipeline = H2MLPipeline(
    config=config,
    metadata=RunMetadata(schema="v2_features", target="pm25", batch="2024-01"),
)
```

Labels appear as columns in all fold and agg DataFrames, making multi-run concatenation trivial.

## Next steps

- [Exploratory Data Analysis](guides/eda.md) — catch data problems before the first run
- [Outer holdout](pipeline.md#outer-holdout) — get one score no selection step has seen
- [Nested CV](pipeline.md#nested-cv) — estimate how well the whole pipeline generalises
- [Seed stability](pipeline.md#seed-stability) — measure how much a run depends on its seed
- [Comparing Runs](guides/comparison.md) — compare results side by side
