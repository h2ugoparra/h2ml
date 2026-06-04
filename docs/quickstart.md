# Quick Start

## Classification

```python
import numpy as np
from h2ml import H2MLPipeline, PipelineConfig, PipelineData

store = PipelineData(
    X=X_arr,                     # np.ndarray (n_samples, n_features)
    feature_names=feature_cols,  # list[str]
    y=y_arr,                     # np.ndarray (n_samples,)
)

or

store = PipelineData.from_frame(
    X,                           # pd.DataFrame with features
    y=y_arr,                     # np.ndarray (n_samples,)
)

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

Steps 1 and 3 evaluate all models × all transforms jointly. The winning (model, transform) pair is selected once and carried forward.

```python
pipeline = H2MLPipeline(config=PipelineConfig(
    task_type="regression",
    metric="R2",
    verbose=True,
))
result = pipeline.run(store, transforms=["log", "sqrt", "count", "winlog"])
print(result.y_transform)   # winning transform name
```

Available transforms: `"count"` (identity), `"log"`, `"sqrt"`, `"wincount"`, `"winlog"`, `"winsqrt"`. Winsorize-based variants are silently skipped when no upper outliers are present.

## Spatial cross-validation

Pass an `(n_samples, 2)` coordinate array to activate spatial CV throughout the pipeline:

```python
store = PipelineData(X=X_arr, feature_names=cols, y=y_arr, coords=coords_arr)

pipeline = H2MLPipeline(config=PipelineConfig(
    task_type="classification",
    spatial_cv_method="block",   # or "spcv"
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

# Resume from step 3 (uses a saved or existing result)
result = pipeline.run_from_step3(result)

# Re-run HPO only on a saved result
from h2ml import PipelineResult
result = PipelineResult.load("runs/experiment_01")
result = pipeline.run_step4_only(result)
```

## Deploy the final model

```python
from h2ml.pipeline.final_model import FinalModel

final = result.build_final_model()
final.predict(X_new)
final.predict_proba(X_new)      # classification only

final.save("models/final.pkl")
final = FinalModel.load("models/final.pkl")
```

## Attach experiment metadata

```python
from h2ml.evaluation.metrics import RunMetadata

pipeline = H2MLPipeline(
    config=config,
    metadata=RunMetadata(schema="v2_features", target="pm25", batch="2024-01"),
)
```

Labels appear as columns in all fold and agg DataFrames, making multi-run concatenation trivial.
