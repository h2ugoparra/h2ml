# Persistence

## Saving and loading a result

```python
result.save("runs/experiment_01")
```

Creates a directory with:

| File | Contents |
|------|----------|
| `step1_fold_df.parquet` | Per-fold metrics, step 1 |
| `step3_fold_df.parquet` | Per-fold metrics, step 3 |
| `step4_fold_df.parquet` | Per-fold metrics, step 4 |
| `features.npy` + metadata | Full feature matrix |
| `features_reduced.npy` + metadata | Reduced feature matrix |
| `selector.pkl` | Fitted `FeatureSelector` |
| `step1_cv_result.pkl` | Raw `CVResult` list, step 1 |
| `step3_cv_result.pkl` | Raw `CVResult` list, step 3 |
| `step4_cv_result.pkl` | `CVResult`, step 4 |
| `meta.json` | Scalar fields (model name, stage, params, …) |

Reload with:

```python
from h2ml import PipelineResult

result = PipelineResult.load("runs/experiment_01")
```

!!! warning
    The `.pkl` files use `joblib` (pickle under the hood). Only load results from trusted sources — pickle executes arbitrary code on deserialisation.

## Resuming a partial run

Save after any partial run and resume later:

```python
# Save after steps 1–2
result = pipeline.run_step1_to_step2(store)
result.save("runs/partial")

# Reload and continue from step 3
result = PipelineResult.load("runs/partial")
result = pipeline.run_from_step3(result)
```

## Saving the final model

`FinalModel` is saved independently of `PipelineResult` — it is the deployment artifact:

```python
from h2ml.pipeline.final_model import FinalModel

final = result.build_final_model()
final.save("models/final.pkl")

final = FinalModel.load("models/final.pkl")
final.predict(X_new)
```
