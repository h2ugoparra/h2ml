# Persistence

## Saving and loading a result

```python
result.save("runs/experiment_01")
```

Creates a directory with:

| File | Contents |
|------|----------|
| `metadata.json` | Scalar fields (model name, stage, params, transform, metric, warnings, …) |
| `step1_fold.parquet` / `step1_agg.parquet` | Per-fold and aggregated metrics, step 1 |
| `step3_fold.parquet` / `step3_agg.parquet` | Per-fold and aggregated metrics, step 3 |
| `step4_fold.parquet` / `step4_agg.parquet` | Per-fold and aggregated metrics, step 4 |
| `holdout_agg.parquet` | Holdout metrics (`holdout_df`), when a holdout was scored |
| `features/` | Full feature store: `X.npy`, `y.npy`, optional `y_true.npy`, `coords.npy`, `times.npy`, and `meta.json` |
| `features_reduced/` | Reduced feature store, same layout |
| `holdout/` + `holdout_idx.npy` | Held-out rows and their indices, when a holdout was used |
| `selector.pkl` | Fitted `FeatureSelector` |
| `best_base_estimator.pkl` | Unfitted winning estimator |
| `step1_cv_result.pkl` / `step3_cv_result.pkl` | Raw `CVResult` lists, steps 1 and 3 |
| `step4_cv_result.pkl` | `CVResult`, step 4 |
| `holdout_cv_result.pkl` | One-fold `CVResult` of the holdout score |
| `splitter.pkl` | CV splitter built in step 1, with its fold and block assignment |

`result.step3_reduced_stores` is not saved, so it is `None` after loading. The splitter is saved so that `run_from_step3()` and `run_step4_only()` on a loaded result score on the folds steps 1–3 used. Results saved by h2ml 0.4.1 or earlier have no `splitter.pkl`: resuming one rebuilds the splitter from the full feature set.

Reload with:

```python
from h2ml import PipelineResult

result = PipelineResult.load("runs/experiment_01")
```

Saving refuses to overwrite a path that exists unless it is an empty directory or a previous save.

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

Results saved before `selection_scoring` existed resume on `"fold_mean"`, the rule they were selected with.

## Multi-metric results

```python
mm.save("runs/multi")                       # multi_metric.json, runs.csv, results/<metric>/
mm = MultiMetricResult.load("runs/multi")   # each results/<metric>/ is a saved PipelineResult
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

Models saved by older versions still load. Fields added since then take neutral values: `mean_ratio` is 1.0, `variogram.status` is `"unknown"`, and `applicability` is `None`. Rebuild with `result.build_final_model()` to get them.
