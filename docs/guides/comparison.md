# Comparing Runs

`compare_results` summarises multiple `PipelineResult` objects side-by-side — useful for comparing configurations, feature schemas, or datasets.

## Usage

```python
from h2ml.evaluation.compare import compare_results

r1 = pipeline_a.run(store)
r2 = pipeline_b.run(store)

df = compare_results(
    [r1, r2],
    labels=["baseline", "spatial_cv"],
    metric="AUC",       # common comparison column (optional)
)
```

## Output columns

| Column | Description |
|--------|-------------|
| `Run` | Label from `labels`, or `Run_0`, `Run_1`, … |
| `Metric` | Short metric name used for `Score_Mean` |
| `Best_Model` | Winning model class name |
| `Best_Stage` | `"default"` \| `"reduced"` \| `"optimized"` |
| `Y_Transform` | Winning y-transform, or `None` |
| `Score_Mean` | With `metric=`: the fold mean, `<metric>_Test_Mean`, of the winning stage. Without it: each result's `best_model_value`, which for regression is the pooled score by default |
| `Score_Std` | Fold std for that metric |
| `Conservative_Bound` | Variance-penalised score: `mean − std/√n` (maximise) or `mean + std/√n` (minimise). Stable models are preferred over high-variance ones. |
| `Brier_Mean` | `Brier_Test_Mean` from the winning stage (classification only) |
| `OOF_Brier` | Brier score on assembled OOF predictions (more robust than `Brier_Mean`) |
| `N_Features` | Feature count in the winning feature stage |
| `Completed_Steps` | Steps that finished, e.g. `[1, 2, 3, 4]` |

The DataFrame is sorted by `Conservative_Bound` descending (or ascending for error metrics). When fold count cannot be inferred, falls back to `Score_Mean`.

## Mixing metrics

When results use different metrics (e.g. one run optimised AUC, another F1), pass `metric=` to force a common comparison column:

```python
df = compare_results([r1, r2], metric="AUC")
```

Without `metric=`, an error is raised if results use different metrics — scores would not be directly comparable.

For regression, passing `metric=` switches `Score_Mean` from the pooled score selection used to the fold mean, so the two calls can rank the same runs differently. Compare runs within one call.
