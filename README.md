# H2ML

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![PyPI](https://img.shields.io/pypi/v/h2ml)](https://pypi.org/project/h2ml/)
[![Docs](https://img.shields.io/badge/docs-h2ugoparra.github.io%2Fh2ml-blue)](https://h2ugoparra.github.io/h2ml/)

A 4-step AutoML pipeline for tabular data that wraps sklearn-compatible estimators. Given a feature matrix and target, it screens all registered models, reduces features via SHAP importance and correlation filtering, and tunes the winner with Optuna — all in one call.

## Installation

```bash
uv add h2ml
# or
pip install h2ml
```

The [`h2mare`](https://github.com/h2ugoparra/h2mare) companion package and `polars` are core dependencies, included with the base install.

For boosting libraries (LightGBM, XGBoost, CatBoost):

```bash
uv add h2ml[boosting]
# or
pip install h2ml[boosting]
```

Spatial map inference via `h2ml.geo.geo_predict` needs no extra — `h2mare` is a core dependency and pulls in `cartopy` and `polars`. The `[geo]` extra is retained for backward compatibility and resolves to the same environment as the plain install:

```bash
uv add h2ml[geo]
# or
pip install h2ml[geo]
```

A runnable example using public sklearn datasets is in [`examples/quickstart.ipynb`](examples/quickstart.ipynb).

## Quick start

```python
import numpy as np
from h2ml import H2MLPipeline, PipelineConfig, PipelineData

# Build the data container
store = PipelineData(
    X=X_arr,
    feature_names=feature_cols,
    y=y_arr,
)

# Configure and run
pipeline = H2MLPipeline(config=PipelineConfig(
    task_type="classification",   # or "regression"; the TaskType enum is also accepted
    metric="AUC",
    n_splits=5,
    n_trials=50,
    verbose=True,
))
result = pipeline.run(store)

# Inspect results
print(result.summary())
print(result.best_model_name, result.best_stage)
```

### Regression with y-transform sweep

```python
config = PipelineConfig(
    task_type="regression",
    metric="R2",
    verbose=True,
)
pipeline = H2MLPipeline(config=config)
result = pipeline.run(store, transforms=["log", "sqrt", "count", "winlog"])
```

Available transform names: `"count"` (identity), `"log"`, `"sqrt"`, `"wincount"`, `"winlog"`, `"winsqrt"`. Winsorize-based transforms are skipped silently when no upper outliers are found.

### Partial runs

All partial-run methods accept an optional `transforms` list (same as `pipeline.run()`).

```python
# Screen models only (step 1)
result = pipeline.run_step1_only(store)

# Steps 1–2: run feature selection, then inspect before continuing
result = pipeline.run_step1_to_step2(store)
print(result.selector.importance_summary())
print(result.features_reduced.feature_names)

# Steps 1–3: full model and stage selection without HPO
result = pipeline.run_step1_to_step3(store)

# Resume from step 3 using a result that already has features_reduced.
# Pass transform_stores if the original run used y-transforms.
result = pipeline.run_from_step3(result)

# Re-run HPO only on a previously saved result (skips steps 1–3).
# Requires: features, features_reduced, selector, best_model_name,
#           best_stage, best_model_value.
result = PipelineResult.load("runs/experiment_01")
result = pipeline.run_step4_only(result)
```

## The 4-step pipeline

| Step | What happens | Key output on `PipelineResult` |
|------|-------------|-------------------------------|
| 1 | K-fold CV all models (× optional y-transforms) on all features | `best_model_name`, `step1_agg_df` |
| 2 | Fit best model → SHAP importance → correlation-based feature drop | `features_reduced`, `selector` |
| 3 | K-fold CV all models on reduced features (winning transform only); compare vs step 1 | `best_stage` (`"default"` or `"reduced"`), `best_feature_stage` |
| 4 | Optuna HPO on the winning (model, stage, transform) | `best_params`, `step4_agg_df` |

Step 4 is skipped when the winning model has `opt_enabled=False` in the registry (e.g. LogisticRegression, GaussianNB, KNeighborsClassifier).

## `PipelineConfig` reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_type` | `"classification"` | `"classification"` or `"regression"` (case-insensitive); a `TaskType` member is also accepted |
| `metric` | `"AUC"` | Short metric name for model selection and HPO. Minimisation direction is derived automatically. Classification: `"AUC"`, `"AUC_PR"`, `"F1"`, `"LogLoss"`, `"Brier"`. Regression: `"R2"`, `"MAE"`, `"RMSE"`. |
| `n_splits` | `5` | Folds for steps 1 and 3 |
| `opt_n_splits` | `3` | Folds used inside Optuna (fewer = faster) |
| `corr_threshold` | `0.7` | Correlation threshold for dropping features in step 2. A feature is dropped if it exceeds this value in any of Pearson, Spearman, or Kendall correlation with a higher-ranked feature. |
| `n_trials` | `50` | Optuna trials in step 4 |
| `n_hpo_repeats` | `1` | Independent HPO runs with different fold seeds; best is kept |
| `min_features` | `1` | Minimum features retained after the correlation filter |
| `handle_imbalance` | `False` | Inject `class_weight="balanced"` for supporting classifiers |
| `random_state` | `42` | Global seed |
| `verbose` | `False` | Log step-by-step progress to stdout |

### Spatial CV parameters

Set `store.coords` to an `(n_samples, 2)` array of spatial coordinates to activate spatial cross-validation. All parameters below are ignored when `coords` is `None`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spatial_cv_method` | `"spcv"` | `"block"` (quantile-grid) or `"spcv"` (AHC + cluster ensemble) |
| `spatial_cv_metric` | `"euclidean"` | `"euclidean"` or `"haversine"` (expects lat/lon in degrees) |
| `n_blocks_per_fold` | `5` | Blocks per test fold for the block splitter |
| `ahc_threshold` | `None` | AHC distance threshold for `spcv`; auto-set to 10th percentile of pairwise distances when `None` |
| `exact_max_samples` | `5000` | n ≤ this → exact scipy AHC; n > → approximate sklearn AHC with k-NN graph |
| `knn_neighbors` | `15` | k for the k-NN connectivity graph in approximate AHC |
| `pca_components` | `0.95` | Variance retained by PCA on block covariates in `spcv` stage 2 |

## Supported models

**Classifiers** — LogisticRegression, GaussianNB, KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, SVC, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, LGBMClassifier\*, CatBoostClassifier\*, XGBClassifier\*

**Regressors** — PoissonRegressor, KNeighborsRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, SVR, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor, LGBMRegressor\*, CatBoostRegressor\*, XGBRegressor\*

\* Registered only when the package is installed. Custom models can be injected by passing a `models` list directly to `H2MLPipeline`.

## `PipelineResult`

```python
result.summary()                  # combined agg DataFrame across all completed stages
result.summary("AUC_Test_Mean")   # sorted by metric
result.completed_steps            # e.g. [1, 2, 3, 4]
result.best_model_name            # winning model
result.best_stage                 # "default" | "reduced" | "optimized"
result.best_feature_stage         # "default" | "reduced" — feature store used by build_final_model()
result.y_transform                # winning y-transform (regression only)
result.cv_type                    # "spatial" | "random" — set from store.coords
result.cv_warnings                # list of warning strings for models with failed folds
result.step1_agg_df               # per-model mean/std metrics from step 1
result.features_reduced           # PipelineData after feature selection
result.selector.importance_summary()  # SHAP importances as a DataFrame
result.oof_predictions            # assembled OOF predictions (None if step 1 only)
result.oof_labels                 # true labels paired with oof_predictions
result.best_cv_result             # CVResult for the final winning model
```

> **Note:** `result.splitter` is not persisted. It will be `None` after `PipelineResult.load()`.

### Exporting the final model

```python
from h2ml.pipeline.final_model import FinalModel

final = result.build_final_model()   # fits on full training set
final.predict(X_new)
final.predict_proba(X_new)           # classification only

final.save("models/final.pkl")
final = FinalModel.load("models/final.pkl")
```

`FinalModel.predict()` accepts a DataFrame (columns aligned by name) or a numpy array (must match `feature_names` order).

### Conformal prediction intervals

`build_final_model()` automatically calibrates a conformal predictor from the out-of-fold CV predictions — no held-out data required.

```python
final = result.build_final_model()

# Regression — 90% prediction interval for each sample
lower, upper = final.predict_interval(X_new, alpha=0.10)

# Classification — prediction set for each sample
sets = final.predict_set(X_new, alpha=0.10)
# sets[i] == [1]    → confident prediction of class 1
# sets[i] == [0]    → confident prediction of class 0
# sets[i] == [0, 1] → uncertain; true label could be either
```

Both methods work on any input — held-out test samples, a prediction grid, spatial rasters, etc. The `alpha` parameter controls the miscoverage level: `alpha=0.10` targets ≥ 90% coverage.

**How it works:** nonconformity scores (`|y − ŷ|` for regression, `1 − p(true class)` for classification) are computed from the OOF folds and a single threshold `q` is stored. At inference time the interval is `ŷ ± q` (regression) or the set of classes with score ≤ `q` (classification).

**Limitations:**

- Intervals are **constant-width** — the same `q` is added to every prediction, so regions of the input space with higher inherent variance get the same interval as low-variance regions.
- Coverage is **marginal**, not conditional: the guarantee holds on average over new draws from the training distribution. Predictions on out-of-distribution inputs (e.g. spatial extrapolation beyond the training extent) may not achieve nominal coverage.
- If `result.y_transform` is set, the interval is in the **transformed space**. Apply `INVERSE_TRANSFORMS[result.y_transform]` to the bounds if you need original-scale intervals.

### Delta model (presence/abundance)

`DeltaFinalModel` combines a presence/absence classifier and a count/abundance regressor into a single model:

```
ŷ = P(present) × E(count | present)
```

The regressor's y-transform is inverted automatically inside `predict()`, so the output is always in the original count scale.

```python
from h2ml.pipeline.final_model import build_delta_final_model, DeltaFinalModel

# clf_result: PipelineResult from a binary classifier trained on all N samples
# reg_result: PipelineResult from a regressor trained on positive-only samples
positive_idx = np.where(y_all > 0)[0]
X_df = pd.DataFrame(X_all, columns=feature_names)

delta = build_delta_final_model(clf_result, reg_result, X_df, y_all, positive_idx)
delta.predict(X_new)                                 # delta predictions
lower, upper = delta.predict_interval(X_new, alpha=0.10)  # conformal interval

delta.save("models/sparrow_delta-model")
delta = DeltaFinalModel.load("models/sparrow_delta-model")
```

Pass a DataFrame when the classifier and regressor use different feature sets — each sub-model selects its own columns by name. Pass a numpy array only when both share the same feature order.

The conformal interval is calibrated on the combined OOF delta output (not each component separately) and the lower bound is clipped at zero.

## Persistence

```python
from h2ml import PipelineResult

result.save("runs/experiment_01")
result = PipelineResult.load("runs/experiment_01")
```

DataFrames are serialised as Parquet, numpy arrays as `.npy`, and Python objects (selector, CV results) as joblib pickles under a single directory.

## Comparing runs

```python
from h2ml.evaluation.compare import compare_results

r1 = pipeline_a.run(store)
r2 = pipeline_b.run(store)

df = compare_results([r1, r2], labels=["baseline", "spatial_cv"], metric="AUC")
```

Returns a DataFrame with one row per result: `Run`, `Metric`, `Best_Model`, `Best_Stage`, `Y_Transform`, `Score_Mean`, `Score_Std`, `Conservative_Bound` (variance-penalised score), `Brier_Mean`, `OOF_Brier`, `N_Features`, `Completed_Steps`.

Pass `n_folds` to override automatic fold-count inference — useful when comparing results loaded from disk whose fold DataFrames may be absent.

## Visualization

```python
from h2ml.plots.plots import (
    pipeline_scores,    # model scores across all three pipeline stages
    cv_diagnostics,     # classification or regression diagnostic panel
    shap_importance,    # horizontal bar chart of SHAP feature importances
    shap_summary_plot,  # SHAP beeswarm for the final best model
    shap_dependence,    # scatter + lowess for top-N features
)

pipeline_scores(result, save_path="plots/scores.png")
shap_importance(result.selector, save_path="plots/shap.png")
```

All functions accept an optional `save_path`; omit it to call `plt.show()` instead.

## Spatial inference (h2mare integration)

`h2ml.geo.geo_predict` provides functions for spatial-temporal prediction on gridded data via the `h2mare` package, which is a core dependency — no extra is required.

```python
from h2ml.geo.geo_predict import predict_map

predict_map(
    model=final,
    indexer=indexer,         # h2mare.ParquetIndexer
    dates=("2020-01", "2020-12"),
    bbox=(lon_min, lat_min, lon_max, lat_max),
    target_col="pm25",
    agg_by="month",
    save_path="maps/pm25_2020.png",
)
```

## RunMetadata

Attach experiment labels to results for multi-run comparison:

```python
from h2ml.evaluation.metrics import RunMetadata

pipeline = H2MLPipeline(
    config=config,
    metadata=RunMetadata(schema="v2_features", target="pm25", batch="2024-01"),
)
```

Labels appear as columns in all fold and agg DataFrames, making it easy to concatenate results across runs.

## Contributing

Contributions are welcome. To set up a development environment:

```bash
git clone https://github.com/h2ugoparra/h2ml
cd h2ml
uv sync --group dev
uv run pytest
```

Please submit issues or pull requests on [GitHub](https://github.com/h2ugoparra/h2ml).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed under the framework of [COSTA project](https://costaproject.org/en/) and [Marine Beacon](https://marinebeacon.eu/).
