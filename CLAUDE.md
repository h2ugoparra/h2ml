# Project: h2ml

## Overview

A 4-step AutoML pipeline wrapping sklearn-compatible estimators.

## Tech Stack

Python 3.11+. Key libraries: `scikit-learn`, `optuna` (HPO), `shap` (feature selection), `joblib` (parallel CV), `lightgbm`/`xgboost`/`catboost` (`[boosting]` extra), `h2mare` (core dep; `polars` is also core; `[geo]` extra adds `cartopy` for `predict_map`). Dev: `uv`, `ruff`, `pytest`, `tox`, `mkdocs`.

## Commands

```bash
uv run pytest                    # all tests
uv run pytest tests/feature/     # module tests
uv run pytest path::Class::test  # single test
uv run tox                       # across Python versions
uv run ruff check h2ml/          # lint
uv run ruff format h2ml/         # format
uv run mkdocs serve              # docs preview
uv run mkdocs build              # docs build
```

## Architecture

```text
H2MLPipeline (pipeline/pipeline.py)
  ├── Step 1: CV all models × transforms  →  best_model_name, step1_agg_df
  ├── Step 2: OOF SHAP + correlation filter  →  features_reduced, selector
  ├── Step 3: CV on reduced features  →  best_stage ("default" or "reduced")
  └── Step 4: Optuna HPO  →  best_params  [skipped if opt_enabled=False]

CrossValidator (pipeline/cv.py)              → joblib.Parallel over models; scaling inside folds
FeatureSelector (features/selector.py)       → shap_importance.py + correlation.py
SpatialBlockSplitter / SPCVSplitter          → activated when store.coords is set; built once in step 1
  (features/spatial_cv.py)
FinalModel (pipeline/final_model.py)         → predict/predict_proba + optional ConformalCalibration
                                               or LocalConformalCalibration (space-time block-local)
DeltaFinalModel (pipeline/final_model.py)    → P(present) × E(count|present); built via build_delta_final_model()
ModelRegistry (utils/registry.py)            → single source of truth; build_models(task)
Optimizer (optimization/optimizer.py)        → run_study(); optimize_all() for batch use
```

Partial runs:

```python
result = pipeline.run_step1_only(store)     # quick model screening
result = pipeline.run_step1_to_step2(store) # steps 1–2
result = pipeline.run_step1_to_step3(store) # steps 1–3, no HPO
result = pipeline.run_from_step3(result)    # resume (needs features, features_reduced, best_model_name, selector)
result = pipeline.run_step4_only(result)    # re-run HPO only (needs above + best_stage, best_model_value)
```

## PipelineConfig

| Parameter | Default | Effect |
|-----------|---------|--------|
| `metric` | `"AUC"` | Model selection and HPO metric. Classification: `"AUC"`, `"AUC_PR"`, `"F1"`, `"LogLoss"`, `"Brier"`. Regression: `"R2"`, `"MAE"`, `"RMSE"`. |
| `n_splits` | `5` | CV folds for model screening (steps 1 and 3) |
| `corr_threshold` | `0.7` | Drop feature if it exceeds this in any of Pearson, Spearman, or Kendall with a retained feature |
| `min_features` | `1` | Minimum features retained after correlation filter |
| `n_trials` | `50` | Optuna trials in step 4 |
| `opt_n_splits` | `3` | Folds used during Optuna |
| `n_hpo_repeats` | `1` | Independent HPO repeats; trials divided evenly, best repeat kept |
| `handle_imbalance` | `False` | Inject `class_weight="balanced"` for classifiers with `supports_class_weight=True` |
| `spatial_cv_method` | `"spcv"` | Spatial CV strategy: `"block"` or `"spcv"` (ignored when `store.coords` is `None`) |
| `spatial_cv_metric` | `"euclidean"` | Distance metric: `"euclidean"` or `"haversine"` |
| `time_bin_resolution` | `"month"` | Temporal bin granularity for spatial CV and local conformal calibration: `"month"` or `"season"` |
| `n_blocks_per_fold` | `5` | Blocks per test fold for the block splitter |
| `ahc_threshold` | `None` | AHC distance threshold for SPCVSplitter (auto-derived when `None`) |
| `pca_components` | `0.95` | Variance retained by PCA on block covariates |
| `exact_max_samples` | `5_000` | Sample threshold for exact vs. approximate AHC |
| `knn_neighbors` | `15` | k for k-NN connectivity graph in approximate AHC |

## PipelineResult

Key fields: `best_model_name`, `best_stage`, `best_feature_stage`, `best_params`, `y_transform`, `metric`, `cv_type`, `cv_warnings`, `splitter`, `step3_reduced_stores`. Note: `splitter` and `step3_reduced_stores` are not persisted — both are `None` after `PipelineResult.load()`.

```python
result.summary(metric=None, ascending=False)   # combined agg DataFrame across all stages
result.build_final_model()                      # refit on full training set → FinalModel
result.save(path) / PipelineResult.load(path)  # serialise/reload

# FinalModel conformal prediction
model.predict_interval(X, alpha)  # regression; alpha=0.10 → 90% coverage
model.predict_set(X, alpha)       # classification prediction sets

# Multi-run comparison
compare_results(results, labels, metric, n_folds)  # sort direction auto-derived; no ascending arg
```

## Conventions

- **CatBoost** uses `random_seed` (not `random_state`) and `thread_count` (not `n_jobs`)
- **SHAP routing**: tree models → `TreeExplainer`; linear SVM (`kernel="linear"`, `probability=False`) → `LinearExplainer`; all others → KernelSHAP with `shap.kmeans` background (capped at `max_background=100`)
- **y-transform names**: `"count"` (identity), `"log"`, `"sqrt"`, `"wincount"`, `"winlog"`, `"winsqrt"`; winsorize variants silently skipped when no outliers
- **`n_jobs=1`** in optimizer always — parallel Optuna trials cause resource-tracker warnings on Windows
- **Boosting extras** require `uv sync --extra boosting`

## h2mare dependency

`h2mare` (PyPI, core dependency) — geospatial storage (`ParquetIndexer`), aggregation, map plotting. Import paths in `h2ml/geo/geo_predict.py` use `h2mare.*`. The `[geo]` extra adds `cartopy`, required only for `predict_map` (`polars` is a core dependency).
