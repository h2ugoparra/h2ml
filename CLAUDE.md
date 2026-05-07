# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/pipeline/test_pipeline.py

# Run a single test by name
uv run pytest tests/pipeline/test_pipeline.py::TestRunStep1Only::test_step1_fold_df_populated

# Run tests for a specific module
uv run pytest tests/feature/

# Run tests across Python versions
uv run tox

# Lint
uv run ruff check h2ml/

# Docs — preview locally
uv run mkdocs serve

# Docs — build static site
uv run mkdocs build
```

## Architecture

h2ml is a 4-step AutoML pipeline that wraps sklearn-compatible estimators. The full workflow is orchestrated by `H2MLPipeline` in `h2ml/pipeline/pipeline.py`.

### The 4-step pipeline

| Step | What happens | Key output |
|------|-------------|------------|
| 1 | K-fold CV all models (× optional y-transforms) on all features | `best_model_name`, `step1_agg_df` |
| 2 | Fit best model → OOF SHAP importance → correlation-based feature drop | `features_reduced`, `selector` |
| 3 | K-fold CV all models on reduced features (winning transform only); compare vs step 1 | `best_stage` (`"default"` or `"reduced"`) |
| 4 | Optuna HPO on the winning (model, stage, transform) | `best_params`, `step4_agg_df` |

Step 4 is skipped gracefully when the winning model has `opt_enabled=False` in the registry. If the optimised result does not improve on step 3 baseline, `best_params` is set to defaults (no HPO benefit found).

### Partial runs

```python
result = pipeline.run_step1_only(store)           # quick model screening
result = pipeline.run_step1_to_step2(store)       # steps 1–2, inspect importance
result = pipeline.run_step1_to_step3(store)       # steps 1–3, no HPO
result = pipeline.run_from_step3(result)          # resume from a pre-reduced result
result = pipeline.run_step4_only(result)          # re-run HPO on an existing result
```

`run_from_step3` requires `result.features`, `result.features_reduced`, `result.best_model_name`, and `result.selector` to be populated. `run_step4_only` requires steps 1–3 to be complete and is useful for loading a saved result and re-optimising without repeating feature selection.

### Key `PipelineConfig` parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `metric` | `"AUC"` | Short metric name for model selection and HPO. Classification: `"AUC"`, `"AUC_PR"`, `"F1"`, `"LogLoss"`, `"Brier"`. Regression: `"R2"`, `"MAE"`, `"RMSE"`. Minimisation direction is derived automatically. |
| `corr_threshold` | `0.7` | Threshold for dropping correlated features in step 2. A feature is dropped if it exceeds this in any of Pearson, Spearman, or Kendall. |
| `min_features` | `1` | Minimum features retained after correlation filter |
| `n_trials` | `50` | Optuna trials in step 4 |
| `opt_n_splits` | `3` | Folds used during Optuna (faster than `n_splits`) |
| `n_hpo_repeats` | `1` | Independent HPO repeats with different fold seeds; best repeat is kept |
| `handle_imbalance` | `False` | Inject `class_weight="balanced"` for classifiers with `supports_class_weight=True` |
| `spatial_cv_method` | `"block"` | Spatial CV strategy: `"block"` or `"spcv"` (ignored when `store.coords` is `None`) |
| `spatial_cv_metric` | `"euclidean"` | Distance metric for spatial CV: `"euclidean"` or `"haversine"` |
| `n_blocks_per_fold` | `5` | Blocks per test fold for the block splitter |
| `ahc_threshold` | `None` | AHC distance threshold for SPCVSplitter (auto-derived when `None`) |
| `pca_components` | `0.95` | Variance retained by PCA on block covariates |
| `exact_max_samples` | `5_000` | Sample threshold for exact vs. approximate AHC |
| `knn_neighbors` | `15` | k for k-NN connectivity graph in approximate AHC |

### `PipelineResult` key fields and methods

Key fields set after a completed run:
- `best_model_name`, `best_stage` (`"default"` or `"reduced"`), `best_feature_stage`, `best_params`
- `y_transform` — winning transform name (regression only)
- `cv_type` — `"spatial"` or `"random"` (set from whether `store.coords` is provided)
- `cv_warnings` — list of warning strings for models with failed CV folds
- `splitter` — CV splitter built once in step 1 and reused across steps
- `step3_reduced_stores` — cached reduced `PipelineData`s keyed by transform name or `""`
- `metric` — short metric name (e.g. `"AUC"`, `"R2"`) copied from config

Key methods:
- `result.summary(metric=None, ascending=False)` — combined agg DataFrame across all completed stages
- `result.build_final_model()` — refit best model on full training set → `FinalModel`
- `result.save(path)` / `PipelineResult.load(path)` — serialise/reload

### Result comparison (`h2ml/evaluation/compare.py`)

`compare_results(results, labels, metric, n_folds)` compares multiple `PipelineResult` objects side-by-side and returns a DataFrame with columns: Run, Metric, Best_Model, Best_Stage, Y_Transform, Score_Mean, Score_Std, Conservative_Bound, Brier_Mean, OOF_Brier, N_Features, Completed_Steps. Sort direction is derived automatically from the metric — there is no `ascending` argument. Useful for selecting the best configuration across runs with different settings.

### Core data container: `PipelineData` (`h2ml/features/feature_store.py`)

A lightweight dataclass that carries `(X: np.ndarray, feature_names, y, y_true, y_transform, coords)` together. Everything inside the pipeline uses this instead of DataFrames. It converts to DataFrame only locally (SHAP, correlation). The `select(features)` method is how features are subset after step 2. `coords` is an optional array of spatial coordinates; when provided it activates spatial CV throughout the pipeline.

### Data validation

`H2MLPipeline._validate_store()` runs before every pipeline entry and checks for: NaN/Inf in X, y, and coords; constant (zero-variance) columns; insufficient classes in classification; class imbalance ratio (warns if minority class < 10%). An exception is raised for hard errors; warnings are appended to `result.cv_warnings`.

### Model registry (`h2ml/utils/registry.py`)

Single source of truth for every model. `ModelEntry` holds:
- `model_cls`, `default_kwargs` — how to instantiate
- `requires_scaling` — whether `CrossValidator` should apply `StandardScaler`
- `param_fn` — Optuna trial function returning a hyperparameter dict
- `opt_enabled` — set `False` to skip step 4 for this model (e.g. LogisticRegression, GaussianNB, KNeighborsClassifier, AdaBoostClassifier, BaggingClassifier and their regressor equivalents)
- `supports_class_weight` — set `True` to receive `class_weight="balanced"` when `handle_imbalance=True` (classifiers only; e.g. LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier, SVC, ExtraTreesClassifier, LGBMClassifier)
- `single_njob` — legacy flag, currently unused (optimizer always runs trials sequentially with `n_jobs=1`)

`build_models(task)` constructs the default `list[ModelWrapper]` for a given task type. LightGBM, XGBoost and CatBoost are imported with `try/except` and only registered if available. They require the `[boosting]` optional extra (`uv sync --extra boosting`).

> **CatBoost note:** CatBoost uses `random_seed` (not `random_state`) and `thread_count` (not `n_jobs`) in its kwargs. Its registry entries and param functions reflect this.

### Optimization engine (`h2ml/optimization/optimizer.py`)

`run_study(name, X, y, task, metric, n_trials, n_splits)` runs an Optuna TPE study for a single registered model. Step 4 in the pipeline calls this directly; `optimize_all()` is a batch wrapper used outside the pipeline.

When `n_hpo_repeats > 1`, the pipeline runs that many independent studies with different fold seeds and keeps the repeat with the highest `best_value`. Trials are divided evenly: `trials_per_repeat = max(1, n_trials // n_hpo_repeats)`.

**Metric selection** — all metrics are maximised internally (error metrics are negated). Available metrics are exported as module-level dicts:

| Dict         | Keys                                          |
|--------------|-----------------------------------------------|
| `CLF_METRICS` | `"AUC"`, `"AUC_PR"`, `"LogLoss"`, `"F1"`, `"Brier"` |
| `REG_METRICS` | `"R2"`, `"MAE"`, `"RMSE"`                    |

The pipeline derives the optimizer metric automatically from `config.metric` by stripping the `_Test_Mean` suffix (e.g. `"R2_Test_Mean"` → `"R2"`), so the same metric is used for both model selection and HPO. Defaults: `"AUC"` (classification), `"R2"` (regression).

**Efficiency design:**

- Fold indices and (for `requires_scaling=True`) scaled fold arrays are pre-computed once per study — they are identical across all trials since X and y are fixed.
- `MedianPruner(n_startup_trials=10, n_warmup_steps=1)` terminates trials whose running fold-mean falls below the median at the same fold step.
- `n_jobs=1` always — parallel Optuna trials via loky cause resource-tracker warnings on Windows.

### Cross-validation engine (`h2ml/pipeline/cv.py`)

`CrossValidator.run_all()` parallelises model CV runs using `joblib.Parallel(backend='loky')` (process-based, bypasses the GIL). Scaling is always applied inside folds to prevent leakage. Results are collected as `FoldResult` → `CVResult` structs and passed to `evaluation/metrics.py` for metric computation.

### Spatial cross-validation (`h2ml/features/spatial_cv.py`)

When `store.coords` is provided, the pipeline swaps standard k-fold for a spatial splitter built from `PipelineConfig`. The splitter is constructed once in step 1 (stored as `result.splitter`) and reused in steps 2, 3, and 4. `result.cv_type` is set to `"spatial"`.

Two splitter classes are available:

- **`SpatialBlockSplitter`** (`spatial_cv_method="block"`) — quantile-grid blocking. Divides the spatial domain into a regular grid of blocks and assigns them to folds, ensuring test blocks are geographically separated from training blocks.
- **`SPCVSplitter`** (`spatial_cv_method="spcv"`) — Agglomerative Hierarchical Clustering (AHC) + cluster ensemble method. Groups samples into spatially coherent clusters. When the dataset exceeds `exact_max_samples`, approximate AHC is used (k-NN connectivity graph, controlled by `knn_neighbors`). `pca_components` controls optional dimensionality reduction on block covariates before clustering.

Both splitters respect `spatial_cv_metric` (`"euclidean"` or `"haversine"`).

### Metrics (`h2ml/evaluation/metrics.py`)

`compute_metrics_all(cv_results, metadata)` produces a per-fold DataFrame. `aggregate_metrics()` collapses it to mean/std per model. `select_best()` picks the winner. Overfitting gap columns are computed for all metrics (Gap = Train − Test for maximise metrics, Test − Train for minimise metrics).

`RunMetadata` carries experiment-level labels (stage, schema, target, batch, y_transform, notes) that end up as columns in the fold DataFrame. The `notes` field accepts free-text for experiment annotations.

Classification metrics include: AUC, AUC_PR, LogLoss, F1, Brier. Regression metrics include: R2, MAE, RMSE.

### y-transform sweep (regression)

Pass `transforms=["log", "sqrt", "count"]` to `pipeline.run()`. Internally this builds one `PipelineData` per transform via `build_transform_stores()` (`h2ml/preprocessing/transform_stores.py`). Steps 1 and 3 sweep all (model × transform) combinations. The winning transform is stored in `result.y_transform`; inverse transforms are looked up from `INVERSE_TRANSFORMS` in `h2ml/preprocessing/transforms.py`.

Available transform names (from `Y_TRANSFORMS` in `transforms.py`): `"count"` (identity), `"log"`, `"sqrt"`, `"wincount"`, `"winlog"`, `"winsqrt"`. Winsorize-based transforms return `None` when no outliers are present and are silently skipped by `build_transform_stores`.

### Feature selection (`h2ml/features/selector.py`)

`FeatureSelector` is fitted once in step 2 on the winning transform's store. In step 3 it applies `selector.transform()` to the winning transform's store to build the reduced variant. The reduced store is cached on `result.step3_reduced_stores` so step 4 can reuse it without recomputing.

**SHAP explainer routing** — the explainer is chosen per model:
- Tree-based models: `TreeExplainer` (fast, exact)
- Linear SVMs (`kernel="linear"`, `probability=False`): `LinearExplainer` (exact)
- All other models: KernelSHAP with background summarisation via `shap.kmeans`

SHAP values are computed out-of-fold by default (`use_oof=True`): each sample's importance is derived from a fold where it was held out, so no training sample leaks into its own SHAP computation. The spatial splitter (`result.splitter`) is forwarded to the OOF SHAP pass when spatial CV is active, ensuring consistent spatial fold boundaries throughout the pipeline.

The `max_background` parameter (default `100`) on `FeatureSelector` caps KernelSHAP compute cost by limiting the background dataset size.

**Correlation filtering** — features are dropped if they exceed `corr_threshold` in any of Pearson, Spearman, or Kendall correlation with a retained feature. `min_features` prevents the filter from dropping below that count.

### Deployment artifact (`h2ml/pipeline/final_model.py`)

`build_final_model(result)` re-fits the best model on the full training set (with best params and correct feature stage) and returns a `FinalModel` that handles predict/predict_proba, optional scaling, and persistence via joblib.

`FinalModel` fields: `conformal` (optional `ConformalCalibration`), `y_transform` (for inverse transformation at inference time).

**Conformal prediction** — `ConformalCalibration` builds from out-of-fold CV residuals/nonconformity scores and exposes:
- `FinalModel.predict_interval(X, alpha)` — finite-sample coverage-guaranteed prediction intervals (regression); `alpha` is the miscoverage level (e.g. `alpha=0.10` → 90% coverage)
- `FinalModel.predict_set(X, alpha)` — prediction sets (classification)

### Plots (`h2ml/plots/plots.py`)

Top-level functions for post-run visualisation: `model_scores`, `pipeline_scores` (all stages), `cv_diagnostics` (classification or regression panel), `shap_importance`, `shap_summary_plot`, `shap_dependence`. All accept an optional `save_path`; when omitted they call `plt.show()`.

### Persistence (`h2ml/persistence/result_io.py`)

`result.save(path)` / `PipelineResult.load(path)` serialises all DataFrames as Parquet, numpy arrays as `.npy`, and Python objects (selector, CV results) as joblib pickles under a single directory.

### Geospatial prediction (`h2ml/geo/geo_predict.py`)

Optional `[geo]` extra. Top-level functions:
- `predict_for_year(models, year, ...)` — load `FinalModel`s and generate predictions for a full calendar year
- `predict_map(store, models, ...)` — predict on a spatial-temporal grid and plot aggregated maps

Integrates with `ParquetIndexer` from `h2gis` for hive-partitioned parquet stores.

### Variogram utility (`h2ml/utils/variogram.py`)

`autocorrelation_range(coords, residuals)` estimates the spatial autocorrelation range from an empirical variogram. Returns a `VariogramResult`. `plot_variogram(result)` visualises it. Useful for measuring unexplained spatial structure in model residuals to inform spatial CV block size.

## Key relationships between files

```bash
pipeline.py          → cv.py (run folds)
                     → features/selector.py (step 2)
                     → features/spatial_cv.py (build splitter when coords provided)
                     → evaluation/metrics.py (fold → agg DataFrames)
                     → optimization/optimizer.py (step 4)
                     → utils/registry.py (build models, check opt_enabled)

cv.py                → pipeline/base.py (TaskType, PredictorStep)
                     → pipeline/step.py (ModelWrapper)

features/selector.py → features/shap_importance.py (OOF SHAP values)
                     → features/correlation.py (remove correlated features)

optimization/optimizer.py → optimization/opt_params.py (get_entry, param_fn)
                          → CLF_METRICS / REG_METRICS (metric → score_fn lookup)

evaluation/compare.py → PipelineResult (multi-run comparison)

geo/geo_predict.py   → pipeline/final_model.py (FinalModel inference)
                     → h2mare.ParquetIndexer (spatial storage)

utils/variogram.py   → (standalone, no internal deps)
```

## h2mare dependency

`h2mare` (PyPI package, repo: `h2ugoparra/h2mare`) is the companion package for geospatial storage (`ParquetIndexer`), spatial aggregation, and map plotting. It is resolved from PyPI via the `[geo]` optional extra. Import paths in `h2ml/geo/geo_predict.py` use `h2mare.*`.
