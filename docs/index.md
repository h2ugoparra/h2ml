# H2ML

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![PyPI](https://img.shields.io/pypi/v/h2ml)](https://pypi.org/project/h2ml/)

A 4-step AutoML pipeline for tabular data that wraps sklearn-compatible estimators. Given a feature matrix and target, it screens all registered models, reduces features via SHAP importance and correlation filtering, and tunes the winner with Optuna — all in one call.

```python
from h2ml import H2MLPipeline, PipelineConfig, PipelineData

store = PipelineData.from_frame(X, y=y_arr)

pipeline = H2MLPipeline(config=PipelineConfig(task_type="classification", metric="AUC"))
result = pipeline.run(store)

print(result.summary())
final = result.build_final_model()   # refit on the full training set
```

## How it works

| Step | What happens |
|------|-------------|
| 1 | K-fold CV all models (× optional y-transforms) on all features |
| 2 | SHAP importance → correlation-based feature drop (optional VIF filter) |
| 3 | K-fold CV all models on reduced features (winning transform only); compare vs step 1 |
| 4 | Optuna HPO on the winning (model, stage, transform); skipped if the winner has no enabled search space |

The pipeline can also be run partially: `run_step1_only()` screens models, and `run_from_step3()` / `run_step4_only()` resume from a previous result. With `holdout_frac > 0`, rows are set aside before step 1 and the final configuration is scored once on them.

## Features

### Core pipeline

- **Model screening** — a curated set of sklearn classifiers and regressors in one parallel CV pass
- **Optional boosting** — LightGBM, XGBoost, CatBoost registered when installed
- **SHAP feature selection** — out-of-fold SHAP avoids leaking importance scores
- **Cross-validation** — stratified k-fold for classification and k-fold for regression, or [spatial block and AHC-based splitters](spatial_cv.md) when `store.coords` is set
- **y-transform sweep** — log, sqrt, and winsorized variants evaluated jointly
- **Persistence** — Parquet + npy + joblib; reload a run and resume from step 3 or 4

### Trustworthy selection

- **[Ranking robustness](pipeline.md#selection-robustness)** — paired-fold ties and per-metric ranks, with an optional [1-SE rule](pipeline.md#preferring-reduced-features-on-ties) favouring reduced features
- **[Per-fold feature selection](pipeline.md#selecting-features-inside-cv-folds)** — an unbiased default-vs-reduced comparison
- **[Outer holdout](pipeline.md#outer-holdout)** — one score no selection step has seen
- **[Nested CV](pipeline.md#nested-cv)** — how well the whole pipeline generalises
- **[Seed stability](pipeline.md#seed-stability)** — how much a run depends on its seed
- **[Metric comparison](pipeline.md#comparing-selection-metrics)** — compare selection metrics while sharing CV work

### Before and after the pipeline

- **[EDA](guides/eda.md)** — dataset profiling and pipeline readiness checks
- **[Conformal prediction](guides/conformal.md)** — distribution-free intervals and prediction sets, calibrated on OOF or holdout predictions, with optional [spatio-temporal local widths](guides/conformal.md#spatio-temporal-local-calibration)
- **[Area of applicability](guides/geo_predict.md#area-of-applicability-columns)** — flags predictions outside the feature space the model was trained on
- **[Delta model](guides/conformal.md#delta-model-presence-abundance)** — P(present) × E(count | present) for presence/abundance data
- **[Geo prediction](guides/geo_predict.md)** — mapped predictions over gridded data through `h2mare`

## Quick links

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Pipeline Overview](pipeline.md)
- Guides: [EDA](guides/eda.md) · [Y-Transforms](guides/transforms.md) · [Conformal Prediction](guides/conformal.md) · [Geo Prediction](guides/geo_predict.md) · [Comparing Runs](guides/comparison.md) · [Visualization](guides/visualization.md) · [Persistence](guides/persistence.md)
- [API Reference](api/pipeline.md)
