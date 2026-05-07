# H2ML

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)

A 4-step AutoML pipeline for tabular data that wraps sklearn-compatible estimators. Given a feature matrix and target, it screens all registered models, reduces features via SHAP importance and correlation filtering, and tunes the winner with Optuna — all in one call.

## How it works

| Step | What happens |
|------|-------------|
| 1 | K-fold CV all models (× optional y-transforms) on all features |
| 2 | SHAP importance → correlation-based feature drop |
| 3 | K-fold CV all models on reduced features; compare vs step 1 |
| 4 | Optuna HPO on the winning (model, stage, transform) |

## Features

- **Model screening** — all sklearn classifiers and regressors in one parallel CV pass
- **Optional boosting** — LightGBM, XGBoost, CatBoost registered when installed
- **SHAP feature selection** — out-of-fold SHAP avoids leaking importance scores
- **Spatial CV** — block or AHC-based splitters for geospatial data
- **y-transform sweep** — log, sqrt, and winsorized variants evaluated jointly
- **Conformal prediction** — guaranteed-coverage intervals built from OOF residuals
- **Persistence** — Parquet + npy + joblib; reload and resume any pipeline stage

## Quick links

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Pipeline Overview](pipeline.md)
- [API Reference](api/pipeline.md)
