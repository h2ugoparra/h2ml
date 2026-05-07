# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-05-07

Initial public release.

### Features

- 4-step AutoML pipeline: model screening (step 1), SHAP feature selection (step 2), reduced-feature CV (step 3), Optuna HPO (step 4)
- Partial-run API: `run_step1_only`, `run_step1_to_step2`, `run_step1_to_step3`, `run_from_step3`, `run_step4_only`
- Spatial cross-validation: `SpatialBlockSplitter` (quantile-grid) and `SPCVSplitter` (AHC cluster ensemble)
- y-transform sweep for regression: `log`, `sqrt`, `count`, `winlog`, `winsqrt`, `wincount`
- Conformal prediction intervals (`FinalModel.predict_interval`) and prediction sets (`FinalModel.predict_set`)
- Optional boosting models via `[boosting]` extra: LightGBM, XGBoost, CatBoost
- Optional spatial inference via `[geo]` extra: `h2ml.geo.geo_predict` (requires `h2mare`)
- Result persistence (`PipelineResult.save` / `load`) and multi-run comparison (`compare_results`)
- Visualization: `pipeline_scores`, `cv_diagnostics`, `shap_importance`, `shap_summary_plot`, `shap_dependence`
- Supports Python 3.11, 3.12, 3.13
