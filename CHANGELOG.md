# Changelog

All notable changes to this project will be documented in this file.

## [0.3.1] - 2026-08-06

### Removed

- The `[geo]` optional extra. It listed `h2mare`, `cartopy` and `polars`, all of
  which the base install already provides — `h2mare` and `polars` are core
  dependencies, and `h2mare` declares `cartopy>=0.23.0` itself — so the extra
  installed nothing. `h2ml.geo.geo_predict` has always worked with the base
  install. Drop the `[geo]` suffix from any pin; installers warn about an unknown
  extra and fall back to the base package, so nothing breaks in the meantime.

### Changed

- The `ImportError` messages in `h2ml/geo/geo_predict.py` no longer tell users to
  install a `[geo]` extra. Since `h2mare` is a core dependency, a failure there
  means a broken install, and the messages now say so.

## [0.2.0] - 2026-06-05

### Added

- `LocalConformalCalibration.summary()` — per-block (and per-time-bin) calibration
  diagnostics as a DataFrame, with a `used` column showing the fallback level each
  cell resolves to (`compound`/`block`/`global`)
- `SpatialBlockSplitter.plot()` / `SPCVSplitter.plot()` — two-panel block/fold
  scatter; `plot_spatial_blocks` is now exported from `h2ml.plots`

### Changed

- SHAP dependence plots now use LOWESS with a bootstrap confidence band

## [0.1.1]

- Promote `h2mare` and `polars` to core dependencies; fix geo-prediction bugs

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
