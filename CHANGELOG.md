# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- `multi_metric` supports regression pipelines: R2, MAE and RMSE (default order) with
  an optional `transforms=` y-transform sweep, shared exactly as for classification
  (step 1 once, steps 2–3 per step-1 winner and transform). Each metric's result still
  equals a separate `run(store, transforms=...)` under that metric. `handle_imbalance`
  is rejected for regression.
- `PipelineConfig(prefer_reduced_on_tie=True)` applies a one-standard-error rule in
  step 3: when the winner uses all features and a reduced-feature candidate is tied
  with it on paired folds (within one Nadeau–Bengio corrected SE), the best-ranked
  tied reduced candidate is selected instead. Step 4 then tunes it on the reduced
  features against its own score. Each switch is logged and recorded in
  `cv_warnings`. Default `False` — selection is unchanged unless enabled.
- Selection robustness check (`h2ml.evaluation.ranking`). After selection in steps 1
  and 3, every candidate is compared with the winner on shared folds (paired
  differences, Nadeau–Bengio corrected SE) and ranked under every metric. Ties with
  the winner, and a winner ranking below 1st on the counterpart-family metric (e.g.
  AUC vs Brier), are added to `cv_warnings`. `PipelineResult.ranking_robustness(step=None)`
  returns the full table and works on loaded results. Advisory only.
- Metric-suitability check in step 1. Warns, using the actual CV folds, when the data
  makes the configured metric unreliable: AUC under class imbalance, single-class
  test folds for AUC/AUCPR, uneven fold prevalence for AUCPR, F1 with class weights,
  low-variance folds for R2, heavy tails for RMSE, zero inflation for MAE. Warnings
  go to `cv_warnings`; the metric is never changed.
- `PipelineConfig(handle_imbalance="auto")` injects `class_weight="balanced"` only
  when the input target's minority-class ratio is below 0.1 (the same threshold as
  the imbalance warning). It is decided when each run starts, including on resume
  via `run_from_step3` / `run_step4_only`. `True` and `False` behave as before.

### Fixed

- `predict_map` failed on a fresh install with an `ImportError`: `h2mare` 0.9.0 moved
  `cartopy` to its `viz` extra, and the lock pinned 0.1.1, so nothing caught it. h2ml
  now depends on `h2mare[viz]>=0.9.0`.
- `PipelineResult.best_cv_result` (and so `oof_predictions`, `oof_labels`, the variogram,
  conformal calibration and `cv_diagnostics`) now follows the selected configuration.
  Before, it returned the step-4 CV whenever tuning ran, even if the tuned model was
  rejected, and otherwise the step-3 CV matched by model name only, even when the
  default feature stage won. `CVResult.y_transform` records the transform so step-1
  results can be told apart. Results saved earlier load with it as `None` and match by
  model name.
- `winsorize` returns `None` when its limit sits at the minimum (e.g. ≥ 75 % zeros), so
  the transform is skipped and a training fold keeps its targets. Before, every target
  was clipped to one constant: models predicted exactly 0, and CatBoost raised
  `All train targets are equal`.
- `build_final_model()` refits on the log/sqrt scale when a win* transform has nothing
  to clip on the refit rows. Before, it fitted raw y and then inverted predictions.
- Candidates with failed folds no longer take part in selection when another candidate
  completed every fold. Means over fewer folds favoured models that failed on the hard
  folds. A tuned model with a failed fold is not accepted in step 4.
- `get_oof_shap_values` skips a fold that fails, with a warning, and computes importance
  over the explained rows. It raises only when every fold fails. Before, one failed
  fold lost step 2, and with it the whole metric.
- `multi_metric(reuse_step4="model")` only shares step-4 tuning between metrics whose
  final feature lists match. Before, a matching (model, transform, feature stage) was
  enough, so a later metric with a different "reduced" feature set reused tuned params
  — and their acceptance CV — computed on another feature set.
- `build_final_model()` now keeps `class_weight="balanced"` when `handle_imbalance`
  was active but step 4 did not run (e.g. after `run_step1_to_step3`). Previously
  the final model was refit from registry defaults without the weights CV used.
  The decision is stored as `PipelineResult.class_weight_balanced` and persisted;
  results saved before this change load with it set to `False`.
- `handle_imbalance` now finds supporting models by estimator class rather than
  display name. A model given a custom name (e.g. `make_classifier(..., name="LR")`)
  previously ran CV without `class_weight`, while step 4 and the final model added it.
- `build_final_model()` now refits the estimator cross-validation actually ran,
  instead of rebuilding it from the model registry. Previously a model with a
  custom name (`make_classifier(..., name="LR")`) failed with "not found in
  registry", and constructor arguments (e.g. `LogisticRegression(max_iter=500)`)
  and the wrapper's `requires_scaling` were replaced by registry defaults. Step-4
  Optuna trials and the step-4 CV had the same problem and are fixed too. The
  unfitted winner is stored and persisted as `PipelineResult.best_base_estimator`.
  Results saved before this change keep the old registry-based rebuild.
- `predict_for_year` and `predict_for_year_delta` with `local=True` passed coordinates
  as (lon, lat) to local conformal, which is calibrated on `store.coords` as (lat, lon),
  so intervals used the wrong spatial block's width. Predictions were unaffected.
  Rerun any maps made this way. `store.coords` must be (lat, lon) for `local=True`: a
  model trained on (lon, lat) got correct widths before and needs retraining now.
- A winning estimator that is not in the registry no longer crashes step 4 with
  `KeyError`; it is treated like `opt_enabled=False` and HPO is skipped.
- CatBoostClassifier HPO no longer fails every trial with "Only one of parameters
  ['verbose', 'logging_level', 'verbose_eval', 'silent'] should be set". Its search
  space set `verbose=0` on top of the registry's `silent=True`.

### Changed

- **Breaking:** `FinalModel.predict()` now returns the original y scale when a
  y-transform won, matching `predict_interval()` and `DeltaFinalModel.predict()`.
  Before, it returned the transformed space, so scoring it against `y` was wrong.
  Remove any manual `INVERSE_TRANSFORMS` step after `predict()`, which would now
  invert twice; models saved earlier behave the same way once loaded. `predict_map`
  now maps original-scale values too: it never inverted, so transformed models drew
  log- or sqrt-scale maps. Use `final.estimator.predict()` for the transformed space.
- Steps 1 and 3 rank candidates by a confidence bound built from the fold-centred std
  (each fold's scores minus that fold's median across candidates) instead of the raw
  fold std. The same score orders `ranking_robustness` and therefore the candidate
  `prefer_reduced_on_tie` switches to. The raw std counted fold difficulty that every
  model shares as instability. Under uneven spatial folds, e.g. prevalence from 1 % to
  53 %, it favoured models that hedge every prediction towards the middle: they score
  uniformly mediocre, while a better model is penalised for its one bad fold. This
  changes selections. On 12 presence/absence targets it changed 34 of 60 step-3 picks,
  and 22 of the 29 scorable changes improved the pooled out-of-fold score.
  `select_best(..., fold_df=None)` and `selection_score` keep the raw-std bound.
- `multi_metric`'s default metric order is now explicit: Brier, LogLoss, AUC, AUCPR,
  F1 for classification (was AUC, AUCPR, LogLoss, F1, Brier) and RMSE, R2, MAE for
  regression (was R2, MAE, RMSE). It only changes results when `metrics=` is omitted
  and `reuse_step4="model"`: shared step-4 tuning is now optimised for the proper
  scoring rule rather than AUC/R2. Otherwise only the row order changes.
- The SVC and SVR search spaces are narrowed so HPO trials no longer stall for
  minutes. On noisy data, a linear kernel with large C made libsvm run millions of
  iterations per fit (over 80s for one fit at ~1k rows). Kernels are now `rbf` and
  `linear` (`poly` and `sigmoid` removed, along with SVR's `degree`). `C` ranges
  over 0.01–10 instead of 0.1–100, `max_iter=1_000_000` caps any remaining slow
  fits, and `gamma` for `rbf` is tuned log-uniformly over 1e-4–1. It replaces the
  `"scale"`/`"auto"` choice, which gives nearly the same value once features are
  scaled. Tuned SVC/SVR results will differ from earlier runs.

- `PipelineResult.best_params` now holds only the overrides applied on top of the
  winning estimator (tuned params plus fixed ones like `class_weight`), not the
  registry's full default kwargs. When step 4 is skipped or does not improve it is
  `{}` (or just the fixed overrides).
- `run_study()` accepts optional `estimator` and `requires_scaling` arguments. With
  `estimator`, each trial is `clone(estimator).set_params(**sampled)`.
- The precision–recall metric is renamed from `"AUC_PR"` to `"AUCPR"`, in line
  with `LogLoss` and XGBoost's `aucpr`. Its result columns follow suit
  (`AUC_PR_Test_Mean` → `AUCPR_Test_Mean`, and likewise for `_Train` and `_Gap`).
  Code that reads the old column names directly must be updated.

### Deprecated

- `metric="AUC_PR"` in `PipelineConfig` and `run_study`. It still works, maps to
  `"AUCPR"` and logs a warning. Results saved under the old name load with the
  metric and columns renamed.

### Documentation

- `<metric>_Test_Mean` columns are now documented as the mean of the per-fold
  held-out scores, as distinct from a single score on the pooled OOF predictions.

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
