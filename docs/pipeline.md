# Pipeline

## The 4 steps

| Step | What happens | Key outputs |
|------|-------------|-------------|
| 1 | K-fold CV all models (× y-transforms) on all features | `best_model_name`, `step1_agg_df` |
| 2 | Fit best model → OOF SHAP → correlation filter | `features_reduced`, `selector` |
| 3 | K-fold CV all models on reduced features; compare vs step 1 | `best_stage`, `step3_agg_df` |
| 4 | Optuna HPO on winning (model, stage, transform) | `best_params`, `step4_agg_df` |

Regression candidates are ranked by a **pooled** score: R2, MAE, RMSE or Poisson deviance computed once over all out-of-fold rows, in steps 1 and 3, in Optuna trials and in step 4's acceptance check. Each row counts equally, so a small or near-constant fold can't dominate, as it can in a mean of fold scores. `selection_scoring="fold_mean"` uses the classification rule below instead. Candidates with no prediction spread in any fold are excluded from selection unless every candidate is flat; both cases add a `[selection]` warning. When every candidate is flat, step 2 keeps all features, because all-zero SHAP values can't rank them.

Classification candidates are ranked by a confidence-bound score (mean − std/√k for maximised metrics, mean + std/√k for minimised ones). The std is **fold-centred**: each fold's scores minus that fold's median across candidates. Fold difficulty that every model shares, such as Brier's floor moving with each spatial fold's prevalence, therefore does not count as instability. With the raw fold std, a model that hedges every prediction towards the middle scores uniformly across uneven folds and would beat a model that is better on average but has one bad fold. With `prefer_reduced_on_tie=True`, step 3 then applies a 1-SE rule: if the winner uses all features but a reduced-feature candidate is statistically tied with it on paired folds, the reduced candidate is selected instead — see [Preferring reduced features on ties](configuration.md#preferring-reduced-features-on-ties).

Step 4 is skipped when the winning model has no enabled search space: `opt_enabled=False` (LogisticRegression, GaussianNB, KNeighborsClassifier, AdaBoost, Bagging — and their regressor equivalents), or an estimator that is not in the registry at all. `best_params` then holds only fixed overrides (e.g. `class_weight`), usually `{}`.

If step 4 runs but does not improve on the step 3 baseline, `best_stage` stays at `"default"` or `"reduced"` and `best_params` holds only fixed overrides — no HPO benefit was found.

With `holdout_frac > 0`, rows are set aside before step 1, all four steps run on the remaining rows, and the final configuration is scored once on the held-out rows — see [Outer holdout](validation.md#outer-holdout).

`best_params` are overrides on top of the winning model's own estimator, not full constructor kwargs. Optuna trials, the step-4 CV and `build_final_model()` all build the model as `clone(winner_estimator).set_params(**best_params)`, so arguments you passed to the estimator (and custom model names or estimators outside the registry) carry through to the deployed model. The unfitted winner is stored as `result.best_base_estimator`.

The rest of this section covers:

- [Configuration](configuration.md) — every `PipelineConfig` option, and the selection options that change how step 3 chooses between all and reduced features
- [Results](results.md) — what a `PipelineResult` holds, how to read its warnings and selection robustness, and how to build the final model
- [Validation & robustness](validation.md) — outer holdout, nested CV, seed stability and comparing selection metrics
- [Spatial CV](spatial_cv.md) — the spatial splitters used when `store.coords` is set

---

## Supported models

**Classifiers** — LogisticRegression, GaussianNB, KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, SVC, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, LGBMClassifier\*, CatBoostClassifier\*, XGBClassifier\*

**Regressors** — PoissonRegressor, KNeighborsRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, SVR, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor, LGBMRegressor\*, CatBoostRegressor\*, XGBRegressor\*

\* Registered only when the package is installed (`uv sync --extra boosting`). Custom models can be injected via the `models` argument to `H2MLPipeline`; see [Custom Models and Search Spaces](guides/models.md).
