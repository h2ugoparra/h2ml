# Pipeline

## The 4 steps

| Step | What happens | Key outputs |
|------|-------------|-------------|
| 1 | K-fold CV all models (× y-transforms) on all features | `best_model_name`, `step1_agg_df` |
| 2 | Fit best model → OOF SHAP → correlation filter | `features_reduced`, `selector` |
| 3 | K-fold CV all models on reduced features; compare vs step 1 | `best_stage`, `step3_agg_df` |
| 4 | Optuna HPO on winning (model, stage, transform) | `best_params`, `step4_agg_df` |

Step 4 is skipped when the winning model has no enabled search space: `opt_enabled=False` (LogisticRegression, GaussianNB, KNeighborsClassifier, AdaBoost, Bagging — and their regressor equivalents), or an estimator that is not in the registry at all. `best_params` then holds only fixed overrides (e.g. `class_weight`), usually `{}`.

If step 4 runs but does not improve on the step 3 baseline, `best_stage` stays at `"default"` or `"reduced"` and `best_params` holds only fixed overrides — no HPO benefit was found.

`best_params` are overrides on top of the winning model's own estimator, not full constructor kwargs. Optuna trials, the step-4 CV and `build_final_model()` all build the model as `clone(winner_estimator).set_params(**best_params)`, so arguments you passed to the estimator (and custom model names or estimators outside the registry) carry through to the deployed model. The unfitted winner is stored as `result.best_base_estimator`.

---

## `PipelineConfig`

Full parameter reference:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_type` | `"classification"` | `"classification"` or `"regression"` (case-insensitive); a `TaskType` member is also accepted |
| `metric` | `"AUC"` | Optimisation target. Classification: `"AUC"`, `"AUCPR"`, `"F1"`, `"LogLoss"`, `"Brier"`. Regression: `"R2"`, `"MAE"`, `"RMSE"`. Minimisation direction is derived automatically. `"AUC_PR"` is a deprecated alias for `"AUCPR"`. |
| `n_splits` | `5` | CV folds for steps 1 and 3 |
| `n_jobs` | `-1` | Worker processes used to CV models in parallel (steps 1 and 3). `-1` uses all cores; must be non-zero. Lower it when memory rather than CPU is the constraint, or to leave headroom on a shared machine. |
| `opt_n_splits` | `3` | CV folds inside each Optuna trial (fewer = faster) |
| `corr_threshold` | `0.7` | Feature drop threshold. A feature is removed if it exceeds this value in **any** of Pearson, Spearman, or Kendall correlation with a higher-ranked feature. |
| `min_features` | `1` | Hard lower bound on retained features after the correlation filter |
| `n_trials` | `50` | Optuna trial budget in step 4 |
| `n_hpo_repeats` | `1` | Independent HPO runs with different fold seeds; the best repeat is kept |
| `handle_imbalance` | `False` | Inject `class_weight="balanced"` for classifiers that support it. `"auto"` injects only when the minority-class ratio is below 0.1 |
| `random_state` | `42` | Global seed for folds, models, and Optuna |
| `verbose` | `False` | Log step-by-step progress |
| `time_bin_resolution` | `"month"` | Time bin granularity for compound (spatial block × time) conformal cells: `"month"` (bins 1–12) or `"season"` (0=DJF, 1=MAM, 2=JJA, 3=SON). Only used when `store.times` is provided. |

Spatial CV parameters are documented in [Spatial CV](spatial_cv.md).

---

## `PipelineResult`

All artifacts produced by a run. Fields are populated progressively — check `completed_steps` to see which steps have finished.

### Inspection

```python
result.completed_steps        # e.g. [1, 2, 3, 4]
result.summary()              # combined agg DataFrame across all stages
result.summary("AUC_Test_Mean", ascending=False)
# Column names are <Metric>_<Test|Train|Gap>_<Mean|Std>. "Test" is the held-out
# fold of each CV split, so AUC_Test_Mean is the mean of per-fold held-out scores,
# not one score on the pooled OOF predictions (result.oof_predictions).

result.best_model_name        # winning model class name
result.best_model_value       # best CV score (maximised internally)
result.best_model_std         # std of that score across folds
result.best_stage             # "default" | "reduced" | "optimized"
result.best_feature_stage     # "default" | "reduced" (never "optimized")
result.best_params            # HPO overrides on top of best_base_estimator ({} or fixed-only if step 4 skipped)
result.y_transform            # winning y-transform (regression only, or None)
result.cv_type                # "spatial" | "random"
result.spatial_cv_metric      # "euclidean" | "haversine" — forwarded from config
result.time_bin_resolution    # "month" | "season" — forwarded from config
result.cv_warnings            # list[str] — failed CV folds, metric-suitability and
                              # ranking-robustness flags (advisory; never change selection)
result.metric                 # short metric name, e.g. "AUC"
```

### Selection robustness

```python
result.ranking_robustness()        # step 3 if run, else step 1
result.ranking_robustness(step=1)  # models × transforms on all features
```

One row per candidate. `Winner_Gap` is the mean per-fold difference to the winner
(positive = winner better), paired on shared folds so fold difficulty cancels;
`Gap_SE` is its Nadeau–Bengio corrected standard error. `Indistinguishable` marks
candidates within one SE of the winner — a heuristic tie, not a significance test.
`Rank_<metric>` columns show where each candidate ranks under every metric.

### Per-step data

```python
result.step1_fold_df          # per-fold metrics, all models, step 1
result.step1_agg_df           # mean/std per model, step 1
result.step3_fold_df          # per-fold metrics, all models, step 3
result.step3_agg_df           # mean/std per model, step 3
result.step4_fold_df          # per-fold metrics, winning model, step 4
result.step4_agg_df           # mean/std, winning model, step 4
```

### Feature selection

```python
result.features               # PipelineData — full feature set
result.features_reduced       # PipelineData — after step 2 selection
result.selector               # FeatureSelector (fitted)
result.selector.importance_summary()   # SHAP importances as DataFrame
result.selector.selected_features_    # list[str]
result.selector.removed_features      # list[str]
```

### OOF predictions

```python
result.oof_predictions        # assembled OOF probabilities or values
result.oof_labels             # matching true labels
result.best_cv_result         # CVResult for the final winning model
```

### Building the final model

```python
final = result.build_final_model()
```

See [Persistence](guides/persistence.md) and [Conformal Prediction](guides/conformal.md).

---

## Supported models

**Classifiers** — LogisticRegression, GaussianNB, KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, SVC, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, LGBMClassifier\*, CatBoostClassifier\*, XGBClassifier\*

**Regressors** — PoissonRegressor, KNeighborsRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, SVR, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor, LGBMRegressor\*, CatBoostRegressor\*, XGBRegressor\*

\* Registered only when the package is installed (`uv sync --extra boosting`). Custom models can be injected via the `models` argument to `H2MLPipeline`.
