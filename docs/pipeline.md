# Pipeline

## The 4 steps

| Step | What happens | Key outputs |
|------|-------------|-------------|
| 1 | K-fold CV all models (× y-transforms) on all features | `best_model_name`, `step1_agg_df` |
| 2 | Fit best model → OOF SHAP → correlation filter | `features_reduced`, `selector` |
| 3 | K-fold CV all models on reduced features; compare vs step 1 | `best_stage`, `step3_agg_df` |
| 4 | Optuna HPO on winning (model, stage, transform) | `best_params`, `step4_agg_df` |

Steps 1 and 3 pick the candidate with the best confidence-bound score (mean − std/√k for maximised metrics, mean + std/√k for minimised ones). With `prefer_reduced_on_tie=True`, step 3 then applies a 1-SE rule: if the winner uses all features but a reduced-feature candidate is statistically tied with it on paired folds, the reduced candidate is selected instead — see [Preferring reduced features on ties](#preferring-reduced-features-on-ties).

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
| `n_jobs` | `-1` | Worker processes used to CV models in parallel (steps 1 and 3) and to explain row chunks in step 2's SHAP pass for model-agnostic explainers (SVC/SVR with non-linear kernels, KNN, …). `-1` uses all cores; must be non-zero. Lower it when memory rather than CPU is the constraint, or to leave headroom on a shared machine. |
| `opt_n_splits` | `3` | CV folds inside each Optuna trial (fewer = faster) |
| `corr_threshold` | `0.7` | Feature drop threshold. A feature is removed if it exceeds this value in **any** of Pearson, Spearman, or Kendall correlation with a higher-ranked feature. |
| `min_features` | `1` | Hard lower bound on retained features after the correlation filter |
| `n_trials` | `50` | Optuna trial budget in step 4 |
| `n_hpo_repeats` | `1` | Independent HPO runs with different fold seeds; the best repeat is kept |
| `handle_imbalance` | `False` | Inject `class_weight="balanced"` for classifiers that support it. `"auto"` injects only when the minority-class ratio is below 0.1 |
| `prefer_reduced_on_tie` | `False` | Step 3: when an all-features winner is statistically tied (paired folds, within one corrected SE) with a reduced-feature candidate, select the best-ranked tied reduced candidate instead. See [Preferring reduced features on ties](#preferring-reduced-features-on-ties). |
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
result.cv_warnings            # list[str] — failed CV folds, [metric] suitability and
                              # [ranking] robustness flags, prefer_reduced_on_tie switches
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
A negative `Winner_Gap` (better mean, but penalised for fold-to-fold spread by the
confidence bound) is always flagged. `Folds_Won` counts the folds where the *winner*
beat that row's candidate, out of `Paired_Folds` (folds both completed; a NaN score
drops a fold from that pair only) — a low count means the candidate often beat the
winner. The winner's own row reads gap 0, `Gap_SE` NaN, `Folds_Won` 0.
`Rank_<metric>` columns show where each candidate ranks under every metric.
`Is_Winner` marks the confidence-bound winner — the candidate *before* any
`prefer_reduced_on_tie` switch, so it can differ from `result.best_stage`.

### Warnings in `cv_warnings`

All are advisory except the `prefer_reduced_on_tie` line, which records a selection
change you opted into. Each is also logged when it is raised.

| `cv_warnings` entry starts with | Log prefix | Raised in | Meaning |
|---|---|---|---|
| a model name, e.g. `RF: 2/5 fold(s) failed` | `[CV]` (end of `run()`) | steps 1, 3 | A model had failed CV folds |
| the metric name, e.g. `AUC with minority…`, `R2: …` | `[metric]` | step 1 | The data makes the configured metric unreliable — AUC under class imbalance, single-class test folds for AUC/AUCPR, uneven fold prevalence for AUCPR, F1 with class weights, low-variance folds for R2, heavy tails for RMSE, zero inflation for MAE. The metric is never changed. |
| `step 1:` / `step 3:` | `[ranking]` | steps 1, 3 | Candidates tied with the winner on paired folds, or the winner ranking below 1st on the counterpart-family metric (AUC/AUCPR ↔ Brier, LogLoss/Brier ↔ AUC, F1 → AUC, R2/RMSE ↔ MAE) |
| `step 3: prefer_reduced_on_tie →` | *(info log)* | step 3 | The tie-break switched the selection to a reduced-feature candidate |

### Preferring reduced features on ties

Tree ensembles predict well with correlated features, so the all-features ("default")
stage often wins step 3 by a margin smaller than the CV noise. The reduced set is
usually the better deployment choice in that case: it is simpler, more stable across
runs, and less likely to rely on a proxy feature whose correlation with the true
driver changes in new regions or periods.

`PipelineConfig(prefer_reduced_on_tie=True)` applies a one-standard-error rule after
step 3's selection:

1. Only acts when the winner is a `"default"`-stage candidate and the reduced set
   dropped at least one feature.
2. Takes the reduced candidates (any model, the winning transform) whose paired-fold
   gap to the winner is within one Nadeau–Bengio corrected SE — the `Indistinguishable`
   rows of `result.ranking_robustness(step=3)`.
3. Selects the best-ranked of them under `metric`, and sets `best_model_name`,
   `best_stage`/`best_feature_stage`, `best_model_value`/`best_model_std` and
   `y_transform` from its row.

Step 4 then tunes the reduced candidate on the reduced features, and judges
improvement against the reduced candidate's own score. The switch is logged and
recorded in `cv_warnings`, e.g.
`step 3: prefer_reduced_on_tie → SVC | reduced (tied with RandomForestClassifier | default: gap 0.003 ≤ SE 0.008)`.

The paired SE is used rather than the classic unpaired one (std/√k of the winner's
score) because the unpaired SE includes fold difficulty, which under spatial CV is
large enough that almost every candidate would tie. Step 2 selects features using
all the data, which makes the reduced stage's CV score slightly optimistic; that
nudges toward reduced in the same direction as this rule.

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
