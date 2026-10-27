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

With `holdout_frac > 0`, rows are set aside before step 1, all four steps run on the remaining rows, and the final configuration is scored once on the held-out rows — see [Outer holdout](#outer-holdout).

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
| `select_features_in_folds` | `False` | Step 3: re-run the step-2 selector on each fold's training rows and score reduced candidates on those per-fold sets, so the default-vs-reduced comparison is unbiased. Costs about `n_splits` extra step-2 runs. See [Selecting features inside CV folds](#selecting-features-inside-cv-folds). |
| `holdout_frac` | `0.0` | Fraction of rows set aside before step 1 and scored once at the end. `0` disables it; range `[0, 0.5]`. See [Outer holdout](#outer-holdout). |
| `holdout_strategy` | `"auto"` | `"auto"` — whole spatial blocks when `store.coords` is set, else random (stratified for classification). `"random"` — random even with coords. `"time"` — the latest rows by `store.times`. |
| `holdout_refit` | `"dev"` | Rows `build_final_model()` fits on when a holdout exists: `"dev"` (the steps' rows only) or `"all"` (dev + holdout) |
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
| `[holdout]` | `[CV]` (end of `run()`) | after step 3 or 4 | The holdout score is worse than the CV score by more than 2 fold SDs — the CV estimate is likely optimistic, or the holdout rows differ from the dev rows |

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
large enough that almost every candidate would tie. By default step 2 selects
features using all the data, which makes the reduced stage's CV score slightly
optimistic; that nudges toward reduced in the same direction as this rule.
`select_features_in_folds=True` removes that bias.

### Selecting features inside CV folds

Step 2 picks the reduced feature set from SHAP importances computed on every dev row,
including the rows step 3 then tests on. The reduced candidates' step-3 scores therefore
reflect a feature set chosen partly on their own test folds, so `"reduced"` can beat
`"default"` because of that leak rather than because it really generalises better.

With `PipelineConfig(select_features_in_folds=True)`, step 3 re-runs the same selector
(winning model, OOF SHAP, correlation filter) on each CV fold's **training rows only**.
Every model's reduced candidate is then fitted and scored per fold on that fold's own
feature subset:

```python
result = H2MLPipeline(PipelineConfig(select_features_in_folds=True)).run(store)
result.step3_fold_features      # list per fold of the features selected on its training rows
result.features_reduced         # the deployed set — still step 2's selection on all dev rows
```

- The reduced stage's score now estimates how well the *selection procedure* generalises,
  which is what the default-vs-reduced choice should compare. The feature set you deploy
  is still the one step 2 chose on all dev rows.
- When `"reduced"` wins, step 4's acceptance CV (tuned vs untuned) also uses the per-fold
  sets, so both sides are estimated the same way. The Optuna trials themselves still use
  the fixed step-2 set.
- With spatial CV, each fold's inner SHAP folds are whole blocks of the outer splitter
  (`BlockGroupSplitter`), so the inner selection is spatially separated too.
- Cost: one extra step-2 run per fold — about `n_splits` × step 2. That is cheap for tree
  models (TreeExplainer) but expensive for model-agnostic SHAP (e.g. RBF SVC/SVR).
- Comparing the sets in `step3_fold_features` with each other and with
  `features_reduced` shows how stable the selection is. Large differences mean the
  reduced set depends on which rows it was chosen from.

### Outer holdout

Every step selects on the same CV folds: models × transforms in step 1, features in
step 2, the stage in step 3, hyperparameters in step 4. The winner's CV score is
therefore the best of many noisy estimates and tends to be optimistic. Setting
`holdout_frac` (0.2 is a sensible start) gives one score no selection step has seen:

```python
config = PipelineConfig(holdout_frac=0.2)          # holdout_strategy="auto"
result = H2MLPipeline(config).run(store)

result.features.n_samples       # dev rows — steps 1–4 ran on these only
result.holdout_store            # raw held-out rows (PipelineData)
result.holdout_idx              # their row indices in `store`
result.holdout_strategy         # "spatial", "random" or "time"
result.holdout_df               # holdout metrics, same *_Mean columns as the agg frames
result.summary()                # includes a "holdout" row
```

The final configuration (model, y-transform, feature stage, `best_params`) is fitted
on the dev rows and scored on the holdout. Scaling, winsorize clipping and inverse
transforms are applied as in CV. The gap between `result.best_model_value` and the
holdout score shows how much selection inflated the CV score. With a holdout of
n rows, the score itself has sampling noise of roughly 1/√n, so read a small gap
as "no evidence of optimism" rather than as an exact correction.

- **Spatial data** — `"auto"` holds out whole blocks of the pipeline's spatial splitter
  (the fold closest to `holdout_frac` of `round(1 / holdout_frac)` folds), so held-out
  rows are not neighbours of training rows.
- **Forecasting** — `holdout_strategy="time"` holds out the latest rows by
  `store.times`, testing prediction forward in time, which spatial CV does not.
- **Partial runs** — every entry point sets the holdout aside; `run_step1_to_step3()`,
  `run_from_step3()` and `run_step4_only()` score it when they finish. Resuming keeps
  the original split because `result.features` already holds only the dev rows.
- **Final model** — conformal calibration uses the holdout residuals (split conformal),
  and local conformal calibration is skipped. With the default `holdout_refit="dev"`,
  the holdout score and interval coverage describe the deployed model exactly;
  `holdout_refit="all"` refits on dev + holdout for more data, after which both are
  approximate.

### Nested CV

A holdout scores one run once. `nested_cv` runs the whole pipeline inside each of k
outer folds and scores each run's final configuration on its outer test fold. That
estimates how well the *procedure* generalises and shows how stable its choices are:

```python
from h2ml import nested_cv

nested = nested_cv(H2MLPipeline(config), store, outer_splits=5)   # transforms=[...] for regression

nested.summary()             # Outer_<metric>_Mean/_Std, Inner_<metric>_Mean, Optimism_Mean
nested.folds_df              # per outer fold: chosen Model / Y_transform / Stage / N_features,
                             # Inner_<metric> (the CV estimate the run reported) and outer test metrics
nested.selection_counts()    # how often each (model, transform, stage) won
nested.feature_frequency()   # fraction of folds whose final configuration used each feature
nested.results               # the inner PipelineResults (outer test fold attached as their holdout)
```

- `Optimism_Mean` is sign-adjusted: positive always means the inner CV estimate was
  too optimistic.
- Outer folds follow the pipeline's own rules: spatial blocks when `store.coords` is set,
  else stratified (classification) or plain KFold. Each inner run builds its own splitter
  on the outer training rows, so no selection step sees the outer test rows.
- Outer folds run one after another; `n_jobs` applies inside each run. The cost is about
  `outer_splits` × one `run()`.
- `holdout_frac` is ignored inside the loop because the outer test fold plays that role.
  A fold whose inner run raises is logged, listed in `failed_folds`, and skipped.
- It does not produce a model to deploy. For that, run `run()` on all the data (optionally
  with a holdout); `nested_cv` tells you what to expect from it.
- Many distinct winners across folds, or features used in only some folds, mean the
  pipeline's choices depend on which rows it sees. Read the reported winner as one of
  several near-equivalent options.

### Seed stability

Each run uses one `random_state`. It decides the CV folds (and the spatial partition),
SHAP sampling, the Optuna sampler and the holdout split. With 5 folds, another seed
can change the winner or move the score by more than the gap between candidates.
`seed_stability` reruns the pipeline under several seeds:

```python
from h2ml import seed_stability

stab = seed_stability(H2MLPipeline(config), store, seeds=range(5))   # through_step=3 by default

stab.summary()             # <metric>_Mean/_Std/_Min/_Max across seeds, Top_Selection(_Fraction)
stab.runs_df               # per seed: Model / Y_transform / Stage / N_features, reported score
                           # (and Holdout_<metric> when holdout_frac > 0)
stab.selection_counts()    # how often each (model, transform, stage) won
stab.feature_frequency()   # share of seeds whose final configuration used each feature
```

- `through_step`: `1` screens models only, `3` (default) runs selection without HPO,
  and `4` runs the full pipeline.
- Seeds set inside the model constructors (e.g. a registry RandomForest's
  `random_state`) are not changed, so the spread isolates the evaluation and selection
  procedure.
- Reading it: if `<metric>_Std` across seeds is comparable to the gaps in
  `ranking_robustness()`, or `Top_Selection_Fraction` is well below 1, the single-run
  winner is partly luck. Treat the candidates that win across seeds as equivalent and
  prefer the simplest. It complements `nested_cv`: that function measures
  generalisation, while this one measures how much a run's choices depend on the seed.

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
result.features               # PipelineData — full feature set (dev rows only with a holdout)
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
