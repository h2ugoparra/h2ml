# Configuration

`PipelineConfig` holds every pipeline option. Full parameter reference:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_type` | `"classification"` | `"classification"` or `"regression"` (case-insensitive); a `TaskType` member is also accepted |
| `metric` | `"AUC"` | Optimisation target. Classification: `"AUC"`, `"AUCPR"`, `"F1"`, `"LogLoss"`, `"Brier"`. Regression: `"R2"`, `"MAE"`, `"RMSE"`, `"PoissonDev"`. Minimisation direction is derived automatically. `"AUC_PR"` is a deprecated alias for `"AUCPR"`. |
| `n_splits` | `5` | CV folds for steps 1 and 3 |
| `n_jobs` | `-1` | Worker processes used to CV models in parallel (steps 1 and 3) and to explain row chunks in step 2's SHAP pass for model-agnostic explainers (SVC/SVR with non-linear kernels, KNN, …). `-1` uses all cores; must be non-zero. Lower it when memory rather than CPU is the constraint, or to leave headroom on a shared machine. |
| `opt_n_splits` | `3` | CV folds inside each Optuna trial (fewer = faster). Trials then train on (k−1)/k = 67% of the rows while the step-4 acceptance CV trains on 80% (`n_splits=5`), so the tuned regularisation can be slightly stronger than the final fit needs. Set `opt_n_splits=n_splits` to tune on the same fold size, at about `n_splits/opt_n_splits`× the HPO cost. |
| `corr_threshold` | `0.7` | Feature drop threshold. A feature is removed if it exceeds this value in **any** of Pearson, Spearman, or Kendall correlation with a higher-ranked feature. |
| `min_features` | `1` | Hard lower bound on retained features after the correlation filter |
| `n_trials` | `50` | Optuna trial budget in step 4 |
| `n_hpo_repeats` | `1` | Independent HPO runs with different fold seeds; the best repeat is kept |
| `handle_imbalance` | `False` | Inject `class_weight="balanced"` for classifiers that support it. `"auto"` injects only when the minority-class ratio is below 0.1 |
| `max_vif` | `None` | Step 2: after the correlation filter, drop the lowest-ranked feature whose variance inflation factor exceeds this value until none does. It catches features that are linear combinations of several retained ones, which pairwise correlation misses. `5` or `10` are common choices. |
| `prefer_reduced_on_tie` | `False` | Step 3: when an all-features winner is statistically tied (paired folds, within one corrected SE) with a reduced-feature candidate, select the best-ranked tied reduced candidate instead. See [Preferring reduced features on ties](#preferring-reduced-features-on-ties). |
| `selection_scoring` | `"pooled"` | Regression selection in steps 1, 3 and 4: one score over all out-of-fold rows. `"fold_mean"` uses the mean of fold scores with a fold-spread penalty. Classification always uses the fold mean. |
| `mean_correction` | `None` | `"ratio"`: `build_final_model()` rescales the predictions of `log`/`sqrt`-family models so their total matches the observed out-of-fold total (`FinalModel.mean_ratio`, clamped to 0.1–10). CV scores, selection and `holdout_df` stay uncorrected. See [Predicted totals](guides/transforms.md#predicted-totals-and-mean_correction). |
| `select_features_in_folds` | `False` | Step 3: re-run the step-2 selector on each fold's training rows and score reduced candidates on those per-fold sets, so the default-vs-reduced comparison is unbiased. Costs about `n_splits` extra step-2 runs. See [Selecting features inside CV folds](#selecting-features-inside-cv-folds). |
| `holdout_frac` | `0.0` | Fraction of rows set aside before step 1 and scored once at the end. `0` disables it; range `[0, 0.5]`. See [Outer holdout](validation.md#outer-holdout). |
| `holdout_strategy` | `"auto"` | `"auto"` — whole spatial blocks when `store.coords` is set, else random (stratified for classification). `"random"` — random even with coords. `"time"` — the latest rows by `store.times`. |
| `holdout_refit` | `"dev"` | Rows `build_final_model()` fits on when a holdout exists: `"dev"` (the steps' rows only) or `"all"` (dev + holdout) |
| `random_state` | `42` | Global seed for folds, models, and Optuna |
| `verbose` | `False` | Log step-by-step progress |
| `time_bin_resolution` | `"month"` | Time bin granularity for compound (spatial block × time) conformal cells: `"month"` (bins 1–12) or `"season"` (0=DJF, 1=MAM, 2=JJA, 3=SON). Only used when `store.times` is provided. |

Spatial CV parameters are documented in [Spatial CV](spatial_cv.md).

---

## Selection options

Two options change how step 3 chooses between the all-features and reduced-feature stages.

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
