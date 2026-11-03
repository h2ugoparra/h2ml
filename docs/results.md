# Results

`pipeline.run()` returns a `PipelineResult` holding all artifacts produced by the run. Fields are populated progressively — check `completed_steps` to see which steps have finished.

## Inspection

```python
result.completed_steps        # e.g. [1, 2, 3, 4]
result.summary()              # combined agg DataFrame across all stages
result.summary("AUC_Test_Mean", ascending=False)
# Column names are <Metric>_<Test|Train|Gap>_<Mean|Std>. "Test" is the held-out
# fold of each CV split, so AUC_Test_Mean is the mean of per-fold held-out scores.
# Regression also has <Metric>_Test_Pooled: one score over all out-of-fold rows.
# R2 is NaN on a fold with zero target variance, so R2_Test_Mean skips that fold.

result.best_model_name        # winning model class name
result.best_model_value       # best CV score, pooled for regression by default
result.selection_scoring      # "pooled" | "fold_mean" — the rule this result selected with
result.best_model_std         # std of that score across folds
result.best_stage             # "default" | "reduced" | "optimized"
result.best_feature_stage     # "default" | "reduced" (never "optimized")
result.best_params            # HPO overrides on top of best_base_estimator ({} or fixed-only if step 4 skipped)
result.y_transform            # winning y-transform (regression only, or None)
result.cv_type                # "spatial" | "random"
result.spatial_cv_metric      # "euclidean" | "haversine" — forwarded from config
result.time_bin_resolution    # "month" | "season" — forwarded from config
result.cv_warnings            # list[str] — failed CV folds, [metric], [ranking], [selection],
                              # [transform] and [holdout] flags, prefer_reduced_on_tie switches
                              # (see "Warnings in cv_warnings" below)
result.metric                 # short metric name, e.g. "AUC"
```

## Selection robustness

```python
result.ranking_robustness()        # step 3 if run, else step 1
result.ranking_robustness(step=1)  # models × transforms on all features
```

One row per candidate. `Winner_Gap` is the mean per-fold difference to the winner
(positive = winner better), paired on shared folds so fold difficulty cancels;
`Gap_SE` is its Nadeau–Bengio corrected standard error. `Indistinguishable` marks
candidates within one SE of the winner — a heuristic tie, not a significance test.
A negative `Winner_Gap` is always flagged: the candidate has the better fold mean, but
the winner has the better pooled score (regression) or the fold-spread penalty counted
against the candidate (classification). `Folds_Won` counts the folds where the *winner*
beat that row's candidate, out of `Paired_Folds` (folds both completed; a NaN score
drops a fold from that pair only) — a low count means the candidate often beat the
winner. The winner's own row reads gap 0, `Gap_SE` NaN, `Folds_Won` 0.
`Rank_<metric>` columns show where each candidate ranks under every metric.
`Is_Flat` marks candidates with no prediction spread, which `Is_Winner` skips unless all
are flat. `Is_Winner` marks the selection's winner — the candidate *before* any
`prefer_reduced_on_tie` switch, so it can differ from `result.best_stage`.

## Warnings in `cv_warnings`

All are advisory except the `prefer_reduced_on_tie` line, which records a selection
change you opted into, and `[selection]`, which reports candidates the guard removed.
Each is also logged when it is raised.

| `cv_warnings` entry starts with | Log prefix | Raised in | Meaning |
|---|---|---|---|
| a model name, e.g. `RF: 2/5 fold(s) failed` | `[CV]` (end of `run()`) | steps 1, 3 | A model had failed CV folds |
| the metric name, e.g. `AUC with minority…`, `R2: …` | `[metric]` | step 1 | The data makes the configured metric unreliable — AUC under class imbalance, single-class test folds for AUC/AUCPR, uneven fold prevalence for AUCPR, F1 with class weights, low-variance folds for R2, heavy tails for RMSE, zero inflation for MAE. The metric is never changed. |
| `step 1:` / `step 3:` | `[ranking]` | steps 1, 3 | Candidates tied with the winner on paired folds, or the winner ranking below 1st on the counterpart-family metric (AUC/AUCPR ↔ Brier, LogLoss/Brier ↔ AUC, F1 → AUC, R2/RMSE/PoissonDev → MAE, MAE → RMSE) |
| `step 3: prefer_reduced_on_tie →` | *(info log)* | step 3 | The tie-break switched the selection to a reduced-feature candidate |
| `[selection]` | `[CV]` (end of `run()`) | steps 1, 3, 4 | Constant-prediction candidates were excluded (noting if one would have won), every candidate is constant, or tuning produced a constant model and was rejected |
| `[transform]` | `[CV]` (end of `run()`) | after step 3 or 4 | A log or sqrt-family winner's out-of-fold total is more than 20% from the observed total. See [`mean_correction`](guides/transforms.md#predicted-totals-and-mean_correction) |
| `[holdout]` | `[CV]` (end of `run()`) | run start; after step 3 or 4 | At the start: too few rows, a size far from `holdout_frac`, and for regression too few non-zero rows or a distribution far from the dev rows. After scoring: the holdout score is worse than the CV score by more than 2 fold SDs — the CV estimate is likely optimistic, or the holdout rows differ from the dev rows. Classification: also when the holdout holds fewer than 20 rows of its rarest class, so its score and conformal calibration are unreliable |

## Per-step data

```python
result.step1_fold_df          # per-fold metrics, all models, step 1
result.step1_agg_df           # mean/std per model, step 1
result.step3_fold_df          # per-fold metrics, all models, step 3
result.step3_agg_df           # mean/std per model, step 3
result.step4_fold_df          # per-fold metrics, winning model, step 4
result.step4_agg_df           # mean/std, winning model, step 4
```

For regression the agg frames also hold `<metric>_Test_Pooled`, the score selection
ranks on by default. Read it, not `<metric>_Test_Mean`, to see why a candidate won.

## Feature selection

```python
result.features               # PipelineData — full feature set (dev rows only with a holdout)
result.features_reduced       # PipelineData — after step 2 selection
result.selector               # FeatureSelector (fitted)
result.selector.importance_summary()   # SHAP importances as DataFrame
result.selector.selected_features_    # list[str]
result.selector.removed_features      # list[str]
result.features.to_frame()            # the feature matrix as a DataFrame

from h2ml.pipeline.nested import final_feature_names
final_feature_names(result)           # features the final model uses (reduced set if that stage won)
```

The step-2 filters are also available on their own, for example to test a threshold
before a run:

```python
from h2ml.features.correlation import (
    remove_correlated_features,    # (X_df, importance, corr_threshold) → kept features, by rank
    variance_inflation_factors,    # (X_df) → VIF per column
    remove_high_vif_features,      # (X_df, features, max_vif) → kept features
)
importance = result.selector.feature_importance_    # mean |SHAP| per feature, ranked
kept = remove_correlated_features(result.features.to_frame(), importance, corr_threshold=0.8)
```

## OOF predictions

```python
result.oof_predictions        # assembled OOF probabilities or values
result.oof_labels             # matching true labels
result.best_cv_result         # CVResult for the final winning model
```

`oof_predictions` and `oof_labels` are assembled from `best_cv_result`, a `CVResult`
that keeps every fold:

```python
cv = result.best_cv_result
cv.model_name, cv.y_transform, cv.n_folds
cv.failed_folds               # ids of folds that raised (excluded from .folds)

fold = cv.folds[0]            # FoldResult
fold.train_idx, fold.test_idx       # row positions in result.features
fold.y_test, fold.y_pred_test       # observed and predicted, original scale
fold.y_prob_test                    # class probabilities (classification)
fold.fit_time                       # seconds

cv.fold_arrays("y_pred_test") # one array per fold
```

The step-wise lists (`result.step1_cv_result`, `step3_cv_result`) hold one `CVResult`
per model × transform, and `result.holdout_cv_result` is a one-fold `CVResult` of the
holdout score.

## Building the final model

```python
final = result.build_final_model()
```

See [Persistence](guides/persistence.md) and [Conformal Prediction](guides/conformal.md).
