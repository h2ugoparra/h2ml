# Validation & robustness

## Outer holdout

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

- **Spatial data** — `"auto"` draws whole blocks of the pipeline's spatial splitter at
  random until the total is closest to `holdout_frac`. The holdout estimates typical
  performance away from immediate neighbours, while spatial CV tests extrapolation.
  Scattered blocks sit closer to training rows than one held-out region would, and
  `spcv` blocks are small, so the score may be somewhat less strict. Blocks depend only
  on coordinates, so the same rows and seed give the same holdout for any target or
  feature set.
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
- **Rare classes** — a holdout's usefulness depends on how many rows of the rarest
  class it holds, not on its total size. Spatial blocks can be very uneven in
  prevalence. Below 20 such rows, a `[holdout]` warning is added. For very rare
  targets, prefer `holdout_frac=0`: CV scores, with conformal calibrated on the CV
  residuals.
- **Dropping thin holdouts automatically** — `holdout_min_minority=20` makes that
  choice per run: when the drawn holdout holds fewer rows than this of its rarest
  class (classification) or of non-zero rows (regression), the pipeline drops it and
  runs as with `holdout_frac=0`. The count is taken on the actual split, and the
  decision is recorded as a `[holdout] Holdout dropped` warning. Useful when one config
  runs many targets of very different prevalence:

    ```python
    config = PipelineConfig(task_type="classification", holdout_frac=0.2, holdout_min_minority=20)
    result = H2MLPipeline(config).run(store)
    result.holdout_store is None  # True when the holdout was dropped
    ```

- **Other `[holdout]` warnings** — `cv_warnings` also flags fewer than 30 rows and a
  size more than 25% from the request. For regression it flags fewer than 20 non-zero
  rows on a target with zeros, and a mean or non-zero rate more than 2× from the dev
  rows. A time holdout can trigger the last check through real drift.

## Nested CV

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

## Seed stability

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

## Comparing selection metrics

The metric decides the winner in steps 1 and 3 and what Optuna optimises in step 4.
`multi_metric` runs the pipeline under several metrics and shares the work that does
not depend on the metric:

```python
from h2ml import multi_metric

mm = multi_metric(
    H2MLPipeline(config), store,
    metrics=["Brier", "LogLoss", "AUC", "AUCPR", "F1"],     # the default (all metrics, this order)
    handle_imbalance={"Brier": False, "LogLoss": False},    # per-metric override of the config
)

mm.runs_df                 # per metric: Model / Stage / Feature_stage / N_features / Class_weight,
                           # CV, CV_Std and Holdout under that metric, Step4_shared_with
mm.results["AUC"]          # a normal PipelineResult: build_final_model(), save(), plots
mm.summary()               # N_distinct_selections, Top_Selection(_Fraction), N_hpo_runs
mm.selection_counts()      # how many metrics chose each (model, transform, stage)
mm.save(path); MultiMetricResult.load(path)
```

Regression works the same way, with the regression metrics and an optional y-transform
sweep (`handle_imbalance` is classification-only and rejected here):

```python
mm = multi_metric(
    H2MLPipeline(reg_config), store,
    metrics=["RMSE", "R2", "MAE"],      # the default; add "PoissonDev" for non-negative targets
    transforms=["log"],                 # swept in step 1, as in run(store, transforms=...)
    reuse_step4="model",
)
mm.runs_df["Y_transform"]               # winning transform per metric
```

What runs once and what runs per metric:

| Work | Runs |
|---|---|
| Holdout split | once |
| Step 1 CV (every model × y-transform, every metric computed) | once per class-weighting setting (once for regression) |
| Steps 2–3 (SHAP, filters, reduced-feature CV) | once per (class weighting, step-1 winner, y-transform) |
| Step 1 / step 3 selection, step 4 HPO, holdout score | per metric |

- With the default `reuse_step4="none"`, `mm.results[m]` equals a separate
  `H2MLPipeline(replace(config, metric=m)).run(store)`. The speed-up comes only from
  not repeating identical CVs.
- `reuse_step4="model"`: when a metric selects the same (class weighting, model,
  transform, feature stage, feature set) as an earlier one, it reuses that tuning instead of running
  Optuna again (`Step4_shared_with` names the earlier metric). Whether the tuned params
  beat step 3 is still decided under each metric, but they were optimised for the
  earlier one.
- Each metric gets its own copy of the models, so one metric's class weighting cannot
  leak into another's. Class weights distort probabilities, which is why the example
  keeps Brier and LogLoss unweighted.
- Reading it: `Top_Selection_Fraction` of 1 means the choice does not depend on the
  metric. When the metrics disagree, pick the metric that matches how predictions will
  be used (probabilities → Brier/LogLoss, ranking → AUC/AUCPR, thresholded decisions → F1)
  rather than the best-looking score. Scores are only comparable within a metric.
- Run order matters with `reuse_step4="model"`: the first metric to select a
  configuration tunes it, and later metrics reuse that tuning. The default order puts
  proper scoring rules first — Brier then LogLoss (probabilities), RMSE then R2 (the
  conditional mean; per fold, R2 = 1 − MSE / fold variance, so the two usually agree) —
  and F1 and MAE (a median target, more often a different model) last. Pass `metrics=`
  to put first the metric your predictions are judged by.
