# Custom Models and Search Spaces

By default `H2MLPipeline` screens every model in the registry for the task. You can pass your own list instead, change a model's settings, or change what step 4 tunes.

## Passing your own models

Wrap each estimator with `make_classifier` or `make_regressor` and pass the list as `models`:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from h2ml import H2MLPipeline, PipelineConfig
from h2ml.pipeline.step import make_regressor

models = [
    make_regressor(RandomForestRegressor(n_estimators=300, min_samples_leaf=5), name="RF"),
    make_regressor(Ridge(alpha=1.0), requires_scaling=True),
]
pipeline = H2MLPipeline(PipelineConfig(task_type="regression", metric="RMSE"), models=models)
```

- **`name`** labels the model in every table and in `best_model_name`. It defaults to the class name, so two models of the same class need distinct names; `H2MLPipeline` raises a `ValueError` on repeated names.
- **`requires_scaling`** (default `False`) standardises features inside each fold, fitted on that fold's training rows. Set it for scale-sensitive models (linear models, SVMs, k-NN). Registry models already carry the right flag.
- **Your constructor arguments are kept.** Step 4's tuned values are applied on top of a clone of your estimator, so settings the search space doesn't touch survive into the final model. The unfitted winner is stored as `result.best_base_estimator`.
- Any estimator with sklearn's `fit`/`predict` (and `predict_proba` for classifiers) works.

## The model registry

The registry describes every built-in model: its class, default arguments, scaling flag and search space. `build_models(task)` builds the default list, which is also the easiest way to run a subset:

```python
from h2ml.utils.registry import REGRESSOR_REGISTRY, build_models

list(REGRESSOR_REGISTRY)            # registered regressor names
keep = {"RandomForestRegressor", "HistGradientBoostingRegressor", "SVR"}
models = [m for m in build_models("regression") if m.name in keep]
```

Each entry is a `ModelEntry` (`CLASSIFIER_REGISTRY` and `REGRESSOR_REGISTRY` hold them):

| Field | Meaning |
|---|---|
| `model_cls` | The estimator class |
| `default_kwargs` | Constructor arguments for the default model |
| `requires_scaling` | Whether features are standardised inside each fold |
| `param_fn` | The Optuna search space for step 4, or `None` |
| `opt_enabled` | `False` keeps the model out of step 4 |
| `supports_class_weight` | Whether `handle_imbalance` may inject `class_weight="balanced"` |

LightGBM, XGBoost and CatBoost entries appear only when the package is installed.

## What step 4 tunes

Step 4 looks up the winner's **class name** in the registry, not its display name:

- A registered class you configured yourself (e.g. `RandomForestRegressor` named `"RF"`) is tuned with the registry's search space.
- A class outside the registry, or one with `opt_enabled=False` (LogisticRegression, GaussianNB, k-NN, Bagging, AdaBoost, PoissonRegressor), skips step 4. `best_params` then holds only fixed overrides such as `class_weight`, usually `{}`.
- `handle_imbalance` likewise applies only to registered classifiers with `supports_class_weight=True`.

The search spaces are functions of an Optuna `trial`, in `h2ml.core.param_spaces.classifiers` and `h2ml.core.param_spaces.regressors`. For example, the random forest regressor's:

```python
def randomforest_r_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 32),
        ...
    }
```

## Changing a search space

Replace the entry's `param_fn` before running the pipeline, in the same process:

```python
from dataclasses import replace
from h2ml.utils.registry import REGRESSOR_REGISTRY

def shallow_forest(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
    }

entry = REGRESSOR_REGISTRY["RandomForestRegressor"]
REGRESSOR_REGISTRY["RandomForestRegressor"] = replace(entry, param_fn=shallow_forest)
```

The same pattern adds a search space to an unregistered class: register a `ModelEntry(model_cls=Ridge, param_fn=..., requires_scaling=True)` under the name `"Ridge"`.

- Return only sampled values, plus runtime settings such as `n_jobs` that don't change predictions. `best_params` keeps only what Optuna sampled, so any other constant in the returned dict would apply during trials but not to the tuned or final model. Set such constants on the estimator you pass, or in the entry's `default_kwargs`. The built-in spaces follow this rule.
- The change lasts until the process ends, so it also affects later runs in the same session.

## Tuning outside the pipeline

`optimize_all` runs a study per registered model on plain arrays, without steps 1–3:

```python
from h2ml.optimization.optimizer import optimize_all

studies = optimize_all(
    ["RandomForestRegressor", "HistGradientBoostingRegressor"],
    X, y, task="regression", metric="RMSE", n_trials=30,
)
for name, study in studies.items():
    print(name, -study.best_value, study.best_params)   # error metrics are stored negated
```

Studies always maximise, so `best_value` is negative for `MAE`, `RMSE`, `PoissonDev`, `LogLoss` and `Brier`. It uses k-fold CV (stratified for classification), with no spatial folds, y-transforms or holdout. Models without an enabled search space are skipped with a warning. For a single model with those options, see `run_study` in the [Optimizer API](../api/optimizer.md).
