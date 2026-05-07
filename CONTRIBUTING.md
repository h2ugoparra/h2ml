# Contributing to h2ml

## Development setup

```bash
git clone https://github.com/h2ugoparra/h2ml
cd h2ml
uv sync
```

The test suite does not require the `[geo]` optional extras. If you need spatial inference features, install the geo extra (pulls `h2mare` from PyPI):

```bash
uv sync --extra geo
```

## Running tests

```bash
# Full suite
uv run pytest

# Single module
uv run pytest tests/pipeline/

# Single test
uv run pytest tests/pipeline/test_pipeline.py::TestFullRun::test_all_four_steps_completed
```

There are no lint or build steps — the project is pure Python.

## Adding a new model

All models are registered in `h2ml/utils/registry.py`. Add a `ModelEntry` to `CLASSIFIER_REGISTRY` or `REGRESSOR_REGISTRY`:

```python
from sklearn.linear_model import ElasticNet
from h2ml.optimization.params import regressors as rp   # add param_fn here

REGRESSOR_REGISTRY["ElasticNet"] = ModelEntry(
    model_cls        = ElasticNet,
    default_kwargs   = {"random_state": 42},
    requires_scaling = True,
    param_fn         = rp.elasticnet_r_params,   # or None to disable HPO
    opt_enabled      = True,
)
```

Then add the corresponding Optuna parameter function in `h2ml/optimization/params/regressors.py` (or `classifiers.py`):

```python
def elasticnet_r_params(trial) -> dict:
    return {
        "alpha":   trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }
```

The registry is the single source of truth — no other files need changing for the model to be picked up by `build_models()`, the CV engine, and the optimizer.

For optional heavy dependencies (LightGBM, XGBoost, CatBoost), wrap the import in `try/except ImportError` as the existing entries do.

## Adding a y-transform

Transforms live in `h2ml/preprocessing/transforms.py`. Add an entry to both `Y_TRANSFORMS` and `INVERSE_TRANSFORMS`:

```python
Y_TRANSFORMS["cbrt"] = lambda y: np.cbrt(y)

INVERSE_TRANSFORMS["cbrt"] = lambda y: np.power(y, 3)
```

That is all — `build_transform_stores()` reads from `Y_TRANSFORMS` automatically.

If the transform can legitimately return `None` (e.g. winsorize when there are no outliers), handle that in the lambda and document the condition.

## Pull requests

- Open an issue first for non-trivial changes so we can align on the approach.
- Keep PRs focused on a single concern.
- Include or update tests for any changed behaviour. Run `uv run pytest` before pushing.
- The CI matrix tests Python 3.11 and 3.12 — avoid features not available on 3.11.
