# Contributing to h2ml

## Development setup

```bash
git clone https://github.com/h2ugoparra/h2ml
cd h2ml
uv sync
```

`uv sync` installs everything the test suite needs, including the spatial inference path: `h2mare` and `polars` are core dependencies, and `h2mare` pulls in `cartopy`.

There is no `[geo]` extra — it was removed in 0.3.1 because every package it listed was already a core dependency.

## Running tests

```bash
# Full suite
uv run pytest

# Single module
uv run pytest tests/pipeline/

# Single test
uv run pytest tests/pipeline/test_pipeline.py::TestFullRun::test_all_four_steps_completed
```

## Linting

CI runs `ruff` over both `h2ml/` and `tests/`. Run it before pushing:

```bash
uv run ruff check h2ml/ tests/
uv run ruff format h2ml/ tests/
```

CI checks formatting with `ruff format --check`, which fails on unformatted files rather than fixing them — so run `ruff format` locally, not just `ruff check`.

These run in the `quality` job, which is informational by default rather than a required merge gate. Treat a red `quality` check as something to fix regardless.

## Adding a new model

All models are registered in `h2ml/utils/registry.py`. Add a `ModelEntry` to `CLASSIFIER_REGISTRY` or `REGRESSOR_REGISTRY`:

```python
from sklearn.linear_model import ElasticNet
from h2ml.core.param_spaces import regressors as rp   # add param_fn here

REGRESSOR_REGISTRY["ElasticNet"] = ModelEntry(
    model_cls        = ElasticNet,
    default_kwargs   = {"random_state": 42},
    requires_scaling = True,
    param_fn         = rp.elasticnet_r_params,   # or None to disable HPO
    opt_enabled      = True,
)
```

Then add the corresponding Optuna parameter function in `h2ml/core/param_spaces/regressors.py` (or `classifiers.py`):

```python
def elasticnet_r_params(trial) -> dict:
    return {
        "alpha":   trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }
```

The registry is the single source of truth for `build_models()`, the CV engine, and the optimizer — those three need no other changes.

**SHAP routing is the exception.** `_select_explainer` in `h2ml/features/shap_importance.py` falls through to `shap.TreeExplainer` for any model it does not recognise, so a non-tree model that is only added to the registry will fail in step 2. Add its class name to `_GENERIC_EXPLAINER_MODELS` (KernelSHAP via `predict`/`predict_proba`) under the right `TaskType`, or add a dedicated branch if it needs a specific explainer — `_TABPFN_MODELS` routing to `shapiq.TabPFNExplainer` is the worked example.

For optional heavy dependencies (LightGBM, XGBoost, CatBoost), wrap the import in `try/except ImportError` as the existing entries do. Dependencies that are heavy *and* need credentials or change runtime characteristics (TabPFN) go a step further: gate registration on an environment variable so installing the extra never silently changes a run.

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
- Include or update tests for any changed behaviour. Run `uv run pytest` and both `ruff` commands (see [Linting](#linting)) before pushing.
- The CI matrix tests Python 3.11, 3.12 and 3.13 (`tox` uses the same three). The supported floor is 3.11 — avoid features not available there. Note that `.python-version` pins **3.13** locally, so a syntax or stdlib feature that works on your machine can still fail the 3.11 leg; run `uv run tox` to check all three before pushing.
