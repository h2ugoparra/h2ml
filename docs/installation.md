# Installation

## Requirements

- Python 3.11 or later

## Standard install

```bash
uv add h2ml
# or
pip install h2ml
```

## With boosting libraries

LightGBM, XGBoost, and CatBoost are optional. Install the `[boosting]` extra to include them:

```bash
uv add h2ml[boosting]
# or
pip install h2ml[boosting]
```

Without this extra, the pipeline runs all sklearn models only. The registry detects which libraries are available at import time — no configuration required.

## TabPFN (optional)

[TabPFN](https://github.com/PriorLabs/TabPFN) is a pretrained transformer for tabular data. It is
gated twice — an extra to install it, and an environment variable to register it — because unlike
the boosting libraries it pulls in `torch`, needs credentials, and materially changes how long a
run takes.

```bash
uv add "h2ml[tabpfn]"
# or
pip install "h2ml[tabpfn]"

export H2ML_ENABLE_TABPFN=1   # PowerShell: $env:H2ML_ENABLE_TABPFN = "1"
```

Without the environment variable the models are not registered and nothing about a pipeline run
changes, even with the extra installed.

**Requires Python 3.12+.** `shapiq` (used for TabPFN SHAP values) dropped 3.11 support at 1.5, so
the extra resolves to nothing on 3.11. h2ml logs a warning if you set the variable there.

**Authentication.** The first fit opens a browser window to log in via PriorLabs and accept the
licence; the token is then cached. For headless or CI environments, accept the licence at
[ux.priorlabs.ai](https://ux.priorlabs.ai) and set `TABPFN_TOKEN` instead. Do this *before* your
first pipeline run — the first fit otherwise happens inside a worker process, where the prompt is
invisible and the run appears to hang.

**Hardware.** TabPFN picks its device automatically (`device="auto"` → CUDA → mps → CPU), so no
configuration is needed. On CPU it is slow; treat ~5 000 samples as a practical ceiling. Beyond
TabPFN's pretraining limits it raises, and the CV engine reports the model as skipped in
`result.cv_warnings` rather than failing the run.

**Parallelism.** Each CV worker loads its own checkpoint, and `device="auto"` claims *every*
visible CUDA GPU. Lower `PipelineConfig(n_jobs=...)` — or set `CUDA_VISIBLE_DEVICES` — when
running TabPFN, or you can exhaust RAM or VRAM.

**Side effect worth knowing.** `tabpfn` depends on `lightgbm`, so installing this extra also makes
`LGBMClassifier`/`LGBMRegressor` available even without `[boosting]`. The registry picks them up
automatically, which slightly widens the default model set.

## With spatial inference

The geo module (`h2ml.geo.geo_predict`) depends on [h2mare](https://github.com/h2ugoparra/h2mare), which is a **core** dependency — it is installed by the base install, along with the `cartopy` and `polars` it requires. Nothing extra is needed:

```bash
uv add h2ml
# or
pip install h2ml
```

The `[geo]` extra was removed in 0.3.1. It listed `h2mare`, `cartopy` and `polars`, all of which the base install already provides, so it never installed anything beyond the base package. If you have `h2ml[geo]` pinned somewhere, drop the suffix — installers warn about an unknown extra and fall back to the base package, so nothing breaks either way.

## Development setup

```bash
git clone https://github.com/h2ugoparra/h2ml
cd h2ml
uv sync --group dev
uv run pytest        # run the test suite
uv run mkdocs serve  # preview the docs locally
```

## Running tests across Python versions

```bash
uv python install 3.11 3.12 3.13
uv run tox
```
