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
