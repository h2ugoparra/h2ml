# Installation

## Requirements

- Python 3.11 or later (tested on 3.11–3.13)

## Standard install

```bash
uv add h2ml
# or
pip install h2ml
```

## With boosting libraries

LightGBM, XGBoost, and CatBoost are optional. Install the `[boosting]` extra to include them:

```bash
uv add "h2ml[boosting]"
# or
pip install "h2ml[boosting]"
```

Without this extra, the pipeline runs the sklearn models only. The registry detects each library at import time, so any one of them installed on its own is picked up too — no configuration required.

## Spatial inference

The geo module (`h2ml.geo.geo_predict`) needs nothing extra. Its dependency, [h2mare](https://github.com/h2ugoparra/h2mare), comes with the standard install, along with `cartopy` and `polars`.

!!! note "`[geo]` extra removed"
    The `[geo]` extra was removed in 0.3.1 because the standard install already provides everything it listed. If you have `h2ml[geo]` pinned somewhere, drop the suffix.

## Development setup

```bash
git clone https://github.com/h2ugoparra/h2ml
cd h2ml
uv sync --group dev --extra boosting
uv run pytest                    # run the test suite
uv run ruff check h2ml/ tests/   # lint
uv run ruff format h2ml/ tests/  # format
uv run mkdocs serve              # preview the docs locally
uv run mkdocs build --strict     # build the docs as CI does
```

## Running tests across Python versions

```bash
uv python install 3.11 3.12 3.13
uv run tox
```
