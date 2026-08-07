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

Note that **`pip`/`uv` install a CPU-only `torch` wheel on Windows**, so an NVIDIA GPU sits unused
even though it is present — h2ml warns about this at registration. To use the GPU, install a CUDA
build explicitly, matching your driver:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

Verify with `python -c "import torch; print(torch.cuda.is_available())"`. This is deliberately not
pinned in `pyproject.toml`, since the right CUDA build depends on your machine.

**Parallelism.** Each CV worker loads its own checkpoint, and `device="auto"` claims *every*
visible CUDA GPU. Lower `PipelineConfig(n_jobs=...)` — or set `CUDA_VISIBLE_DEVICES` — when
running TabPFN, or you can exhaust RAM or VRAM.

**Reproducibility — read this before using TabPFN for anything you intend to publish.** Every
other model in the registry is deterministic local computation, pinned by `uv.lock`. TabPFN is
not: it runs pretrained weights downloaded from a repo PriorLabs controls, and `model_path="auto"`
resolves to whichever checkpoint the installed package treats as current. Two machines, or the
same machine months apart, can produce different step-1 rankings with no change to h2ml or its
lockfile. It also needs `api.priorlabs.ai` reachable to verify your token, so a run is not
reproducible offline the way the rest of the pipeline is.

Treat TabPFN as an **exploration tool** — good for finding out whether a foundation model beats
your tuned baselines — rather than as part of a pipeline whose results you need to reproduce
exactly later. When it participates, h2ml records what it used in
`result.model_provenance` (e.g. `{"TabPFNClassifier": "tabpfn==8.2.0, model_version=v3"}`), which
is persisted with the result and logged as a warning when TabPFN wins. Keep that alongside any
result you report.

**Side effects worth knowing.** `tabpfn` depends on `lightgbm`, so installing this extra also
makes `LGBMClassifier`/`LGBMRegressor` available even without `[boosting]`; the registry picks
them up automatically. Note also that `uv sync --extra tabpfn` *replaces* your environment — if
you want the boosting models too, ask for both:

```bash
uv sync --extra boosting --extra tabpfn
```

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
