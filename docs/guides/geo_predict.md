# Geo Prediction

The `h2ml.geo` module bridges trained `FinalModel`s and hive-partitioned parquet feature stores (via `h2mare.ParquetIndexer`) to generate spatial predictions at scale.

Requires the `[geo]` optional dependencies:

```bash
uv pip install h2ml[geo]
```

## predict_for_year

Generate predictions for a full calendar year across one or more targets.

```python
from pathlib import Path
from h2ml.geo.geo_predict import predict_for_year

df = predict_for_year(
    target=["sparrow", "finch"],
    year=2023,
    root_dir=Path("project/"),
    input_parquet_dir=Path("data/features/"),
    schema="v1",
    geo_extent=(-10.0, 35.0, 30.0, 70.0),  # xmin, ymin, xmax, ymax
)
```

Models are loaded from `root_dir/models/{target}_{schema}_final-model.pkl`. Targets whose model file is missing or whose prediction raises an exception are skipped with a warning — the rest still complete.

Output columns: `['index', 'time', 'lon', 'lat', '{target}_{schema}', ...]`

Rows with any null feature value appear as `null` in the prediction columns, preserving alignment with the original spatial grid.

### Conformal interval columns

Pass `alpha` to add conformal bound columns for calibrated models. Uncalibrated models and multiclass classifiers always return point predictions only.

```python
df = predict_for_year(
    target="sparrow",
    year=2023,
    root_dir=Path("project/"),
    input_parquet_dir=Path("data/features/"),
    schema="v1",
    geo_extent=(-10.0, 35.0, 30.0, 70.0),
    alpha=0.10,  # 90% coverage
)
# columns: sparrow_v1, sparrow_v1_pi_lower, sparrow_v1_pi_upper
```

The meaning of the bound columns differs by task type:

**Regression** — per-sample outcome bounds in the original (inverse-transformed) count scale:

```
pi_lower = max(0, ŷ − q)
pi_upper = ŷ + q
```

These carry a formal conformal coverage guarantee: the true value falls inside `[pi_lower, pi_upper]` with probability ≥ 1 − alpha.

**Binary classification** — per-sample calibration bands in probability space:

```
pi_lower = clip(p − q, 0, 1)
pi_upper = clip(p + q, 0, 1)
```

Where `p` is the predicted positive-class probability and `q` is the conformal threshold. Because nonconformity scores for binary classification are `1 − p(true class)` — which live on [0, 1] — `q` and `p` share the same units and the formula is directly analogous to regression. The bands vary spatially: pixels with `p` near 0 or 1 (confident) produce narrow intervals; pixels near 0.5 (uncertain) produce wide intervals spanning the decision boundary.

!!! note
    Unlike regression, these bounds are not a formal coverage guarantee on the class label — they are a calibration band expressing how much the predicted probability could plausibly shift given past model errors.

## predict_for_year_delta

Same as `predict_for_year` but loads `DeltaFinalModel`s (two-component presence × abundance models). Model directories are expected at `root_dir/models/{target}_{schema}_final-model/`.

```python
from h2ml.geo.geo_predict import predict_for_year_delta

df = predict_for_year_delta(
    target=["sparrow", "finch"],
    year=2023,
    root_dir=Path("project/"),
    input_parquet_dir=Path("data/features/"),
    schema="v1",
    geo_extent=(-10.0, 35.0, 30.0, 70.0),
    alpha=0.10,
)
# columns per target: {target}_{schema}, {target}_{schema}_pi_lower, {target}_{schema}_pi_upper
```

The delta prediction is `P(present) × E(count | present)`. The conformal intervals are calibrated on the full combined delta output (not per component) and are always in the original count scale. Targets without a saved `ConformalCalibration` produce point predictions only.

## predict_map

Quickly visualise model predictions aggregated over a spatial-temporal grid without saving results to disk. Useful for exploratory inspection of a single model.

```python
from h2mare.storage import ParquetIndexer
from h2ml.geo.geo_predict import predict_map

indexer = ParquetIndexer(Path("data/features/"))

predict_map(
    model=final,                          # FinalModel
    indexer=indexer,
    dates=("2023-01-01", "2023-12-31"),
    bbox=(-10.0, 35.0, 30.0, 70.0),      # xmin, ymin, xmax, ymax
    target_col="sparrow_v1",
    agg_by="month",                       # "month" or "season"
)
```

Set `save_path` to write the figure to disk instead of calling `plt.show()`:

```python
predict_map(..., save_path="figures/sparrow_2023.png")
```

`vminmax` clips the colormap to a fixed range, useful when comparing maps across years:

```python
predict_map(..., vminmax=(0.0, 50.0))
```

### Differences from `predict_for_year`

| | `predict_for_year` | `predict_map` |
|---|---|---|
| Output | Polars DataFrame | Plot only (`None`) |
| Targets | Multiple | Single model |
| Conformal intervals | Yes (with `alpha`) | No |
| Temporal aggregation | None (raw time column) | `"month"` or `"season"` |
| Typical use | Production pipeline | Exploratory / QA |
