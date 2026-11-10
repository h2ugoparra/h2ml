# Spatial Cross-Validation

Standard k-fold CV assumes samples are exchangeable. When data has spatial autocorrelation, nearby samples are more similar than distant ones, so random folds leak information across train/test boundaries and inflate performance estimates.

h2ml activates spatial CV automatically when `store.coords` is provided.

## Activation

```python
store = PipelineData(
    X=X_arr,
    feature_names=cols,
    y=y_arr,
    coords=np.column_stack([lats, lons]),  # (n_samples, 2), lat then lon — activates spatial CV
    times=date_array,                      # (n_samples,) YYYY-MM-DD — activates temporal conformal cells
)
pipeline = H2MLPipeline(PipelineConfig(spatial_cv_metric="haversine"))  # lat/lon in degrees
result = pipeline.run(store)
print(result.cv_type)   # "spatial"
```

The splitter is built **once** in step 1 and reused across all steps and HPO repeats, ensuring consistent spatial structure throughout the pipeline.

---

## Methods

### Block splitter (`spatial_cv_method="block"`)

Quantile-grid blocking: divides the spatial domain into a regular grid of blocks and assigns blocks to folds round-robin. Fast, no clustering.

```python
config = PipelineConfig(
    spatial_cv_method="block",
    n_blocks_per_fold=5,       # blocks per test fold; total = n_splits × n_blocks_per_fold
    spatial_cv_metric="euclidean",  # or "haversine" for lat/lon in degrees
)
```

### SPCV splitter (`spatial_cv_method="spcv"`)

Two-stage method based on [Wang et al. 2023](https://www.sciencedirect.com/science/article/pii/S1569843223001887?via%3Dihub):

1. **AHC blocks** — Agglomerative Hierarchical Clustering on coordinates groups samples into spatially coherent blocks
2. **Cluster ensemble (HBGF)** — three independent KMeans runs (on location, covariates, and target) are combined into a consensus fold assignment via SpectralClustering

Blocks that co-cluster across the three views end up in the same fold, so each test fold is a distinct region of geographic, covariate, and target space. The folds are deliberately *not* representative of the full data: scores estimate extrapolation to unseen conditions and are typically more pessimistic than block or random CV.

```python
config = PipelineConfig(
    spatial_cv_method="spcv",
    ahc_threshold=None,         # auto: 10th percentile of pairwise distances
    pca_components=0.95,        # variance retained by PCA on block covariates
    exact_max_samples=5_000,    # above this, approximate AHC is used
    knn_neighbors=15,           # k for the k-NN connectivity graph
    spcv_use_target=True,       # False: assign folds without the target view
)
```

`spcv_use_target=False` assigns folds from location and covariates alone, so targets measured on the same rows with the same `X` share folds. Folds still differ when features or rows differ. Holdouts and `nested_cv` outer folds never use the target, because they estimate performance.

> **Note:** AHC uses `linkage="ward"` by default. Ward requires Euclidean distances, so when `spatial_cv_metric="haversine"` it is automatically downgraded to `"average"`. For projected coordinates (metres, km) `spatial_cv_metric="euclidean"` keeps ward valid.

The linkage shapes the blocks:

| | Ward (`"euclidean"`) | Average (`"haversine"`) |
|---|---|---|
| Merges | the pair adding the least within-cluster variance | the pair with the smallest mean pairwise distance |
| Blocks | compact and similar in size | follow sampling density, so shapes and sizes vary |
| CV folds | more even, so less noisy scores | can be lopsided, so noisier scores |

On evenly spread points the two give similar blocks. On clumpy sampling, average tends to give a few large blocks and many tiny ones, while ward splits the clumps into balanced regions. If you want ward, project the coordinates and use `"euclidean"`. [Plot the splitter](#checking-the-spatial-clustering) to see which you got.

---

## Parameter reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spatial_cv_method` | `"spcv"` | `"block"` or `"spcv"` |
| `spatial_cv_metric` | `"euclidean"` | `"euclidean"` (projected) or `"haversine"` (lat/lon degrees) |
| `n_blocks_per_fold` | `5` | Blocks per test fold for the block splitter |
| `ahc_threshold` | `None` | AHC distance cut. Auto-set to 10th percentile when `None`. |
| `exact_max_samples` | `5000` | Below this, exact scipy AHC; above, approximate sklearn AHC |
| `knn_neighbors` | `15` | k for the k-NN graph in approximate AHC |
| `pca_components` | `0.95` | Variance retained by PCA on block covariates in SPCV stage 2 |
| `spcv_use_target` | `True` | Include the target view in SPCV fold assignment. Holdouts and `nested_cv` outer folds never use it. |
| `time_bin_resolution` | `"month"` | Temporal granularity for compound conformal cells: `"month"` (1–12) or `"season"` (DJF/MAM/JJA/SON). Only active when `store.times` is provided. |

---

## Choosing a method

Use **block** when:

- Data is large (> 10k samples) — block splitting is O(n) vs O(n²) for AHC
- You want a fast, interpretable split with no tuning
- Spatial distribution is roughly uniform

Use **spcv** when:

- Spatial autocorrelation is the primary concern
- You want a strict, extrapolation-style estimate: test folds that differ from training in covariate and target space
- Dataset is small enough for AHC (or approximate AHC handles the scale)

---

## Checking the spatial clustering

After a run, plot the splitter to see how the rows were grouped and which region each fold holds out:

```python
result = pipeline.run(store)
result.splitter.plot()                               # or save_path="figures/spatial_folds.png"

from h2ml.plots import plot_spatial_blocks           # the same plot as a function
plot_spatial_blocks(result.splitter, save_path="figures/spatial_folds.png")
```

The figure has two panels:

- **Spatial blocks** (left) — each point coloured by its block: quantile-grid cells for `"block"`, AHC clusters for `"spcv"`.
- **Fold assignments** (right) — each point coloured by the fold whose test set holds it.

What to look for:

- **Compact blocks.** Each block should be one contiguous patch. Blocks that interleave mean nearby rows can still fall on both sides of a split.
- **Block size against the autocorrelation range.** Blocks much smaller than the residuals' [variogram](guides/conformal.md#variogram-diagnostic) range leave neighbours correlated across folds, so scores stay optimistic. With `"spcv"`, raise `ahc_threshold` for fewer, larger clusters; with `"block"`, lower `n_blocks_per_fold`.
- **Fold layout.** Block folds are scattered tiles across the whole area. SPCV folds group blocks that are alike in location, covariates and (with `spcv_use_target`) the target, so each fold is a distinct region of that combined space. A fold confined to one corner tests extrapolation into it.
- **Empty areas.** With a holdout, the splitter covers the dev rows only, so held-out blocks show as gaps.

`lon_col` / `lat_col` (default `1` / `0`) pick the coordinate columns for the x and y axes and assume `[lat, lon]` coords; pass `lon_col=0, lat_col=1` for `[lon, lat]` or projected `[x, y]`. With many AHC clusters, colours of neighbouring block IDs are similar and the colour bar is crowded, so judge compactness by eye rather than by label.

`result.splitter` is saved with the result. Results saved by h2ml 0.4.1 or earlier load without it; see [Visualization](guides/visualization.md#spatial-blocks-and-folds) to rebuild it.

---

## Spatial blocks and local conformal calibration

The same `block_id_` assignments built for CV are reused by `build_final_model()` to partition OOF residuals into a `LocalConformalCalibration`. Each block (and optionally each block × time bin) gets its own nonconformity score distribution, so prediction intervals widen in regions where the model is historically less accurate.

- **Block splitter** creates a predictable `n_splits × n_blocks_per_fold` blocks with roughly uniform sample counts — compound cells are well-populated for typical dataset sizes.
- **SPCV splitter** creates more, smaller AHC clusters. With many blocks, compound cells can be sparse — prefer `time_bin_resolution="season"` (4 bins) over `"month"` (12 bins), or increase `ahc_threshold` to reduce the block count.

See [Conformal Prediction](guides/conformal.md#spatio-temporal-local-calibration) for diagnostics and tuning.

---

## Spatial autocorrelation diagnostics

After a pipeline run, check for residual spatial autocorrelation using the variogram fitted on the out-of-fold residuals:

```python
from h2ml.utils.variogram import plot_variogram

final = result.build_final_model()
vr = final.variogram
print(vr.practical_range, vr.status)

residuals = result.oof_predictions - result.oof_labels
plot_variogram(result.features.coords, residuals, metric=result.spatial_cv_metric)
```

The range is in coordinate units, or km with `spatial_cv_metric="haversine"`. With status `"ok"`, a short range relative to your block size means the spatial CV is well-sized, and a long range suggests the blocks are too small to achieve true spatial independence. `"at_bound"` means the range is at least the value shown, and `"no_structure"` means none was found, so ignore the range. See the [variogram diagnostic](guides/conformal.md#variogram-diagnostic).
