# Spatial Cross-Validation

Standard k-fold CV assumes samples are exchangeable. When data has spatial autocorrelation, nearby samples are more similar than distant ones, so random folds leak information across train/test boundaries and inflate performance estimates.

h2ml activates spatial CV automatically when `store.coords` is provided.

## Activation

```python
store = PipelineData(
    X=X_arr,
    feature_names=cols,
    y=y_arr,
    coords=np.column_stack([lats, lons]),  # (n_samples, 2)
)
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

Produces folds that are geographically separated *and* representative of the covariate and label space.

```python
config = PipelineConfig(
    spatial_cv_method="spcv",
    ahc_threshold=None,         # auto: 10th percentile of pairwise distances
    pca_components=0.95,        # variance retained by PCA on block covariates
    exact_max_samples=5_000,    # above this, approximate AHC is used
    knn_neighbors=15,           # k for the k-NN connectivity graph
)
```

> **Note:** AHC uses `linkage="ward"` by default. Ward requires Euclidean distances, so when `spatial_cv_metric="haversine"` it is automatically downgraded to `"average"`. For projected coordinates (metres, km) `spatial_cv_metric="euclidean"` keeps ward valid.

---

## Parameter reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spatial_cv_method` | `"block"` | `"block"` or `"spcv"` |
| `spatial_cv_metric` | `"euclidean"` | `"euclidean"` (projected) or `"haversine"` (lat/lon degrees) |
| `n_blocks_per_fold` | `5` | Blocks per test fold for the block splitter |
| `ahc_threshold` | `None` | AHC distance cut. Auto-set to 10th percentile when `None`. |
| `exact_max_samples` | `5000` | Below this, exact scipy AHC; above, approximate sklearn AHC |
| `knn_neighbors` | `15` | k for the k-NN graph in approximate AHC |
| `pca_components` | `0.95` | Variance retained by PCA on block covariates in SPCV stage 2 |

---

## Choosing a method

Use **block** when:

- Data is large (> 10k samples) — block splitting is O(n) vs O(n²) for AHC
- You want a fast, interpretable split with no tuning
- Spatial distribution is roughly uniform

Use **spcv** when:

- Spatial autocorrelation is the primary concern
- You have covariate and target structure you want folds to respect
- Dataset is small enough for AHC (or approximate AHC handles the scale)

---

## Spatial autocorrelation diagnostics

After a pipeline run, check for residual spatial autocorrelation using the variogram utility:

```python
from h2ml.utils.variogram import autocorrelation_range, plot_variogram

final = result.build_final_model()
y_pred = final.predict(store.X)
residuals = store.y - y_pred

vr = autocorrelation_range(store.coords, residuals)
plot_variogram(vr)
print(vr.range_km)  # estimated autocorrelation range
```

A short range relative to your block size means the spatial CV is well-sized. A long range suggests the blocks are too small to achieve true spatial independence.
