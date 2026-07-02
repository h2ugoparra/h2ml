"""
h2ml/features/spatial_cv.py

Two sklearn-compatible spatial cross-validation splitters:

SpatialBlockSplitter — quantile-grid blocks, fast, geography-only.

SPCVSplitter — two-stage method:
  Stage 1: Agglomerative Hierarchical Clustering (AHC) on coordinates
           produces spatially coherent blocks that respect autocorrelation.
  Stage 2: Cluster Ensembles (CE) with HBGF consensus assigns blocks to
           folds by balancing location, covariates, and target variable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
from loguru import logger
from sklearn.model_selection import KFold, StratifiedKFold

from h2ml.core.base import TaskType
from h2ml.core.spatial_config import SpatialCVConfig

SpatialMetric = Literal["euclidean", "haversine"]
LinkageCriterion = Literal["ward", "complete", "average", "single"]


def _warn_if_degrees(coords: np.ndarray, metric: SpatialMetric) -> None:
    """Warn when coords look like lat/lon degrees but metric is Euclidean."""
    if metric != "euclidean":
        return
    lat, lon = coords[:, 0], coords[:, 1]
    looks_like_degrees = -90 <= lat.min() and lat.max() <= 90 and -180 <= lon.min() and lon.max() <= 180
    span = max(lat.max() - lat.min(), lon.max() - lon.min())
    if looks_like_degrees and span > 5:
        logger.warning(
            f"Coords appear to be in degrees (span={span:.1f}°) but metric='euclidean'. "
            "Euclidean distance distorts large geographic areas — consider "
            "metric='haversine' (for lat/lon inputs) or reprojecting to km first."
        )


class SpatialBlockSplitter:
    """
    K-fold splitter that groups spatially adjacent samples into blocks
    and holds out entire blocks as test sets.

    Blocks are formed by quantile-binning coordinates on each axis
    (robust to uneven spatial distributions) and assigned to folds
    round-robin. Each fold's test set is one or more contiguous blocks.

    Args:
        coords:           (n_samples, 2) array of spatial coordinates
                          (lat/lon, x/y, or any 2-D position).
        n_splits:         Number of CV folds.
        n_blocks_per_fold: Number of blocks per test fold. Total blocks =
                           n_splits × n_blocks_per_fold. More blocks give
                           finer spatial granularity at the cost of smaller
                           individual test sets.
        random_state:     Seed for the block→fold shuffle (reproducible folds).
        metric:           "euclidean" (projected coords) or "haversine"
                          (lat/lon in decimal degrees).

    Example:
        >>> coords = np.column_stack([lats, lons])
        >>> splitter = SpatialBlockSplitter(coords, n_splits=5, n_blocks_per_fold=5)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     model.fit(X[train_idx], y[train_idx])
        ...     score = model.score(X[test_idx], y[test_idx])
    """

    def __init__(
        self,
        coords: np.ndarray,
        n_splits: int = 5,
        n_blocks_per_fold: int = 5,
        random_state: int = 42,
        metric: SpatialMetric = "euclidean",
    ) -> None:
        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be shape (n_samples, 2), got {coords.shape}.")
        n_samples = coords.shape[0]
        if n_splits * n_blocks_per_fold > n_samples:
            raise ValueError(
                f"n_splits ({n_splits}) × n_blocks_per_fold ({n_blocks_per_fold}) = "
                f"{n_splits * n_blocks_per_fold} exceeds n_samples ({n_samples}). "
                "Reduce n_splits or n_blocks_per_fold."
            )
        _warn_if_degrees(coords, metric)
        self.coords = coords
        self.n_splits = n_splits
        self.n_blocks_per_fold = n_blocks_per_fold
        self.random_state = random_state
        self.metric: SpatialMetric = metric
        self._fold_of_sample = self._assign_blocks()

    # ------------------------------------------------------------------
    # Sklearn splitter interface
    # ------------------------------------------------------------------

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of folds (sklearn splitter interface; X/y/groups ignored)."""
        return self.n_splits

    def plot(self, lon_col: int = 1, lat_col: int = 0, save_path: Optional[Path] = None) -> None:
        """Two-panel scatter of block and fold assignments.

        Thin wrapper over h2ml.plots.plot_spatial_blocks(self); see that function for details.
        """
        from h2ml.plots.plots import plot_spatial_blocks

        plot_spatial_blocks(self, lon_col=lon_col, lat_col=lat_col, save_path=save_path)

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, test_idx) for each of n_splits folds.

        Args:
            X:      Array-like whose row count must match the calibration coords.
            y:      Ignored (sklearn signature compatibility).
            groups: Ignored (sklearn signature compatibility).

        Yields:
            (train_idx, test_idx) integer index arrays per fold.

        Raises:
            ValueError: If len(X) does not match the number of stored coords.
        """
        n_samples = len(X)
        if n_samples != len(self.coords):
            raise ValueError(f"X has {n_samples} rows but coords has {len(self.coords)} rows.")
        all_idx = np.arange(n_samples)
        for k in range(self.n_splits):
            test_idx = all_idx[self._fold_of_sample == k]
            train_idx = all_idx[self._fold_of_sample != k]
            if test_idx.size == 0:
                logger.warning(
                    f"SpatialBlockSplitter: fold {k} has zero test samples. "
                    "Consider reducing n_blocks_per_fold or n_splits."
                )
            yield train_idx, test_idx

    # ------------------------------------------------------------------
    # Block assignment
    # ------------------------------------------------------------------

    def clone_with_n_splits(self, n_splits: int, random_state: Optional[int] = None) -> "SpatialBlockSplitter":
        """Return a new splitter with n_splits folds, reusing the same coords and metric."""
        return SpatialBlockSplitter(
            coords=self.coords,
            n_splits=n_splits,
            n_blocks_per_fold=self.n_blocks_per_fold,
            random_state=random_state if random_state is not None else self.random_state,
            metric=self.metric,
        )

    def _assign_blocks(self) -> np.ndarray:
        """
        Assign each sample to a fold via quantile-grid block assignment.

        Returns:
            fold_of_sample: (n_samples,) int array where entry i is the
                            fold index [0, n_splits) for sample i.
        """
        n_target_blocks = self.n_splits * self.n_blocks_per_fold
        # Number of bins per axis: smallest q such that q² ≥ n_target_blocks
        q = math.ceil(math.sqrt(n_target_blocks))

        coords = self.coords
        row_bins = self._quantile_bin(coords[:, 0], q)
        col_bins = self._quantile_bin(coords[:, 1], q)

        # Encode (row_bin, col_bin) → block_id
        raw_block_id = row_bins * q + col_bins

        # Compact-re-index to remove gaps from empty grid cells
        unique_blocks, compact_id = np.unique(raw_block_id, return_inverse=True)
        n_blocks = len(unique_blocks)

        # Shuffle block order before round-robin so folds are spatially dispersed
        # rather than forming longitude strips (an artefact of sorting by block ID).
        rng = np.random.default_rng(self.random_state)
        block_order = rng.permutation(n_blocks)
        block_to_fold = np.empty(n_blocks, dtype=np.int32)
        block_to_fold[block_order] = np.arange(n_blocks) % self.n_splits
        fold_of_sample = block_to_fold[compact_id]

        # Expose block-level assignments as a public attribute so callers
        # can distinguish multiple blocks within the same fold (e.g. for plotting).
        self.block_id_ = compact_id.astype(np.int32)

        return fold_of_sample.astype(np.int32)

    @staticmethod
    def _quantile_bin(values: np.ndarray, q: int) -> np.ndarray:
        """
        Bin 1-D values into q equal-count bins using quantile breakpoints.
        Returns integer bin indices in [0, q-1].
        """
        breakpoints = np.quantile(values, np.linspace(0, 1, q + 1))
        # np.searchsorted places duplicates into the rightmost matching bin;
        # clip to [0, q-1] so the last breakpoint doesn't produce bin q.
        bins = np.searchsorted(breakpoints[1:-1], values, side="right")
        return bins.astype(np.int32)


class SPCVSplitter:
    """
    Two-stage spatial cross-validation splitter (SP-CV).
    Reference: Wang et al. 2023. Spatial+: A new cross-validation method to evaluate geospatial machine learning models.

    Stage 1 — AHC blocks:
        Agglomerative Hierarchical Clustering (AHC) on spatial coordinates forms
        blocks that respect the autocorrelation structure of the data.
        The AHC distance threshold controls block granularity; if not
        provided it defaults to the 10th percentile of pairwise distances.

    Stage 2 — Cluster Ensemble fold assignment:
        Each block is represented by its mean coordinates, covariates, and
        target value. Three independent KMeans runs (on location, covariates,
        and target) are combined via Hybrid Bipartite Graph Formulation (HBGF):
        a co-occurrence affinity matrix is built from the stacked membership
        matrices and SpectralClustering produces the final k folds.

    The result is folds that are geographically separated *and* representative
    of the full covariate and label space.

    Args:
        coords:       (n_samples, 2) spatial coordinates (lat/lon or projected).
        X:            (n_samples, n_features) covariate matrix.
        y:            (n_samples,) target array.
        n_splits:     Number of CV folds (k).
        threshold:    AHC distance threshold. Defaults to the 10th percentile
                      of pairwise coordinate distances when None.
        linkage:      Linkage criterion for AHC ('ward', 'average', 'complete').
                      'ward' minimises within-cluster variance and works well
                      for compact geographic blocks.
        random_state: Seed for KMeans and SpectralClustering reproducibility.
        metric:       "euclidean" (projected coords) or "haversine" (lat/lon in
                      decimal degrees). See the Note on linkage interaction.
        pca_components: Variance fraction retained by PCA applied to block
                      covariates before clustering. Default 0.95.
        exact_max_samples: Sample count below which exact scipy AHC is used; above
                      it an approximate sklearn AHC (k-NN graph) is used. Default 5000.
        knn_neighbors: k for the k-NN connectivity graph in approximate AHC.
                      Default 15.

    Note:
        'ward' linkage operates on Euclidean distances. When metric='haversine',
        ward is silently downgraded to 'average' (in both the exact and approximate
        AHC paths) because ward is undefined for non-Euclidean distances. For
        projected coords (km, m) use metric='euclidean' and ward linkage is valid.

    Example:
        >>> splitter = SPCVSplitter(coords, X, y, n_splits=5)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     model.fit(X[train_idx], y[train_idx])
        ...     score = model.score(X[test_idx], y[test_idx])
    """

    def __init__(
        self,
        coords: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        threshold: Optional[float] = None,
        linkage: LinkageCriterion = "ward",
        random_state: int = 42,
        metric: SpatialMetric = "euclidean",
        pca_components: float = 0.95,
        exact_max_samples: int = 5_000,
        knn_neighbors: int = 15,
    ) -> None:
        coords = np.asarray(coords, dtype=float)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be (n_samples, 2), got {coords.shape}.")
        n_samples = coords.shape[0]
        if X.shape[0] != n_samples:
            raise ValueError(f"X has {X.shape[0]} rows but coords has {n_samples} rows.")
        if y.shape[0] != n_samples:
            raise ValueError(f"y has {y.shape[0]} rows but coords has {n_samples} rows.")
        if n_samples < n_splits:
            raise ValueError(f"n_samples ({n_samples}) must be >= n_splits ({n_splits}).")

        _warn_if_degrees(coords, metric)
        self.coords = coords
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.threshold = threshold
        self.linkage: LinkageCriterion = linkage
        self.random_state = random_state
        self.metric: SpatialMetric = metric
        self.pca_components = pca_components
        self.exact_max_samples = exact_max_samples
        self.knn_neighbors = knn_neighbors

        block_labels = self._stage1_blocks()
        self._fold_of_sample = self._stage2_folds(block_labels)

    # ------------------------------------------------------------------
    # Sklearn splitter interface
    # ------------------------------------------------------------------

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of folds (sklearn splitter interface; X/y/groups ignored)."""
        return self.n_splits

    def plot(self, lon_col: int = 1, lat_col: int = 0, save_path: Optional[Path] = None) -> None:
        """Two-panel scatter of block and fold assignments.

        Thin wrapper over h2ml.plots.plot_spatial_blocks(self); see that function for details.
        """
        from h2ml.plots.plots import plot_spatial_blocks

        plot_spatial_blocks(self, lon_col=lon_col, lat_col=lat_col, save_path=save_path)

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, test_idx) for each of n_splits folds.

        Args:
            X:      Array-like whose row count must match the calibration coords.
            y:      Ignored (sklearn signature compatibility).
            groups: Ignored (sklearn signature compatibility).

        Yields:
            (train_idx, test_idx) integer index arrays per fold.

        Raises:
            ValueError: If len(X) does not match the number of stored coords.
        """
        n_samples = len(X)
        if n_samples != len(self.coords):
            raise ValueError(f"X has {n_samples} rows but coords has {len(self.coords)} rows.")
        all_idx = np.arange(n_samples)
        for k in range(self.n_splits):
            test_idx = all_idx[self._fold_of_sample == k]
            train_idx = all_idx[self._fold_of_sample != k]
            if test_idx.size == 0:
                logger.warning(
                    f"SPCVSplitter: fold {k} has zero test samples. Consider a smaller threshold or fewer n_splits."
                )
            yield train_idx, test_idx

    def clone_with_n_splits(self, n_splits: int, random_state: Optional[int] = None) -> "SPCVSplitter":
        """
        Return a new splitter with n_splits folds, reusing the AHC block structure.

        Stage 1 (AHC clustering) is skipped — block_id_ is copied directly.
        Stage 2 (fold assignment) is rerun with the new n_splits and random_state,
        so each HPO repeat gets different fold boundaries without repeating AHC.
        """
        clone = object.__new__(SPCVSplitter)
        clone.coords = self.coords
        clone.X = self.X
        clone.y = self.y
        clone.n_splits = n_splits
        clone.threshold = self.threshold
        clone.linkage = self.linkage
        clone.random_state = random_state if random_state is not None else self.random_state
        clone.metric = self.metric
        clone.pca_components = self.pca_components
        clone.exact_max_samples = self.exact_max_samples
        clone.knn_neighbors = self.knn_neighbors
        clone.block_id_ = self.block_id_
        clone._fold_of_sample = clone._stage2_folds(self.block_id_)
        return clone

    # ------------------------------------------------------------------
    # Stage 1 — AHC → blocks
    # ------------------------------------------------------------------

    def _stage1_blocks(self) -> np.ndarray:
        """
        Cluster samples into spatially coherent blocks via AHC.

        For n_samples <= exact_max_samples uses scipy pdist + linkage (exact,
        O(n²) memory). For larger datasets auto-switches to sklearn
        AgglomerativeClustering with a k-NN connectivity graph (O(n × k)
        memory) and logs a warning.

        Returns:
            block_labels: (n_samples,) int array of 0-based block IDs.
        """
        n_samples = self.coords.shape[0]
        if n_samples <= self.exact_max_samples:
            labels = self._ahc_exact()
        else:
            logger.warning(
                f"SPCVSplitter: n_samples={n_samples} > exact_max_samples={self.exact_max_samples}. "
                f"Switching to approximate AHC with k-NN connectivity graph "
                f"(knn_neighbors={self.knn_neighbors}) to avoid O(n²) memory. "
                "Block boundaries may differ slightly from exact AHC."
            )
            labels = self._ahc_approximate()

        _, compact = np.unique(labels, return_inverse=True)
        n_blocks = int(compact.max()) + 1

        if n_blocks < self.n_splits:
            raise ValueError(
                f"AHC produced {n_blocks} block(s) but n_splits={self.n_splits}. "
                "Decrease the threshold to form more blocks."
            )

        self.block_id_ = compact.astype(np.int32)
        return compact.astype(np.int32)

    def _ahc_exact(self) -> np.ndarray:
        """
        Exact AHC via scipy linkage — O(n²) memory, suitable for small n.

        Haversine distances are computed with sklearn pairwise_distances (scipy pdist
        does not support haversine) and converted to a condensed form via squareform.
        ward linkage falls back to average for haversine, matching the approximate path.
        """
        from scipy.cluster.hierarchy import linkage as ahc_linkage, fcluster

        if self.metric == "haversine":
            from sklearn.metrics import pairwise_distances
            from scipy.spatial.distance import squareform

            coords_rad = np.deg2rad(self.coords)
            D = pairwise_distances(coords_rad, metric="haversine")
            distances = squareform(D, checks=False)
        else:
            from scipy.spatial.distance import pdist

            distances = pdist(self.coords, metric=self.metric)  # type: ignore[call-overload]

        linkage = self.linkage
        if self.metric == "haversine" and linkage == "ward":
            linkage = "average"
            logger.debug(
                "SPCVSplitter: linkage='ward' is incompatible with haversine — using 'average' for the exact AHC path."
            )

        threshold = self._resolve_threshold(distances)
        Z = ahc_linkage(distances, method=linkage)
        return fcluster(Z, t=threshold, criterion="distance")  # 1-based

    def _ahc_approximate(self) -> np.ndarray:
        """
        Approximate AHC via sklearn AgglomerativeClustering with a sparse
        k-NN connectivity graph — O(n × knn_neighbors) memory.

        ward linkage requires Euclidean distances; for haversine the coords are
        converted to radians so the k-NN graph distances remain consistent.
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph

        if self.metric == "haversine":
            coords_for_graph = np.deg2rad(self.coords)
            sklearn_metric = "haversine"
        else:
            coords_for_graph = self.coords
            sklearn_metric = "euclidean"

        # ward linkage only supports euclidean; fall back to average for haversine
        linkage = self.linkage
        if self.metric == "haversine" and linkage == "ward":
            linkage = "average"
            logger.debug(
                "SPCVSplitter: linkage='ward' is incompatible with haversine — "
                "using 'average' for the approximate AHC path."
            )

        k = min(self.knn_neighbors, coords_for_graph.shape[0] - 1)
        connectivity = kneighbors_graph(
            coords_for_graph,
            n_neighbors=k,
            metric=sklearn_metric,
            mode="connectivity",
            include_self=False,
        )

        threshold = self._resolve_threshold_approximate(coords_for_graph, sklearn_metric)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            connectivity=connectivity,  # type: ignore[arg-type]
            linkage=linkage,
            metric="euclidean" if linkage == "ward" else sklearn_metric,
        )
        clustering.fit(coords_for_graph)
        return clustering.labels_ + 1  # 1-based to match scipy fcluster

    def _resolve_threshold(self, distances: np.ndarray) -> float:
        """Return threshold in the same units as the condensed distance array."""
        if self.threshold is None:
            t = float(np.percentile(distances, 10))
            units = "rad" if self.metric == "haversine" else "coord units"
            logger.debug(f"SPCVSplitter: threshold auto-set to {t:.4f} {units} (10th pct of pairwise distances).")
            return t
        if self.metric == "haversine":
            return float(np.deg2rad(self.threshold))
        return float(self.threshold)

    def _resolve_threshold_approximate(self, coords_for_graph: np.ndarray, sklearn_metric: SpatialMetric) -> float:
        """
        Return threshold for sklearn AgglomerativeClustering.
        When threshold=None, estimate from a sample of pairwise distances to
        avoid materialising the full O(n²) matrix.
        """
        from scipy.spatial.distance import cdist

        if self.threshold is not None:
            if self.metric == "haversine":
                return float(np.deg2rad(self.threshold))
            return float(self.threshold)

        # Estimate 10th percentile from a random subsample of 1,000 points
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(
            coords_for_graph.shape[0],
            size=min(1_000, coords_for_graph.shape[0]),
            replace=False,
        )
        sub = coords_for_graph[idx]
        dists = cdist(sub, sub, metric=sklearn_metric)  # type: ignore[call-overload]
        upper = dists[np.triu_indices_from(dists, k=1)]
        t = float(np.percentile(upper, 10))
        units = "rad" if self.metric == "haversine" else "coord units"
        logger.debug(f"SPCVSplitter: threshold auto-set to {t:.4f} {units} (10th pct, subsample of 1000).")
        return t

    # ------------------------------------------------------------------
    # Stage 2 — CE (HBGF) → folds
    # ------------------------------------------------------------------

    def _stage2_folds(self, block_labels: np.ndarray) -> np.ndarray:
        """
        Assign blocks to folds via Cluster Ensemble with HBGF consensus.

        Steps:
          1. Compute per-block mean of coords, X, and y.
          2. Run KMeans(k) on each feature set → three cluster assignments.
          3. Stack one-hot memberships → co-occurrence affinity matrix.
          4. SpectralClustering on the affinity → k fold labels per block.
          5. Map back to samples.

        Returns:
            fold_of_sample: (n_samples,) int array of fold indices [0, n_splits).
        """
        from sklearn.cluster import KMeans, SpectralClustering

        n_blocks = int(block_labels.max()) + 1
        k = self.n_splits
        rs = self.random_state

        # --- per-block representative values ---
        # Per-block means via bincount-with-weights: it accumulates each block's
        # samples in ascending index order, identical to arr[mask].mean(), but in a
        # single C pass instead of one boolean-mask reduction per block. block_labels
        # is compacted (contiguous 0..n_blocks-1) upstream, so every block count ≥ 1.
        counts = np.bincount(block_labels, minlength=n_blocks)

        def _block_mean(col: np.ndarray) -> np.ndarray:
            return np.bincount(block_labels, weights=col, minlength=n_blocks) / counts

        block_coords = np.column_stack([_block_mean(self.coords[:, j]) for j in range(2)])
        block_X = np.column_stack([_block_mean(self.X[:, j]) for j in range(self.X.shape[1])])
        block_y = _block_mean(self.y)[:, None]

        # --- PCA on block covariates to remove redundancy before clustering ---
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        if block_X.shape[1] > 1:
            block_X_scaled = StandardScaler().fit_transform(block_X)
            # pca_components is a variance fraction in (0, 1); PCA picks the number of
            # components needed to retain it, bounded internally by min(n_blocks,
            # n_features). (The previous min() against the integer dims was a no-op — a
            # fraction < 1 always wins.)
            block_X_proj = PCA(n_components=self.pca_components, random_state=rs).fit_transform(block_X_scaled)
            logger.debug(
                f"SPCVSplitter stage 2: PCA reduced {block_X.shape[1]} covariates → "
                f"{block_X_proj.shape[1]} components ({self.pca_components} variance threshold)."
            )
        else:
            block_X_proj = block_X

        # --- three independent KMeans clusterings ---
        def _km(data: np.ndarray) -> np.ndarray:
            return KMeans(n_clusters=k, random_state=rs, n_init=10).fit_predict(data)

        labels_L = _km(block_coords)
        labels_C = _km(block_X_proj)
        labels_T = _km(block_y)

        # --- HBGF: build co-occurrence affinity ---
        eye = np.eye(k)
        H = np.hstack([eye[labels_L], eye[labels_C], eye[labels_T]])  # (n_blocks, 3k)
        S = H @ H.T  # (n_blocks, n_blocks)

        # Regularise: two blocks that share no KMeans cluster in any view have
        # S[i,j]=0 and form isolated components in the spectral graph, causing
        # SpectralClustering to warn and assign disconnected blocks arbitrarily.
        # A uniform floor of 1/(n_blocks+1) is << the co-occurrence signal
        # (~3/k per matching view) but guarantees full graph connectivity.
        if n_blocks < 2 * k:
            logger.warning(
                f"SPCVSplitter stage 2: only {n_blocks} AHC blocks for {k} folds. "
                "HBGF affinity may be nearly disconnected — fold assignments could be "
                "suboptimal. Lower the AHC threshold to produce more blocks."
            )
        S += 1.0 / (n_blocks + 1)
        S /= S.max()  # normalise to [0, 1]

        # --- spectral clustering on affinity ---
        fold_per_block = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=rs,
            assign_labels="kmeans",
        ).fit_predict(S)

        return fold_per_block[block_labels].astype(np.int32)


# ---------------------------------------------------------------------------
# Splitter factory — shared by the CV engine and the optimizer
# ---------------------------------------------------------------------------


def build_splitter(
    task_type: TaskType | str,
    n_splits: int,
    random_state: int,
    shuffle: bool = True,
    coords: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    spatial: Optional[SpatialCVConfig] = None,
):
    """
    Build the CV splitter: spatial when coords are provided, otherwise
    stratified/standard KFold.

    Single construction point shared by CrossValidator and the Optuna
    objective, so splitter selection and parameter forwarding cannot drift
    between the two.

    Args:
        task_type:    TaskType or its string value; selects StratifiedKFold
                      (classification) vs KFold (regression) in the non-spatial case.
        n_splits:     Number of CV folds.
        random_state: Seed for fold shuffling / block assignment.
        shuffle:      Shuffle before splitting in the non-spatial case.
        coords:       (n_samples, 2) coordinates. When None, a random
                      (non-spatial) splitter is returned.
        X, y:         Required by SPCVSplitter (passed through for AHC).
        spatial:      Spatial-CV parameters (method, metric, AHC/PCA knobs).
                      Defaults to SpatialCVConfig() when None — note this uses
                      the engine-level "block" method, not PipelineConfig's
                      "spcv" default.

    Returns:
        A scikit-learn-compatible splitter exposing split() and get_n_splits().
    """
    spatial = spatial or SpatialCVConfig()
    if coords is not None:
        if spatial.spatial_cv_method == "spcv":
            assert X is not None and y is not None, "SPCVSplitter requires X and y."
            return SPCVSplitter(
                coords=coords,
                X=X,
                y=y,
                n_splits=n_splits,
                threshold=spatial.ahc_threshold,
                random_state=random_state,
                metric=spatial.spatial_cv_metric,
                pca_components=spatial.pca_components,
                exact_max_samples=spatial.exact_max_samples,
                knn_neighbors=spatial.knn_neighbors,
            )
        return SpatialBlockSplitter(
            coords=coords,
            n_splits=n_splits,
            n_blocks_per_fold=spatial.n_blocks_per_fold,
            random_state=random_state,
            metric=spatial.spatial_cv_metric,
        )
    kwargs = dict(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if TaskType(task_type) == TaskType.CLASSIFICATION:
        return StratifiedKFold(**kwargs)
    return KFold(**kwargs)
