"""
h2ml/core/spatial_config.py

SpatialCVConfig — the bundle of parameters that control spatial cross-validation
splitter construction. These seven values used to be threaded individually through
PipelineConfig → pipeline steps → CrossValidator → optimizer; grouping them into one
object keeps the long call chains readable and makes adding a parameter a one-line change.

Defaults match the engine-level defaults in CrossValidator, so constructing
SpatialCVConfig() reproduces a direct CrossValidator call with no spatial overrides.

NOTE — deliberate default divergence: SpatialCVConfig defaults to
spatial_cv_method="block" (fast, robust on small data) for direct engine use
(CrossValidator.run / run_study with coords but no config), while PipelineConfig
defaults to "spcv" and always passes an explicit config, so the pipeline never
relies on this default. Pass spatial_cv_method explicitly if you need direct
calls to match pipeline behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from h2ml.features.spatial_cv import SpatialMetric


@dataclass(frozen=True)
class SpatialCVConfig:
    """
    Parameters governing spatial-CV splitter construction.

    Attributes:
        n_blocks_per_fold: Blocks per test fold for SpatialBlockSplitter.
        spatial_cv_method: "block" → SpatialBlockSplitter, "spcv" → SPCVSplitter.
                           Defaults to "block" here; PipelineConfig defaults to
                           "spcv" (see the module docstring for why they differ).
        ahc_threshold:     AHC distance threshold for SPCVSplitter (auto when None).
        spatial_cv_metric: "euclidean" or "haversine".
        pca_components:    Variance retained by PCA on block covariates (SPCV).
        exact_max_samples: Sample threshold for exact vs approximate AHC (SPCV).
        knn_neighbors:     k for the k-NN connectivity graph in approximate AHC.
    """

    n_blocks_per_fold: int = 5
    spatial_cv_method: str = "block"
    ahc_threshold: Optional[float] = None
    spatial_cv_metric: "SpatialMetric" = "euclidean"
    pca_components: float = 0.95
    exact_max_samples: int = 5_000
    knn_neighbors: int = 15
