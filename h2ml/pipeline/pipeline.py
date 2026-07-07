"""
h2ml/pipeline/pipeline.py

H2MLPipeline orchestrates the 4-step AutoML workflow; PipelineConfig holds all
tunable parameters; PipelineResult carries every artifact produced by the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, cast

from loguru import logger

if TYPE_CHECKING:
    from h2ml.features.spatial_cv import SpatialMetric
    from h2ml.pipeline.final_model import FinalModel
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from h2ml.core.base import TaskType
from h2ml.core.feature_store import PipelineData
from h2ml.core.spatial_config import SpatialCVConfig
from h2ml.core.step import ModelWrapper
from h2ml.evaluation.metrics import (
    # Metric metadata (short name → agg column, minimise direction) — used by
    # PipelineConfig.metric_col / minimize_metric.
    METRIC_COL,
    METRIC_MINIMIZE,
    RunMetadata,
    aggregate_metrics,
    compute_metrics_all,
    select_best,
)
from h2ml.features.selector import FeatureSelector
from h2ml.optimization.optimizer import run_study
from h2ml.pipeline.cv import CrossValidator, CVResult
from h2ml.preprocessing.transform_stores import build_transform_stores
from h2ml.utils.registry import build_models


@dataclass
class PipelineConfig:
    """
    Configuration for H2MLPipeline.

    Attributes:
        task_type:           Task to run. Accepts the string "classification" or
                             "regression" (case-insensitive) or a TaskType member;
                             normalised to TaskType in __post_init__.
        metric:              Short metric name used for model selection (steps 1–3) and
                             HPO (step 4). Minimisation direction and sort order are
                             derived automatically — no need to set a separate flag.
                             Classification: "AUC" (default), "AUC_PR", "F1", "LogLoss",
                             "Brier". Regression: "R2" (default), "MAE", "RMSE".
        n_splits:            CV folds used in steps 1–3 for model screening and feature
                             selection. More folds = more reliable estimates, higher
                             cost. Default: 5.
        random_state:        Seed for fold splitting and model initialisation. Default: 42.
        verbose:             Log step-by-step progress. Default: False.

        corr_threshold:      (Step 2) Drop a feature when its correlation with any
                             already-retained feature exceeds this value in Pearson,
                             Spearman, or Kendall. Range (0, 1]. Default: 0.7.
        min_features:        (Step 2) Hard lower bound on features kept after the
                             correlation filter; prevents the selector from removing
                             everything. Default: 1.

        n_trials:            (Step 4) Total Optuna trials budget. Each trial evaluates
                             one hyperparameter configuration using opt_n_splits-fold
                             CV, so total model fits = n_trials × opt_n_splits (plus a
                             final n_splits-fold CV on the best params). Default: 50.
        opt_n_splits:        (Step 4) CV folds inside each Optuna trial. Fewer folds make
                             each trial faster at the cost of a noisier score estimate.
                             Should be ≥ n_splits // 2 to avoid unreliable HPO scores;
                             setting it equal to n_splits gives unbiased estimates at
                             higher compute cost. Default: 3.
        n_hpo_repeats:       (Step 4) Number of independent HPO runs with different fold
                             seeds. The repeat with the highest best_value is kept.
                             Trials are divided evenly across repeats: trials_per_repeat
                             = max(1, n_trials // n_hpo_repeats), so the total fit count
                             stays constant. Default: 1.

        spatial_cv_method:   Splitting strategy when spatial coordinates are provided
                             (all spatial-CV fields are ignored when store.coords is
                             None). "block" — quantile-grid blocking; fast, no
                             clustering. "spcv" — Agglomerative Hierarchical Clustering +
                             cluster ensemble; more spatially coherent folds, slower.
                             Default: "spcv".
        spatial_cv_metric:   Distance metric used by both splitters. "euclidean" —
                             projected coordinates (metres). "haversine" — geographic
                             lat/lon in decimal degrees. Default: "euclidean".
        n_blocks_per_fold:   Number of spatial blocks assigned to the test set per fold
                             in the block splitter. Default: 5.
        time_bin_resolution: Temporal granularity for binning dates in spatial CV and
                             local conformal calibration. "month" — 12 bins (Jan…Dec).
                             "season" — 4 bins (DJF, MAM, JJA, SON). Default: "month".
        ahc_threshold:       Distance threshold for cutting the AHC dendrogram in
                             SPCVSplitter. Derived automatically from the data when None.
                             Default: None.
        pca_components:      Variance fraction retained by PCA applied to block
                             covariates before AHC clustering in SPCVSplitter.
                             Default: 0.95.
        exact_max_samples:   Sample count below which exact scipy AHC is used; above this
                             an approximate sklearn AHC (k-NN graph) is used instead.
                             Default: 5 000.
        knn_neighbors:       k for the k-NN connectivity graph in approximate AHC.
                             Default: 15.

        handle_imbalance:    (Classification only) Inject class_weight="balanced" into
                             every model whose registry entry has
                             supports_class_weight=True. No effect on regression tasks.
                             Default: False.
    """

    task_type: TaskType | str = TaskType.CLASSIFICATION
    metric: str = "AUC"
    n_splits: int = 5
    opt_n_splits: int = 3
    corr_threshold: float = 0.7
    n_trials: int = 50
    random_state: int = 42
    verbose: bool = False
    min_features: int = 1
    n_hpo_repeats: int = 1
    n_blocks_per_fold: int = 5
    spatial_cv_method: str = "spcv"
    ahc_threshold: Optional[float] = None
    spatial_cv_metric: SpatialMetric = "euclidean"
    time_bin_resolution: str = "month"
    pca_components: float = 0.95
    exact_max_samples: int = 5_000
    knn_neighbors: int = 15
    handle_imbalance: bool = False

    def __post_init__(self):
        # Accept the string "classification" / "regression" so callers need not import TaskType.
        if not isinstance(self.task_type, TaskType):
            try:
                self.task_type = TaskType(str(self.task_type).lower())
            except ValueError:
                raise ValueError(
                    f'task_type must be "classification" or "regression" (or a TaskType), got {self.task_type!r}'
                ) from None
        if self.task_type not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            raise ValueError(f'task_type must be "classification" or "regression", got "{self.task_type.value}"')
        if self.metric not in METRIC_COL:
            raise ValueError(f"metric must be one of {list(METRIC_COL)}, got {self.metric!r}")
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.opt_n_splits < 2:
            raise ValueError(f"opt_n_splits must be >= 2, got {self.opt_n_splits}")
        if self.opt_n_splits < self.n_splits // 2:
            logger.warning(
                f"PipelineConfig: opt_n_splits={self.opt_n_splits} is less than half of "
                f"n_splits={self.n_splits}. HPO fold estimates will be noisy and may not "
                f"generalise to the step-4 evaluation. Consider setting "
                f"opt_n_splits={self.n_splits} for unbiased HPO at higher compute cost."
            )
        if not 0 < self.corr_threshold <= 1:
            raise ValueError(f"corr_threshold must be in (0, 1], got {self.corr_threshold}")
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.min_features < 1:
            raise ValueError(f"min_features must be >= 1, got {self.min_features}")
        if self.n_hpo_repeats < 1:
            raise ValueError(f"n_hpo_repeats must be >= 1, got {self.n_hpo_repeats}")
        if self.n_blocks_per_fold < 1:
            raise ValueError(f"n_blocks_per_fold must be >= 1, got {self.n_blocks_per_fold}")
        if self.spatial_cv_method not in ("block", "spcv"):
            raise ValueError(f"spatial_cv_method must be 'block' or 'spcv', got {self.spatial_cv_method!r}")

    @property
    def minimize_metric(self) -> bool:
        """True for error metrics (LogLoss, Brier, MAE, RMSE), False for score metrics."""
        return METRIC_MINIMIZE[self.metric]

    @property
    def metric_col(self) -> str:
        """Full agg DataFrame column name, e.g. 'AUC' → 'AUC_Test_Mean'."""
        return METRIC_COL[self.metric]

    @property
    def spatial_cv(self) -> SpatialCVConfig:
        """Bundle the spatial-CV parameters passed to the splitter and optimizer."""
        return SpatialCVConfig(
            n_blocks_per_fold=self.n_blocks_per_fold,
            spatial_cv_method=self.spatial_cv_method,
            ahc_threshold=self.ahc_threshold,
            spatial_cv_metric=self.spatial_cv_metric,
            pca_components=self.pca_components,
            exact_max_samples=self.exact_max_samples,
            knn_neighbors=self.knn_neighbors,
        )


@dataclass
class PipelineResult:
    """
    Container for all artifacts produced by H2MLPipeline.

    Fields are populated progressively as steps complete; check completed_steps
    to see which steps have run. Use summary() to compare all stages at once and
    build_final_model() to refit on the full training set.

    Attributes:
        features:             Input PipelineData (full feature set).
        step1_fold_df:        Per-fold metrics from step 1 (all models × transforms).
        step1_agg_df:         Aggregated (mean ± std) step-1 metrics per model.
        best_model_name:      Winning model name after step 1.
        best_model_value:     Best model's score on the selection metric.
        best_model_std:       Fold std of the best model's selection metric.
        features_reduced:     PipelineData after step-2 feature reduction.
        selector:             Fitted FeatureSelector (SHAP + correlation filter).
        step3_fold_df:        Per-fold metrics from step 3 (reduced features).
        step3_agg_df:         Aggregated step-3 metrics per model.
        best_stage:           Winning stage after step 3: "default" or "reduced".
        best_feature_stage:   Feature stage step 4 is built on — "default" or
                              "reduced", never "optimized".
        step3_reduced_stores: Cached reduced PipelineDatas keyed by transform name
                              ("" for no transform). Not persisted.
        best_params:          Best hyperparameters found in step 4 (None = defaults).
        step4_fold_df:        Per-fold metrics from the step-4 final CV.
        step4_agg_df:         Aggregated step-4 metrics.
        step1_cv_result:      Raw step-1 CVResults — preserved for plots/persistence.
        step3_cv_result:      Raw step-3 CVResults.
        step4_cv_result:      Raw step-4 CVResult.
        y_transform:          Winning y-transform name (set when run() sweeps transforms).
        cv_type:              "spatial" when store.coords was set, else "random".
        spatial_cv_metric:    Distance metric forwarded from PipelineConfig; used by
                              build_final_model to build LocalConformalCalibration.
        time_bin_resolution:  Temporal bin resolution forwarded from PipelineConfig.
        cv_warnings:          Warnings for models with ≥1 failed CV fold (steps 1 & 3).
        metric:               Short selection metric name, e.g. "AUC" or "R2".
        splitter:             Splitter built once in step 1 and reused in steps 3-4.
                              Not persisted.
    """

    # Input data
    features: Optional[PipelineData] = None
    # Step 1: CV all models × all transforms + select best (model, transform)
    step1_fold_df: Optional[pd.DataFrame] = None
    step1_agg_df: Optional[pd.DataFrame] = None
    best_model_name: Optional[str] = None
    best_model_value: Optional[float] = None
    best_model_std: Optional[float] = None
    # Step 2: feature reduction based on winning (model, transform)
    features_reduced: Optional[PipelineData] = None
    selector: Optional[FeatureSelector] = None
    # Step 3: CV all models on reduced features (winning transform only) + select best stage
    step3_fold_df: Optional[pd.DataFrame] = None
    step3_agg_df: Optional[pd.DataFrame] = None
    best_stage: Optional[str] = None
    # Feature stage that step 4 is built on — "default" or "reduced", never "optimized"
    best_feature_stage: Optional[str] = None
    # Step 3: cached reduced PipelineDatas keyed by transform name (or "" for no transform)
    step3_reduced_stores: Optional[dict[str, "PipelineData"]] = field(default=None, repr=False)
    # Step 4: hyperparameter optimization + final CV + select overall best
    best_params: Optional[dict] = None
    step4_fold_df: Optional[pd.DataFrame] = None
    step4_agg_df: Optional[pd.DataFrame] = None
    # Raw CV results — preserved for plots and persistence
    step1_cv_result: Optional[list[CVResult]] = None
    step3_cv_result: Optional[list[CVResult]] = None
    step4_cv_result: Optional[CVResult] = None
    # Winning y-transform (set when run() is called with transforms)
    y_transform: Optional[str] = None
    # CV strategy used — "spatial" when store.coords was set, "random" otherwise
    cv_type: str = "random"
    # Distance metric forwarded from PipelineConfig — used by build_final_model
    # to construct LocalConformalCalibration with the correct spatial metric
    spatial_cv_metric: str = "euclidean"
    # Temporal bin resolution forwarded from PipelineConfig
    time_bin_resolution: str = "month"
    # Models that had at least one failed CV fold — populated by step 1 and step 3
    cv_warnings: list[str] = field(default_factory=list)
    # Short metric name used for model selection, e.g. "AUC" or "R2" (from PipelineConfig)
    metric: Optional[str] = None
    # Splitter built once in step 1 and reused in steps 3 and 4 (not persisted)
    splitter: Any = field(default=None, repr=False, compare=False)
    # Transient cache — populated on first call to shap_summary_plot / shap_dependence
    # Not persisted (result_io.py enumerates fields explicitly).
    _final_shap_cache: Optional[tuple] = field(default=None, repr=False, compare=False, init=False)

    @property
    def oof_predictions(self) -> Optional[np.ndarray]:
        """
        Out-of-fold predictions for the best model's CV result.
        Classification: positive-class probability (binary) or probability matrix (multiclass).
        Regression: predicted values. None if no CV result is available.
        """
        cv = self.best_cv_result
        return cv.oof_predictions if cv is not None else None

    @property
    def oof_labels(self) -> Optional[np.ndarray]:
        """True labels paired with oof_predictions."""
        cv = self.best_cv_result
        return cv.oof_labels if cv is not None else None

    @property
    def best_cv_result(self) -> Optional[CVResult]:
        """
        CVResult for the final winning model.
        Returns step4_cv_result when HPO ran, otherwise the best model's
        CVResult from step3 (used when opt_enabled=False).
        """
        if self.step4_cv_result is not None:
            return self.step4_cv_result
        if self.step3_cv_result and self.best_model_name:
            for cv in self.step3_cv_result:
                if cv.model_name == self.best_model_name:
                    return cv
        return None

    @property
    def completed_steps(self) -> list[int]:
        """List of completed step numbers, e.g. [1, 2, 3] after run_step1_to_step3()."""
        steps = []
        if self.step1_agg_df is not None and self.best_model_name is not None:
            steps.append(1)
        if self.features_reduced is not None:
            steps.append(2)
        if self.step3_agg_df is not None and self.best_stage is not None:
            steps.append(3)
        if self.best_params is not None:
            steps.append(4)
        return steps

    def summary(self, metric: Optional[str] = None, ascending: bool = False) -> pd.DataFrame:
        """
        Combined agg DataFrame across all completed stages.

        Args:
            metric:    Column name to sort by (e.g. "AUC_Test_Mean"). When None,
                       rows are returned in stage order (default → reduced → optimized).
            ascending: Sort direction; set True for error metrics (RMSE, MAE).

        Returns:
            DataFrame with a "Stage" column prepended. Empty if no steps have run.

        Raises:
            ValueError: If metric is given but not present in the summary columns.
        """
        rows = []
        for agg_df, stage in [
            (self.step1_agg_df, "default"),
            (self.step3_agg_df, "reduced"),
            (self.step4_agg_df, "optimized"),
        ]:
            if agg_df is not None:
                row = agg_df.copy()
                row["Stage"] = stage
                cols = ["Stage"] + [c for c in row.columns if c != "Stage"]
                rows.append(row[cols])

        if not rows:
            return pd.DataFrame()

        df = pd.concat(rows, ignore_index=True)

        if metric is not None:
            if metric not in df.columns:
                available = [c for c in df.columns if c.endswith("_Mean")]
                raise ValueError(f"Metric '{metric}' not found in summary. Available: {available}")
            df = df.sort_values(metric, ascending=ascending).reset_index(drop=True)

        return df

    def build_final_model(self) -> "FinalModel":
        """
        Fit the best model on the full training dataset and return a FinalModel.

        The FinalModel handles inference (predict / predict_proba), scaling,
        and can be saved independently of this result.

        Returns:
            FinalModel ready for prediction on new data.
        """
        from h2ml.pipeline.final_model import FinalModel  # noqa: F401
        from h2ml.pipeline.final_model import build_final_model as _build

        return _build(self)

    def save(self, path: str | Path) -> None:
        """Persist this result to *path*. Reload with PipelineResult.load(path)."""
        from h2ml.persistence.result_io import save_result

        save_result(self, path)

    @classmethod
    def load(cls, path: str) -> "PipelineResult":
        """Reload a result previously saved with result.save(path)."""
        from h2ml.persistence.result_io import load_result

        return load_result(path)

    def __repr__(self) -> str:
        return (
            f"PipelineResult("
            f"completed_steps={self.completed_steps}, "
            f"metric={self.metric!r}, "
            f"best_model={self.best_model_name!r}, "
            f"best_stage={self.best_stage!r}, "
            f"cv_type={self.cv_type!r})"
        )


class H2MLPipeline:
    """
    Orchestrates the full h2ml 4-step workflow.

    Step 1 — CV all models (× all y-transforms) on all features → select best (model [, transform])
    Step 2 — SHAP importance + correlation-based feature reduction using winning model/transform
    Step 3 — CV all models on reduced features (winning transform only) → select best stage
    Step 4 — Hyperparameter optimization + final CV → select overall best

    Y-transform sweep:
        Pass a list of transform names to run() to sweep transforms within each step
        rather than running the full pipeline independently per transform.

        >>> result = pipeline.run(store, transforms=["log", "sqrt", "count"])

    Partial runs:
        run_step1_only()      — step 1 only, quick model screening
        run_step1_to_step2()  — steps 1-2, inspect SHAP importance / reduced features
        run_step1_to_step3()  — steps 1-3, full selection without HPO
        run_from_step3()      — resume from a pre-reduced result (steps 3-4)

    Args:
        config:   Pipeline configuration.
        models:   Override the default model registry list. When None, build_models()
                  constructs a task-appropriate set from the registry.
        metadata: Optional experiment labels (schema, target, batch) that appear as
                  columns in fold/agg DataFrames. When None, only stage is set.

    Raises:
        ValueError: If the resolved model list is empty.

    Example:
        >>> pipeline = H2MLPipeline(config=PipelineConfig())
        >>> result   = pipeline.run(store)
        >>> result.summary()
    """

    def __init__(
        self,
        config: PipelineConfig,
        models: Optional[list[ModelWrapper]] = None,
        metadata: Optional[RunMetadata] = None,
    ):
        if models is None:
            models = build_models(config.task_type.value)
        if not models:
            raise ValueError("models cannot be empty.")
        self.models = models
        self._model_map = {m.name: m for m in models}
        self.config = config
        self.metadata = metadata
        if config.handle_imbalance and config.task_type == TaskType.CLASSIFICATION:
            self._inject_class_weights()
        self._cv = CrossValidator(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=config.random_state,
            verbose=config.verbose,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        store: PipelineData,
        transforms: Optional[Iterable[str]] = None,
    ) -> PipelineResult:
        """
        Execute the full 4-step pipeline.

        Args:
            store:      Input features and target.
            transforms: Optional sequence of y-transform names to sweep (regression only).
                        When provided, step 1 runs all models × all transforms and selects
                        the best (model, transform). Step 3 re-runs only the winning transform
                        on the reduced feature set. Defaults to None (no transform sweep).

        Returns:
            PipelineResult with all four steps completed.
        """
        self._validate_store(store)
        result = self._new_result(store)
        transform_stores = self._build_transform_stores(store, transforms)
        result = self._run_step1(result, transform_stores)
        result = self._run_step2(result, transform_stores)
        result = self._run_step3(result, transform_stores)
        result = self._run_step4(result, transform_stores)
        self._log_summary(result)
        return result

    # ------------------------------------------------------------------
    # Partial runs
    # ------------------------------------------------------------------

    def run_step1_only(
        self,
        store: PipelineData,
        transforms: Optional[Iterable[str]] = None,
    ) -> PipelineResult:
        """
        CV all models on all features and select the best — quick model screening.
        After: result.best_model_name, result.step1_agg_df.

        Returns:
            PipelineResult with step 1 completed.
        """
        self._validate_store(store)
        transform_stores = self._build_transform_stores(store, transforms)
        return self._run_step1(
            self._new_result(store),
            transform_stores,
        )

    def run_step1_to_step2(
        self,
        store: PipelineData,
        transforms: Optional[Iterable[str]] = None,
    ) -> PipelineResult:
        """
        Run steps 1-2.
        After: result.selector.importance_summary() / result.features_reduced.feature_names

        Returns:
            PipelineResult with steps 1-2 completed.
        """
        self._validate_store(store)
        transform_stores = self._build_transform_stores(store, transforms)
        result = self._new_result(store)
        result = self._run_step1(result, transform_stores)
        result = self._run_step2(result, transform_stores)
        return result

    def run_step1_to_step3(
        self,
        store: PipelineData,
        transforms: Optional[Iterable[str]] = None,
    ) -> PipelineResult:
        """
        Run steps 1-3 (model selection on full and reduced features) without HPO.
        Useful for inspecting best_stage and comparing models before committing to
        step 4's compute cost.
        After: result.best_model_name, result.best_stage, result.step3_agg_df

        Returns:
            PipelineResult with steps 1-3 completed (no HPO).
        """
        self._validate_store(store)
        transform_stores = self._build_transform_stores(store, transforms)
        result = self._new_result(store)
        result = self._run_step1(result, transform_stores)
        result = self._run_step2(result, transform_stores)
        result = self._run_step3(result, transform_stores)
        return result

    def run_from_step3(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]] = None,
    ) -> PipelineResult:
        """
        Resume from step 3 using a PipelineResult from run_step1_to_step2().
        Requires result.features, result.features_reduced and result.best_model_name to be set.

        Args:
            result:           PipelineResult carrying steps 1-2 state.
            transform_stores: Pre-built y-transform stores; rebuilt from result.features
                              when None.

        Returns:
            PipelineResult with steps 3-4 completed.

        Raises:
            ValueError: If features, features_reduced, best_model_name, or selector
                is missing from result.
        """
        missing = [
            attr
            for attr in ("features", "features_reduced", "best_model_name", "selector")
            if getattr(result, attr) is None
        ]
        if missing:
            raise ValueError(f"result must have {missing} set. Run run_step1_to_step2() first.")
        transform_stores = self._rebuild_transform_stores(result, transform_stores)
        result = self._run_step3(result, transform_stores)
        result = self._run_step4(result, transform_stores)
        return result

    def run_step4_only(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]] = None,
    ) -> PipelineResult:
        """
        Run only step 4 (HPO + final CV) on a result that already has steps 1–3 complete.

        Typical use: load a saved result, then call this to add or redo optimisation
        without repeating CV on all models.

        Requires: features, features_reduced, selector, best_model_name, best_stage,
                  best_model_value.

        Args:
            result:           PipelineResult with steps 1-3 complete.
            transform_stores: Pre-built y-transform stores; rebuilt from result.features
                              when None.

        Returns:
            PipelineResult with step 4 completed.

        Raises:
            ValueError: If any of the required steps 1-3 fields are missing.
        """
        missing = [
            attr
            for attr in (
                "features",
                "features_reduced",
                "selector",
                "best_model_name",
                "best_stage",
                "best_model_value",
            )
            if getattr(result, attr) is None
        ]
        if missing:
            raise ValueError(
                f"Cannot run step 4: missing fields {missing}. "
                "Steps 1–3 must be complete (run run_step1_to_step3() or load a result "
                "that includes them)."
            )
        transform_stores = self._rebuild_transform_stores(result, transform_stores)
        return self._run_step4(result, transform_stores)

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _run_step1(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]] = None,
    ) -> PipelineResult:
        """CV all models (× transforms) on all features, then select best."""
        assert result.features is not None, "result.features must be set before step 1."
        n_transforms = len(transform_stores) if transform_stores else 1
        self._log(
            f"Step 1 — {self.config.n_splits}-fold CV {len(self.models)} models × {n_transforms} transform(s) "
            f"on all features → select best"
        )

        stores = transform_stores or {"": result.features}
        all_fold_dfs: list[pd.DataFrame] = []
        all_cv_results: list[CVResult] = []

        # Build the splitter once here using the first store.
        # Coords and X are identical across all y-transforms, so one splitter
        # covers every transform iteration and can be reused in steps 3 and 4.
        first_store = next(iter(stores.values()))
        result.splitter = self._cv._build_splitter(
            self.config.task_type,
            X=first_store.X,
            y=first_store.y,
            coords=first_store.coords,
            spatial=self.config.spatial_cv,
        )

        # Pre-compute StandardScaler fold arrays once — X is identical across all
        # transforms, so scaler fits are the same for every (transform, model) pair.
        # Workers receive the pre-computed arrays and skip _maybe_scale() entirely.
        fold_indices = list(result.splitter.split(first_store.X, first_store.y))
        has_scaling = any(getattr(m, "requires_scaling", False) for m in self.models)
        precomputed_scaled_folds = None
        if has_scaling:
            precomputed_scaled_folds = []
            for tr_idx, te_idx in fold_indices:
                sc = StandardScaler()
                precomputed_scaled_folds.append(
                    (
                        sc.fit_transform(first_store.X[tr_idx]),
                        sc.transform(first_store.X[te_idx]),
                    )
                )

        # Flatten all (transform × model) pairs into one parallel batch.
        # Avoids 5 serial run_all() calls and better saturates available CPUs:
        # ⌈60/n_cpus⌉ scheduling rounds instead of 5 × ⌈12/n_cpus⌉.
        splitter = result.splitter

        def _run_one(transform_name, store, model):
            import warnings

            warnings.filterwarnings("ignore")
            inv = self._get_inverse_fn(store)
            sf = precomputed_scaled_folds if getattr(model, "requires_scaling", False) else None
            return transform_name, self._cv.run(
                model,
                store.X,
                store.y,
                y_true=store.y_true,
                inverse_fn=inv,
                coords=store.coords,
                spatial=self.config.spatial_cv,
                _splitter=splitter,
                _scaled_folds=sf,
            )

        raw = cast(
            list[tuple[str, CVResult]],
            Parallel(n_jobs=-1, backend="loky")(
                delayed(_run_one)(tn, st, m) for tn, st in stores.items() for m in self.models
            ),
        )

        # Group by transform and compute metrics — identical to the serial path.
        by_transform: dict[str, list[CVResult]] = {}
        for transform_name, cv_result in raw:
            by_transform.setdefault(transform_name, []).append(cv_result)

        for transform_name, cv_results in by_transform.items():
            metadata = self._metadata_with(stage="default")
            if transform_name:
                metadata = replace(metadata, y_transform=transform_name)
            all_cv_results.extend(cv_results)
            all_fold_dfs.append(compute_metrics_all(cv_results, metadata=metadata))

        result.step1_cv_result = all_cv_results
        result.step1_fold_df = pd.concat(all_fold_dfs, ignore_index=True)
        result.step1_agg_df = aggregate_metrics(result.step1_fold_df)
        result.cv_warnings.extend(self._collect_cv_warnings(all_cv_results))

        best = select_best(
            result.step1_agg_df,
            metric=self.config.metric_col,
            minimize=self.config.minimize_metric,
            n_folds=self.config.n_splits,
        )
        result.best_model_name = best["model_name"]
        result.best_model_value = float(best["value"])
        std_col = self.config.metric_col.replace("_Mean", "_Std")
        result.best_model_std = float(best["row"].get(std_col, float("nan")))
        result.best_stage = "default"
        result.y_transform = best["row"].get("Y_transform")
        result.metric = self.config.metric

        self._log(
            f"  Best: {result.best_model_name}"
            + (f" | transform={result.y_transform}" if result.y_transform else "")
            + f" ({self.config.metric}={result.best_model_value:.4f} ± {result.best_model_std:.4f})"
        )
        return result

    def _run_step2(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]] = None,
    ) -> PipelineResult:
        """SHAP importance + correlation-based feature reduction on winning transform's store."""
        assert result.features is not None, "result.features must be set before step 2."
        self._log(f"Step 2 — Feature reduction (threshold={self.config.corr_threshold})")

        # Use the winning transform's store for SHAP; fall back to the raw store
        if transform_stores and result.y_transform:
            fit_store = transform_stores[result.y_transform]
        else:
            fit_store = result.features

        best_model = self._get_model(result.best_model_name)
        selector = FeatureSelector(
            step=best_model,
            task_type=self.config.task_type,
            corr_threshold=self.config.corr_threshold,
            min_features=self.config.min_features,
            use_oof=True,
            n_splits=self.config.n_splits,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            _splitter=result.splitter,
        )
        result.selector = selector
        result.features_reduced = selector.fit_transform(fit_store)
        return result

    def _run_step3(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]] = None,
    ) -> PipelineResult:
        """CV all models on reduced features (step-1-winning transform only), then select best stage."""
        assert result.features_reduced is not None, "features_reduced must be set before step 3."
        assert result.selector is not None, "selector must be set before step 3."
        n_features = result.features_reduced.n_features
        self._log(
            f"Step 3 — {self.config.n_splits}-fold CV {len(self.models)} models"
            + (f" | transform={result.y_transform}" if result.y_transform else "")
            + f" on {n_features} reduced features → select best stage"
        )

        # Lock to the step-1-winning transform — the selector was fitted under that transform,
        # so the selected features are only guaranteed relevant for it.
        # result.features_reduced is already the correct reduced store (built in step 2).
        transform_name = result.y_transform or ""
        reduced_store = result.features_reduced
        inverse_fn = self._get_inverse_fn(reduced_store)
        metadata = self._metadata_with(stage="reduced")
        if transform_name:
            metadata = replace(metadata, y_transform=transform_name)
        cv_results = self._cv.run_all(
            self.models,
            reduced_store.X,
            reduced_store.y,
            y_true=reduced_store.y_true,
            inverse_fn=inverse_fn,
            coords=reduced_store.coords,
            spatial=self.config.spatial_cv,
            _splitter=result.splitter,
        )
        result.step3_reduced_stores = {transform_name: reduced_store}
        result.step3_cv_result = cv_results
        result.step3_fold_df = compute_metrics_all(cv_results, metadata=metadata)
        result.step3_agg_df = aggregate_metrics(result.step3_fold_df)
        result.cv_warnings.extend(self._collect_cv_warnings(cv_results))

        # Compare step1 (default) and step3 (reduced) agg dfs directly — pick the best row
        assert result.step1_agg_df is not None and result.step3_agg_df is not None
        all_stages_agg = pd.concat([result.step1_agg_df, result.step3_agg_df], ignore_index=True)
        best = select_best(
            all_stages_agg,
            metric=self.config.metric_col,
            minimize=self.config.minimize_metric,
            n_folds=self.config.n_splits,
        )
        result.best_model_name = best["model_name"]
        result.best_model_value = float(best["value"])
        std_col = self.config.metric_col.replace("_Mean", "_Std")
        result.best_model_std = float(best["row"].get(std_col, float("nan")))
        result.best_stage = str(best["row"]["Stage"])
        result.best_feature_stage = result.best_stage  # preserved — never overwritten by "optimized"
        result.y_transform = best["row"].get("Y_transform")

        self._log(
            f"  Best: {result.best_model_name} | stage={result.best_stage}"
            + (f" | transform={result.y_transform}" if result.y_transform else "")
            + f" ({self.config.metric}={result.best_model_value:.4f} ± {result.best_model_std:.4f})"
        )
        return result

    def _run_step4(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]] = None,
    ) -> PipelineResult:
        """Hyperparameter optimization + final CV + select overall best."""
        assert result.features is not None, "result.features must be set before step 4."
        assert result.features_reduced is not None, "features_reduced must be set before step 4."
        assert result.best_model_value is not None, "best_model_value must be set before step 4."
        self._log(
            f"Step 4 — Optimize {result.best_model_name} | "
            f"stage='{result.best_stage}'"
            + (f" | transform={result.y_transform}" if result.y_transform else "")
            + f" ({self.config.n_trials} trials)"
        )

        store_for_opt = self._resolve_opt_store(result, transform_stores)
        X = store_for_opt.X
        y = store_for_opt.y
        y_true = store_for_opt.y_true
        inverse_fn = self._get_inverse_fn(store_for_opt)

        best_model = self._get_model(result.best_model_name)
        estimator_name = best_model.estimator.__class__.__name__

        # Fixed params are injected into every Optuna trial alongside searched params.
        # Built before the opt_enabled check: the skip path must also carry them, or a
        # deployed opt-disabled winner would silently lose class_weight="balanced"
        # even though CV ranked it with weights injected.
        fixed_params: dict = {}
        if self.config.handle_imbalance and self._supports_class_weight(estimator_name):
            fixed_params["class_weight"] = "balanced"

        if not self._is_opt_enabled(estimator_name):
            self._log(f"  {estimator_name} has opt_enabled=False — skipping optimization.")
            result.best_params = {**self._get_default_params(estimator_name), **fixed_params}
            return result

        study = run_study(
            name=estimator_name,
            X=X,
            y=y,
            task=self.config.task_type.value,
            metric=self.config.metric,
            n_trials=self.config.n_trials,
            n_splits=self.config.opt_n_splits,
            random_state=self.config.random_state,
            coords=store_for_opt.coords,
            spatial=self.config.spatial_cv,
            n_hpo_repeats=self.config.n_hpo_repeats,
            fixed_params=fixed_params,
            y_true=y_true,
            inverse_fn=inverse_fn,
            _splitter=result.splitter if store_for_opt.coords is not None else None,
        )

        # study.best_params only contains trial.suggest_* values — merge with
        # registry default_kwargs so fixed kwargs (e.g. probability=True for SVC)
        # are preserved when _build_fold_model reconstructs the estimator.
        # fixed_params are re-applied last: every trial was evaluated with them
        # overriding sampled values (see _build_objective), so the final model must
        # match — otherwise a sampled class_weight=None from the search space would
        # silently undo handle_imbalance in the deployed params.
        default_kwargs = {**self._get_default_params(estimator_name), **fixed_params}
        merged_params = {**default_kwargs, **study.best_params, **fixed_params}

        metadata = self._metadata_with(stage="optimized")
        if result.y_transform:
            metadata = replace(metadata, y_transform=result.y_transform)
        cv_result = self._cv.run(
            best_model,
            X,
            y,
            best_params=merged_params,
            y_true=y_true,
            inverse_fn=inverse_fn,
            coords=store_for_opt.coords,
            spatial=self.config.spatial_cv,
            _splitter=result.splitter,
        )
        result.step4_cv_result = cv_result
        result.step4_fold_df = compute_metrics_all([cv_result], metadata=metadata)
        result.step4_agg_df = aggregate_metrics(result.step4_fold_df, group_by_stage=False)

        optimized_value = float(result.step4_agg_df.iloc[0][self.config.metric_col])
        std_col = self.config.metric_col.replace("_Mean", "_Std")
        optimized_std = float(result.step4_agg_df.iloc[0].get(std_col, float("nan")))
        improved = (
            optimized_value < result.best_model_value
            if self.config.minimize_metric
            else optimized_value > result.best_model_value
        )
        if improved:
            result.best_model_value = optimized_value
            result.best_model_std = optimized_std
            result.best_stage = "optimized"
            result.best_params = merged_params
        else:
            result.best_params = default_kwargs

        self._log(
            f"  Optimized {self.config.metric}={optimized_value:.4f} ± {optimized_std:.4f}"
            f" → best stage: {result.best_stage}"
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_transform_stores(
        self,
        store: PipelineData,
        transforms: Optional[Iterable[str]],
    ) -> Optional[dict[str, PipelineData]]:
        """
        Build per-transform PipelineDatas from the raw store.

        Args:
            store:      Raw (untransformed) PipelineData.
            transforms: y-transform names to build; None disables the sweep.

        Returns:
            Dict of transform name → PipelineData, or None when transforms is None.

        Raises:
            ValueError: If transforms are requested for a non-regression task, or
                if every requested transform returned None (e.g. winsorize variants
                with no outliers).
        """
        if transforms is None:
            return None
        if self.config.task_type != TaskType.REGRESSION:
            raise ValueError(
                f"transforms are only supported for regression tasks. Got task_type='{self.config.task_type.value}'."
            )
        stores = build_transform_stores(
            store.X, store.y, store.feature_names, list(transforms), coords=store.coords, times=store.times
        )
        if not stores:
            raise ValueError(
                "No valid transform stores built — all transforms returned None. "
                "Check that the y array has outliers if using winsorize-based transforms."
            )
        return stores

    def _rebuild_transform_stores(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]],
    ) -> Optional[dict[str, PipelineData]]:
        """
        Rebuild the winning transform's store for resumed runs (steps 3-4).

        result.features holds the raw (untransformed) store, so a resume without
        pre-built transform_stores would otherwise optimise on raw y even though
        the pipeline selected a y-transform. Returns transform_stores unchanged
        when provided or when no transform won.
        """
        if transform_stores is not None or not result.y_transform:
            return transform_stores
        assert result.features is not None  # validated by the callers
        return self._build_transform_stores(result.features, [result.y_transform])

    def _resolve_opt_store(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]],
    ) -> PipelineData:
        """Resolve the correct PipelineData for step-4 optimization."""
        # Pick the full or reduced store depending on winning stage
        if transform_stores and result.y_transform:
            full_store = transform_stores[result.y_transform]
        else:
            assert result.features is not None
            full_store = result.features

        if result.best_stage == "reduced":
            assert result.selector is not None
            key = result.y_transform or ""
            if result.step3_reduced_stores is not None and key in result.step3_reduced_stores:
                return result.step3_reduced_stores[key]
            return result.selector.transform(full_store)
        return full_store

    def _new_result(self, store: PipelineData) -> PipelineResult:
        """Build a fresh PipelineResult seeded with store and CV metadata from config."""
        return PipelineResult(
            features=store,
            cv_type="spatial" if store.coords is not None else "random",
            spatial_cv_metric=self.config.spatial_cv_metric,
            time_bin_resolution=self.config.time_bin_resolution,
        )

    def _validate_store(self, store: PipelineData) -> None:
        """
        Validate a store for common data-quality issues before any fitting begins.

        Args:
            store: PipelineData to validate.

        Raises:
            ValueError: If the store has fewer samples than config.n_splits (or any
                other checked data-quality issue).
        """
        if store.n_samples < self.config.n_splits:
            raise ValueError(
                f"store has {store.n_samples} samples but config.n_splits={self.config.n_splits}. "
                "Provide more data or reduce n_splits."
            )
        if len(set(store.feature_names)) < len(store.feature_names):
            from collections import Counter

            counts = Counter(store.feature_names)
            dups = sorted(n for n, c in counts.items() if c > 1)
            raise ValueError(
                f"store.feature_names contains duplicate names: {dups}. Each feature must have a unique name."
            )
        if not np.all(np.isfinite(store.X)):
            bad_mask = ~np.isfinite(store.X)
            col_counts = bad_mask.sum(axis=0)
            bad_cols = [(store.feature_names[i], int(col_counts[i])) for i in np.where(col_counts > 0)[0]]
            n_bad_rows = int(bad_mask.any(axis=1).sum())
            pct = 100 * n_bad_rows / store.n_samples
            col_str = ", ".join(f"{name} ({n})" for name, n in bad_cols[:10])
            if len(bad_cols) > 10:
                col_str += f" … and {len(bad_cols) - 10} more"
            raise ValueError(
                f"store.X contains NaN or infinite values: "
                f"{n_bad_rows}/{store.n_samples} rows affected ({pct:.1f}%). "
                f"Affected features: {col_str}. "
                "Remove or impute them before calling pipeline.run()."
            )
        if not np.all(np.isfinite(store.y)):
            n_bad = int((~np.isfinite(store.y)).sum())
            pct = 100 * n_bad / store.n_samples
            raise ValueError(
                f"store.y contains NaN or infinite values: "
                f"{n_bad}/{store.n_samples} values affected ({pct:.1f}%). "
                "Remove or impute them before calling pipeline.run()."
            )
        constant_cols = np.where(np.all(store.X == store.X[0], axis=0))[0]
        if constant_cols.size > 0:
            names = [store.feature_names[i] for i in constant_cols]
            raise ValueError(
                f"store.X contains {len(names)} constant column(s): {names}. "
                "Drop them before calling pipeline.run() — zero variance breaks SHAP."
            )
        if self.config.task_type == TaskType.CLASSIFICATION:
            n_classes = len(np.unique(store.y))
            if n_classes < 2:
                raise ValueError(
                    f"store.y contains only {n_classes} unique class(es). Classification requires at least 2 classes."
                )
            classes, counts = np.unique(store.y, return_counts=True)
            minority_ratio = counts.min() / counts.sum()
            if minority_ratio < 0.1:
                suggestion = (
                    "Consider setting handle_imbalance=True in PipelineConfig."
                    if not self.config.handle_imbalance
                    else "handle_imbalance=True is active — class_weight='balanced' will be applied."
                )
                logger.warning(
                    f"Class imbalance detected: minority class ratio={minority_ratio:.3f} "
                    f"(classes={classes.tolist()}, counts={counts.tolist()}). {suggestion}"
                )
        if store.coords is not None and not np.all(np.isfinite(store.coords)):
            raise ValueError(
                "store.coords contains NaN or infinite values. Remove or impute them before calling pipeline.run()."
            )

    def _collect_cv_warnings(self, cv_results: list[CVResult]) -> list[str]:
        """Return warning strings for any models that had failed folds."""
        return [
            f"{r.model_name}: {len(r.failed_folds)}/{self.config.n_splits} fold(s) failed (fold IDs: {r.failed_folds})"
            for r in cv_results
            if r.failed_folds
        ]

    def _get_inverse_fn(self, store: PipelineData):
        """Return the inverse transform callable for the store's y_transform, or None."""
        if store.y_transform is None:
            return None
        from h2ml.preprocessing.transforms import INVERSE_TRANSFORMS

        return INVERSE_TRANSFORMS.get(store.y_transform)

    def _is_opt_enabled(self, estimator_name: str) -> bool:
        """Return False if the model's registry entry has opt_enabled=False."""
        from h2ml.utils.registry import CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY

        registry = CLASSIFIER_REGISTRY if self.config.task_type == TaskType.CLASSIFICATION else REGRESSOR_REGISTRY
        entry = registry.get(estimator_name)
        return entry.opt_enabled if entry is not None else True

    def _supports_class_weight(self, estimator_name: str) -> bool:
        """Return True if the classifier's registry entry has supports_class_weight=True."""
        from h2ml.utils.registry import CLASSIFIER_REGISTRY

        entry = CLASSIFIER_REGISTRY.get(estimator_name)
        return entry.supports_class_weight if entry is not None else False

    def _inject_class_weights(self) -> None:
        """Set class_weight='balanced' on every supporting classifier in self.models."""
        from h2ml.utils.registry import CLASSIFIER_REGISTRY

        injected = []
        for model in self.models:
            entry = CLASSIFIER_REGISTRY.get(model.name)
            if entry is not None and entry.supports_class_weight:
                model.estimator.set_params(class_weight="balanced")
                injected.append(model.name)
        if injected:
            logger.debug(f"handle_imbalance: class_weight='balanced' set on {injected}")

    def _get_default_params(self, estimator_name: str) -> dict:
        """Return the default_kwargs for a model from the registry."""
        from h2ml.utils.registry import CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY

        registry = CLASSIFIER_REGISTRY if self.config.task_type == TaskType.CLASSIFICATION else REGRESSOR_REGISTRY
        entry = registry.get(estimator_name)
        return entry.default_kwargs if entry is not None else {}

    def _get_model(self, name: Optional[str]) -> ModelWrapper:
        if name is None:
            raise RuntimeError("best_model_name is not set.")
        model = self._model_map.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' not found. Available: {list(self._model_map)}")
        return model

    def _metadata_with(self, stage: str) -> RunMetadata:
        if self.metadata is None:
            return RunMetadata(stage=stage)
        return replace(self.metadata, stage=stage)

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            logger.info(msg)

    def _log_summary(self, result: PipelineResult) -> None:
        if result.cv_warnings:
            for w in result.cv_warnings:
                logger.warning(f"[CV] {w}")
        if not self.config.verbose:
            return
        logger.info("=" * 60)
        logger.info("Pipeline complete.")
        logger.info(f"  Best stage:        {result.best_stage}")
        logger.info(f"  Best model:        {result.best_model_name}")
        if result.y_transform:
            logger.info(f"  Y-transform:       {result.y_transform}")
        std_str = f" ± {result.best_model_std:.4f}" if result.best_model_std is not None else ""
        logger.info(f"  Metric:            {self.config.metric} = {result.best_model_value:.4f}{std_str}")
        feature_stage = result.best_feature_stage or result.best_stage
        features = (
            result.features_reduced.n_features
            if feature_stage == "reduced" and result.features_reduced is not None
            else result.features.n_features
            if result.features is not None
            else None
        )
        if features:
            logger.info(f"  Features selected: {features}")
        if result.best_params:
            logger.info(f"  Best params:       {result.best_params}")
        logger.info("=" * 60)

    def __repr__(self) -> str:
        return (
            f"H2MLPipeline("
            f"n_models={len(self.models)}, "
            f"task={self.config.task_type.value!r}, "
            f"metric={self.config.metric!r})"
        )
