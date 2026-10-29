"""
h2ml/pipeline/pipeline.py

H2MLPipeline orchestrates the 4-step AutoML workflow; PipelineConfig holds all
tunable parameters; PipelineResult carries every artifact produced by the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, cast

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
    resolve_metric,
    select_best,
)
from h2ml.evaluation.ranking import KEY_COLS, ranking_robustness, ranking_warnings
from h2ml.features.selector import FeatureSelector
from h2ml.features.spatial_cv import BlockGroupSplitter
from h2ml.optimization.optimizer import run_study
from h2ml.pipeline.cv import CrossValidator, CVResult
from h2ml.pipeline.holdout import (
    HOLDOUT_REFITS,
    HOLDOUT_STRATEGIES,
    MIN_HOLDOUT_MINORITY,
    FixedSplit,
    concat_stores,
    split_holdout,
    stage_store,
)
from h2ml.preprocessing.transform_stores import build_transform_stores
from h2ml.preprocessing.transforms import get_train_target_fn
from h2ml.utils.registry import build_models

# Minority-class share below which a target counts as imbalanced — drives both the
# validation warning and handle_imbalance="auto".
IMBALANCE_RATIO_THRESHOLD = 0.1

# Metric-suitability thresholds (see H2MLPipeline._check_metric_suitability).
# Max/min ratio of the minority-class share across test folds above which AUCPR is noisy.
PREVALENCE_SPREAD_RATIO = 2.0
# Test-fold variance of y, as a fraction of the overall variance, below which R2 is unstable.
LOW_FOLD_VARIANCE_RATIO = 0.1
# Share of the total squared deviation carried by the top 1% of points above which RMSE is
# dominated by a few observations.
TAIL_SSD_SHARE = 0.5


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
                             Classification: "AUC" (default), "AUCPR", "F1", "LogLoss",
                             "Brier". Regression: "R2" (default), "MAE", "RMSE".
        n_splits:            CV folds used in steps 1–3 for model screening and feature
                             selection. More folds = more reliable estimates, higher
                             cost. Default: 5.
        random_state:        Seed for fold splitting and model initialisation. Default: 42.
        verbose:             Log step-by-step progress. Default: False.
        n_jobs:              Worker processes used to CV models in parallel (steps 1 and 3)
                             and to explain row chunks in step 2's SHAP pass for
                             model-agnostic explainers (SVC/SVR with non-linear kernels, KNN, ...).
                             -1 uses all cores. Lower it when memory is the constraint
                             rather than CPU — each worker holds its own copy of the
                             model being fitted — or to leave headroom on a shared
                             machine. Must not be 0. Default: -1.

        corr_threshold:      (Step 2) Drop a feature when its correlation with any
                             already-retained feature exceeds this value in Pearson,
                             Spearman, or Kendall. Range (0, 1]. Default: 0.7.
        min_features:        (Step 2) Hard lower bound on features kept after the
                             correlation filter; prevents the selector from removing
                             everything. Default: 1.
        max_vif:             (Step 2) After the correlation filter, drop the lowest-ranked
                             feature whose variance inflation factor exceeds this value,
                             repeating until none does. Catches features that are linear
                             combinations of several retained ones, which pairwise
                             correlations miss. Must be > 1 (5 or 10 are common). None
                             skips it. Default: None.

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
                             supports_class_weight=True. True — always; False — never;
                             "auto" — only when the minority-class ratio of the input
                             target is below IMBALANCE_RATIO_THRESHOLD (0.1), decided
                             when a run starts. No effect on regression tasks.
                             Default: False.
        prefer_reduced_on_tie: In step 3, when the winner uses all features ("default"
                             stage) but a reduced-feature candidate is indistinguishable
                             from it on paired folds (within one Nadeau–Bengio corrected
                             SE — the Indistinguishable flag of ranking_robustness), select
                             the best-ranked such reduced candidate instead (1-SE rule
                             favouring the simpler feature set). Skipped when the reduced
                             set kept every feature. Each switch is logged and recorded in
                             cv_warnings. Default: False.
        select_features_in_folds: (Step 3) Re-run the step-2 selector on each CV fold's
                             training rows and score the reduced candidates on those
                             per-fold feature sets, so the default-vs-reduced comparison
                             is not biased by a feature set chosen on every row. Step 4's
                             acceptance CV uses the same per-fold sets when the reduced
                             stage won. The deployed feature set is still step 2's.
                             Costs about n_splits extra step-2 runs. Default: False.

        holdout_frac:        Fraction of rows set aside before step 1 and scored once on
                             the final selected configuration — an estimate no selection
                             step has seen. Steps 1–4 run on the remaining (dev) rows. With
                             a holdout, conformal calibration uses the holdout residuals
                             and local conformal calibration is skipped. Range [0, 0.5];
                             0 disables it. Default: 0.0.
        holdout_strategy:    How holdout rows are chosen. "auto" — whole spatial blocks
                             when store.coords is set, else random (stratified for
                             classification). "random" — random even with coords. "time" —
                             the latest rows by store.times. Default: "auto".
        holdout_refit:       Rows build_final_model fits on when a holdout exists. "dev" —
                             the dev rows only, so the holdout score and conformal coverage
                             describe the deployed model exactly. "all" — dev + holdout
                             rows; more data, but both are then approximate.
                             Default: "dev".
    """

    task_type: TaskType | str = TaskType.CLASSIFICATION
    metric: str = "AUC"
    n_splits: int = 5
    opt_n_splits: int = 3
    corr_threshold: float = 0.7
    max_vif: Optional[float] = None
    n_trials: int = 50
    random_state: int = 42
    verbose: bool = False
    n_jobs: int = -1
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
    handle_imbalance: bool | Literal["auto"] = False
    prefer_reduced_on_tie: bool = False
    select_features_in_folds: bool = False
    holdout_frac: float = 0.0
    holdout_strategy: Literal["auto", "random", "time"] = "auto"
    holdout_refit: Literal["dev", "all"] = "dev"

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
        self.metric = resolve_metric(self.metric)
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
        if self.max_vif is not None and self.max_vif <= 1:
            raise ValueError(f"max_vif must be > 1, got {self.max_vif}")
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.min_features < 1:
            raise ValueError(f"min_features must be >= 1, got {self.min_features}")
        if self.n_hpo_repeats < 1:
            raise ValueError(f"n_hpo_repeats must be >= 1, got {self.n_hpo_repeats}")
        if self.n_blocks_per_fold < 1:
            raise ValueError(f"n_blocks_per_fold must be >= 1, got {self.n_blocks_per_fold}")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be non-zero (-1 uses all cores), got 0")
        if self.spatial_cv_method not in ("block", "spcv"):
            raise ValueError(f"spatial_cv_method must be 'block' or 'spcv', got {self.spatial_cv_method!r}")
        if not (isinstance(self.handle_imbalance, bool) or self.handle_imbalance == "auto"):
            raise ValueError(f"handle_imbalance must be True, False or 'auto', got {self.handle_imbalance!r}")
        if not 0 <= self.holdout_frac <= 0.5:
            raise ValueError(f"holdout_frac must be in [0, 0.5], got {self.holdout_frac}")
        if self.holdout_strategy not in HOLDOUT_STRATEGIES:
            raise ValueError(f"holdout_strategy must be one of {HOLDOUT_STRATEGIES}, got {self.holdout_strategy!r}")
        if self.holdout_refit not in HOLDOUT_REFITS:
            raise ValueError(f"holdout_refit must be one of {HOLDOUT_REFITS}, got {self.holdout_refit!r}")
        if not isinstance(self.prefer_reduced_on_tie, bool):
            raise ValueError(f"prefer_reduced_on_tie must be True or False, got {self.prefer_reduced_on_tie!r}")

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
        best_base_estimator:  Unfitted clone of the winning model's estimator, exactly as
                              CV ran it (constructor args and injected class_weight
                              included). build_final_model refits clone(this) with
                              best_params set on top.
        best_requires_scaling: Whether the winning model was scaled in CV.
        best_params:          Overrides applied to best_base_estimator in step 4 (tuned
                              params plus fixed ones such as class_weight). None when
                              step 4 did not run. Results saved before
                              best_base_estimator existed hold full constructor kwargs.
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
        cv_warnings:          Warnings for models with ≥1 failed CV fold (steps 1 & 3), plus
                              data conditions that make the selection metric unreliable
                              (checked in step 1), ranking-robustness flags (steps 1 & 3),
                              and any prefer_reduced_on_tie switch (step 3). All but the
                              switch are advisory.
        class_weight_balanced: True when CV ran with class_weight="balanced" injected
                              (handle_imbalance resolved active); build_final_model
                              re-applies it when step 4 did not produce best_params.
        metric:            Short selection metric name, e.g. "AUC" or "R2".
        splitter:             Splitter built once in step 1 and reused in steps 3-4.
                              Not persisted.
        step3_fold_features:  Feature names selected on each step-3 CV fold's training rows
                              (select_features_in_folds=True), indexed like the splitter's
                              folds; None otherwise.
        holdout_store:        Raw rows set aside by holdout_frac; None without a holdout.
                              features then holds only the dev rows steps 1–4 ran on.
        holdout_idx:          Row indices of holdout_store in the store passed to run().
        holdout_strategy:     Resolved holdout strategy: "spatial", "random" or "time".
        holdout_refit:        "dev" or "all" — rows build_final_model fits on (from
                              PipelineConfig.holdout_refit).
        holdout_cv_result:    One-fold CVResult: the final configuration fitted on the dev
                              rows and scored on the holdout rows.
        holdout_df:           Holdout metrics, same _Mean columns as the agg DataFrames
                              (_Std is NaN — there is one fold).
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
    # Winning estimator as CV ran it (unfitted) — the base build_final_model refits
    best_base_estimator: Any = field(default=None, repr=False)
    best_requires_scaling: bool = False
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
    # Failed CV folds (steps 1 & 3), metric-suitability (step 1), ranking robustness (steps 1 & 3)
    cv_warnings: list[str] = field(default_factory=list)
    # Whether CV ran with class_weight="balanced" (resolved handle_imbalance)
    class_weight_balanced: bool = False
    # Short metric name used for model selection, e.g. "AUC" or "R2" (from PipelineConfig)
    metric: Optional[str] = None
    # Splitter built once in step 1 and reused in steps 3 and 4 (not persisted)
    splitter: Any = field(default=None, repr=False, compare=False)
    # Per-fold feature sets from select_features_in_folds (step 3)
    step3_fold_features: Optional[list[list[str]]] = None
    # Outer holdout — rows no selection step has seen, scored once on the final configuration
    holdout_store: Optional[PipelineData] = field(default=None, repr=False)
    holdout_idx: Optional[np.ndarray] = field(default=None, repr=False)
    holdout_strategy: Optional[str] = None
    holdout_refit: str = "dev"
    holdout_cv_result: Optional[CVResult] = field(default=None, repr=False)
    holdout_df: Optional[pd.DataFrame] = None
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
        Combined agg DataFrame across all completed stages, plus a "holdout" row
        when a holdout was scored.

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
            (self.holdout_df, "holdout"),
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

    def ranking_robustness(self, step: Optional[int] = None) -> pd.DataFrame:
        """
        Paired-fold and cross-metric robustness of the step-1 or step-3 selection.

        See h2ml.evaluation.ranking.ranking_robustness for the columns. Works on
        loaded results — fold DataFrames and metric are persisted. Is_Winner marks
        the confidence-bound winner; after a prefer_reduced_on_tie switch it differs
        from best_stage (the switch is recorded in cv_warnings).

        Args:
            step: 1 (models × transforms on all features) or 3 (default and reduced
                  candidates together, as step 3 selects among both). None → 3 when
                  step 3 has run, else 1.

        Returns:
            One row per candidate, sorted by rank under result.metric.

        Raises:
            ValueError: If step is not 1/3 or that step has not run.
        """
        if step is None:
            step = 3 if self.step3_fold_df is not None else 1
        if step not in (1, 3):
            raise ValueError(f"step must be 1 or 3, got {step!r}")
        if self.metric is None or self.step1_fold_df is None or (step == 3 and self.step3_fold_df is None):
            raise ValueError(f"Step {step} has not run on this result.")
        fold_df = self.step1_fold_df
        if step == 3:
            fold_df = pd.concat([self.step1_fold_df, self.step3_fold_df], ignore_index=True)
        return ranking_robustness(fold_df, self.metric)

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
        # Whether class_weight="balanced" is in effect. "auto" is resolved per run by
        # _resolve_class_weights() once the target is known.
        self._class_weight_active = False
        if config.handle_imbalance is True and config.task_type == TaskType.CLASSIFICATION:
            self._inject_class_weights()
            self._class_weight_active = True
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
        result, store = self._start(store)
        transform_stores = self._build_transform_stores(store, transforms)
        result = self._run_step1(result, transform_stores)
        result = self._run_step2(result, transform_stores)
        result = self._run_step3(result, transform_stores)
        result = self._run_step4(result, transform_stores)
        result = self._score_holdout(result)
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
        result, store = self._start(store)
        transform_stores = self._build_transform_stores(store, transforms)
        return self._run_step1(result, transform_stores)

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
        result, store = self._start(store)
        transform_stores = self._build_transform_stores(store, transforms)
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
        result, store = self._start(store)
        transform_stores = self._build_transform_stores(store, transforms)
        result = self._run_step1(result, transform_stores)
        result = self._run_step2(result, transform_stores)
        result = self._run_step3(result, transform_stores)
        return self._score_holdout(result)

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
        assert result.features is not None  # validated above
        self._resolve_class_weights(result.features.y)
        result.class_weight_balanced = self._class_weight_active
        transform_stores = self._rebuild_transform_stores(result, transform_stores)
        result = self._run_step3(result, transform_stores)
        result = self._run_step4(result, transform_stores)
        return self._score_holdout(result)

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
        assert result.features is not None  # validated above
        self._resolve_class_weights(result.features.y)
        result.class_weight_balanced = self._class_weight_active
        transform_stores = self._rebuild_transform_stores(result, transform_stores)
        result = self._run_step4(result, transform_stores)
        return self._score_holdout(result)

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _run_step1(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]] = None,
    ) -> PipelineResult:
        """CV all models (× transforms) on all features, then select best."""
        fold_indices = self._cv_step1(result, transform_stores)
        return self._select_step1(result, fold_indices)

    def _cv_step1(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Step-1 CV of all models (× transforms) on all features. Metric-independent:
        sets the splitter and step1_* fields and returns the step-1 fold indices.
        """
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
                train_target_fn=get_train_target_fn(store.y_transform),
                _splitter=splitter,
                _scaled_folds=sf,
            )

        raw = cast(
            list[tuple[str, CVResult]],
            Parallel(n_jobs=self.config.n_jobs, backend="loky")(
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
        return fold_indices

    def _select_step1(
        self,
        result: PipelineResult,
        fold_indices: list[tuple[np.ndarray, np.ndarray]],
    ) -> PipelineResult:
        """Select the step-1 winner under config.metric from the step-1 CV (see _cv_step1)."""
        assert result.features is not None and result.step1_agg_df is not None
        assert result.step1_fold_df is not None
        result.cv_warnings.extend(self._check_metric_suitability(result.features.y, fold_indices))
        best = select_best(
            result.step1_agg_df,
            metric=self.config.metric_col,
            minimize=self.config.minimize_metric,
            n_folds=self.config.n_splits,
            fold_df=result.step1_fold_df,
        )
        result.best_model_name = best["model_name"]
        result.best_model_value = float(best["value"])
        std_col = self.config.metric_col.replace("_Mean", "_Std")
        result.best_model_std = float(best["row"].get(std_col, float("nan")))
        result.best_stage = "default"
        result.y_transform = best["row"].get("Y_transform")
        self._record_best_estimator(result)
        result.metric = self.config.metric
        robust_df = ranking_robustness(result.step1_fold_df, self.config.metric, n_folds=self.config.n_splits)
        result.cv_warnings.extend(self._check_ranking_robustness(robust_df, step=1))

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

        selector = self._make_selector(result.best_model_name, fit_store, result.splitter)
        result.selector = selector
        result.features_reduced = selector.fit_transform(fit_store)
        return result

    def _run_step3(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]] = None,
    ) -> PipelineResult:
        """CV all models on reduced features (step-1-winning transform only), then select best stage."""
        self._cv_step3(result, transform_stores)
        return self._select_step3(result)

    def _cv_step3(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]] = None,
    ) -> None:
        """
        Step-3 CV of all models on the reduced features. Metric-independent given the
        step-2 selection: sets step3_* fields (and step3_fold_features when per-fold).
        """
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
        cv_store, fold_columns = reduced_store, None
        if self.config.select_features_in_folds:
            # Score the selection procedure, not a set chosen on every row: each fold
            # runs on the full features with the subset selected on its training rows.
            cv_store = self._full_store(result, transform_stores)
            if result.splitter is None:  # not persisted — rebuilt as step 1 built it
                result.splitter = self._cv._build_splitter(
                    self.config.task_type,
                    X=cv_store.X,
                    y=cv_store.y,
                    coords=cv_store.coords,
                    spatial=self.config.spatial_cv,
                )
            result.step3_fold_features = self._fold_feature_sets(result, cv_store)
            fold_columns = self._fold_columns(cv_store, result.step3_fold_features)
        cv_results = self._cv.run_all(
            self.models,
            cv_store.X,
            cv_store.y,
            y_true=cv_store.y_true,
            inverse_fn=inverse_fn,
            coords=cv_store.coords,
            spatial=self.config.spatial_cv,
            n_jobs=self.config.n_jobs,
            train_target_fn=get_train_target_fn(cv_store.y_transform),
            fold_columns=fold_columns,
            _splitter=result.splitter,
        )
        result.step3_reduced_stores = {transform_name: reduced_store}
        result.step3_cv_result = cv_results
        result.step3_fold_df = compute_metrics_all(cv_results, metadata=metadata)
        result.step3_agg_df = aggregate_metrics(result.step3_fold_df)
        result.cv_warnings.extend(self._collect_cv_warnings(cv_results))

    def _select_step3(self, result: PipelineResult) -> PipelineResult:
        """Select the best (model, stage) under config.metric from the step-1 and step-3 CVs."""
        # Compare step1 (default) and step3 (reduced) agg dfs directly — pick the best row
        assert result.step1_agg_df is not None and result.step3_agg_df is not None
        all_stages_agg = pd.concat([result.step1_agg_df, result.step3_agg_df], ignore_index=True)
        assert result.step1_fold_df is not None and result.step3_fold_df is not None
        both_stages_fold_df = pd.concat([result.step1_fold_df, result.step3_fold_df], ignore_index=True)
        best = select_best(
            all_stages_agg,
            metric=self.config.metric_col,
            minimize=self.config.minimize_metric,
            n_folds=self.config.n_splits,
            fold_df=both_stages_fold_df,
        )
        robust_df = ranking_robustness(both_stages_fold_df, self.config.metric, n_folds=self.config.n_splits)
        result.cv_warnings.extend(self._check_ranking_robustness(robust_df, step=3))
        best_row = best["row"]
        if self.config.prefer_reduced_on_tie:
            best_row = self._prefer_reduced_on_tie(result, robust_df, all_stages_agg, best_row)

        result.best_model_name = best_row["Model"]
        result.best_model_value = float(best_row[self.config.metric_col])
        std_col = self.config.metric_col.replace("_Mean", "_Std")
        result.best_model_std = float(best_row.get(std_col, float("nan")))
        result.best_stage = str(best_row["Stage"])
        result.best_feature_stage = result.best_stage  # preserved — never overwritten by "optimized"
        result.y_transform = best_row.get("Y_transform")
        self._record_best_estimator(result)

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
        tuned_params, fixed_params = self._tune_step4(result, transform_stores)
        return self._select_step4(result, tuned_params, fixed_params)

    def _tune_step4(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]] = None,
    ) -> tuple[Optional[dict], dict]:
        """
        Optuna HPO of the step-3 winner under config.metric, then the acceptance CV with
        the tuned params (sets step4_* fields).

        Returns:
            (tuned_params, fixed_params): the tuned overrides merged with the fixed ones, or
            None when the winner has no enabled search space; and the fixed params
            (class_weight) every candidate carries.
        """
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
        train_target_fn = get_train_target_fn(store_for_opt.y_transform)

        best_model = self._get_model(result.best_model_name)
        estimator_name = best_model.estimator.__class__.__name__
        # Re-record: a resumed run (run_step4_only on a new pipeline) tunes this
        # pipeline's model, which the final model must then be rebuilt from.
        self._record_best_estimator(result)

        # Fixed params are injected into every Optuna trial alongside searched params.
        # Built before the opt_enabled check: the skip path must also carry them, or a
        # deployed opt-disabled winner would silently lose class_weight="balanced"
        # even though CV ranked it with weights injected.
        fixed_params: dict = {}
        if self._class_weight_active and self._supports_class_weight(estimator_name):
            fixed_params["class_weight"] = "balanced"

        if not self._is_opt_enabled(estimator_name):
            self._log(f"  {estimator_name} has no enabled search space — skipping optimization.")
            return None, fixed_params

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
            train_target_fn=train_target_fn,
            _splitter=result.splitter if store_for_opt.coords is not None else None,
            estimator=best_model.estimator,
            requires_scaling=best_model.requires_scaling,
        )

        # best_params are overrides on top of the winner's own estimator — trials,
        # the final CV (_build_fold_model) and build_final_model all apply them via
        # clone(estimator).set_params(), so constructor args (e.g. probability=True
        # for SVC, or a user's max_iter) are kept everywhere.
        # fixed_params are re-applied last: every trial was evaluated with them
        # overriding sampled values (see _build_objective), so the final model must
        # match — otherwise a sampled class_weight=None from the search space would
        # silently undo handle_imbalance in the deployed params.
        merged_params = {**study.best_params, **fixed_params}

        metadata = self._metadata_with(stage="optimized")
        if result.y_transform:
            metadata = replace(metadata, y_transform=result.y_transform)
        # With per-fold feature sets the reduced stage's step-3 score came from them, so
        # the acceptance CV must use them too — otherwise the tuned params would be
        # compared against a baseline estimated a different way.
        fold_columns = None
        if result.step3_fold_features is not None and result.best_feature_stage == "reduced":
            full_store = self._full_store(result, transform_stores)
            X = full_store.X
            fold_columns = self._fold_columns(full_store, result.step3_fold_features)
        cv_result = self._cv.run(
            best_model,
            X,
            y,
            best_params=merged_params,
            y_true=y_true,
            inverse_fn=inverse_fn,
            coords=store_for_opt.coords,
            spatial=self.config.spatial_cv,
            train_target_fn=train_target_fn,
            fold_columns=fold_columns,
            _splitter=result.splitter,
        )
        result.step4_cv_result = cv_result
        result.step4_fold_df = compute_metrics_all([cv_result], metadata=metadata)
        result.step4_agg_df = aggregate_metrics(result.step4_fold_df, group_by_stage=False)
        return merged_params, fixed_params

    def _select_step4(
        self,
        result: PipelineResult,
        tuned_params: Optional[dict],
        fixed_params: dict,
    ) -> PipelineResult:
        """Keep the tuned params when their acceptance CV (step4_agg_df) beats the step-3
        winner under config.metric; otherwise the winner's defaults (plus fixed_params)."""
        if tuned_params is None:
            result.best_params = dict(fixed_params)
            return result
        assert result.step4_agg_df is not None and result.best_model_value is not None
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
            result.best_params = dict(tuned_params)
        else:
            result.best_params = dict(fixed_params)

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
        full_store = self._full_store(result, transform_stores)

        if result.best_stage == "reduced":
            assert result.selector is not None
            key = result.y_transform or ""
            if result.step3_reduced_stores is not None and key in result.step3_reduced_stores:
                return result.step3_reduced_stores[key]
            return result.selector.transform(full_store)
        return full_store

    def _full_store(
        self,
        result: PipelineResult,
        transform_stores: Optional[dict[str, PipelineData]],
    ) -> PipelineData:
        """All-features store in the winning y-space (the store step 2 selected on)."""
        if transform_stores and result.y_transform:
            return transform_stores[result.y_transform]
        assert result.features is not None
        return result.features

    def _make_selector(self, model_name: Optional[str], store: PipelineData, splitter: Any) -> FeatureSelector:
        """Step-2 FeatureSelector for model_name; splitter drives its OOF SHAP folds."""
        return FeatureSelector(
            step=self._get_model(model_name),
            task_type=self.config.task_type,
            corr_threshold=self.config.corr_threshold,
            min_features=self.config.min_features,
            use_oof=True,
            n_splits=self.config.n_splits,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            train_target_fn=get_train_target_fn(store.y_transform),
            n_jobs=self.config.n_jobs,
            max_vif=self.config.max_vif,
            _splitter=splitter,
        )

    def _fold_feature_sets(self, result: PipelineResult, store: PipelineData) -> list[list[str]]:
        """
        Re-run the step-2 selector on each CV fold's training rows.

        Spatial runs split each training fold along the outer splitter's blocks
        (BlockGroupSplitter), so the inner OOF SHAP folds stay spatially separated;
        otherwise the selector's own stratified/plain KFold is used.
        """
        block_ids = getattr(result.splitter, "block_id_", None)
        fold_sets: list[list[str]] = []
        for k, (train_idx, _) in enumerate(result.splitter.split(store.X, store.y)):
            inner = None
            if block_ids is not None:
                n_blocks = np.unique(block_ids[train_idx]).size
                if n_blocks >= 2:
                    inner = BlockGroupSplitter(
                        block_ids[train_idx],
                        n_splits=min(self.config.n_splits, n_blocks),
                        random_state=self.config.random_state,
                    )
            selector = self._make_selector(result.best_model_name, store, inner)
            selector.fit(store.take(train_idx))
            fold_sets.append(list(selector.selected_features_))
            self._log(f"  Fold {k}: {selector.n_selected}/{store.n_features} features selected on its training rows")
        return fold_sets

    @staticmethod
    def _fold_columns(store: PipelineData, fold_features: list[list[str]]) -> list[np.ndarray]:
        """Column indices into store.X for each fold's feature names."""
        index = {name: i for i, name in enumerate(store.feature_names)}
        return [np.array([index[f] for f in names]) for names in fold_features]

    def _start(self, store: PipelineData, log_class_weights: bool = True) -> tuple[PipelineResult, PipelineData]:
        """Validate store, set aside the holdout rows, and return a fresh result plus the dev rows."""
        self._validate_store(store)
        holdout_store, holdout_idx, holdout_strategy = None, None, None
        if self.config.holdout_frac > 0:
            dev_idx, holdout_idx, holdout_strategy = split_holdout(
                store,
                self.config.task_type,
                frac=self.config.holdout_frac,
                strategy=self.config.holdout_strategy,
                random_state=self.config.random_state,
                spatial=self.config.spatial_cv,
            )
            holdout_store = store.take(holdout_idx)
            store = store.take(dev_idx)
            self._log(
                f"Holdout ({holdout_strategy}): {holdout_idx.size}/{holdout_idx.size + dev_idx.size} rows set aside"
            )
        self._resolve_class_weights(store.y, log=log_class_weights)
        result = self._new_result(store)
        result.holdout_store = holdout_store
        result.holdout_idx = holdout_idx
        result.holdout_strategy = holdout_strategy
        return result, store

    def _new_result(self, store: PipelineData) -> PipelineResult:
        """Build a fresh PipelineResult seeded with store and CV metadata from config."""
        return PipelineResult(
            features=store,
            cv_type="spatial" if store.coords is not None else "random",
            spatial_cv_metric=self.config.spatial_cv_metric,
            time_bin_resolution=self.config.time_bin_resolution,
            class_weight_balanced=self._class_weight_active,
            holdout_refit=self.config.holdout_refit,
        )

    def _score_holdout(self, result: PipelineResult) -> PipelineResult:
        """
        Fit the final configuration on the dev rows and score it once on the holdout rows.

        Reuses CrossValidator.run with a one-fold split over dev + holdout, so scaling,
        per-fold winsorize clipping and inverse transforms match the CV runs exactly.
        No-op without a holdout. Adds a [holdout] warning when the holdout score is worse
        than the CV score by more than two fold standard deviations, and (classification)
        when the holdout holds fewer than MIN_HOLDOUT_MINORITY rows of its rarest class.
        """
        if result.holdout_store is None:
            return result
        assert result.features is not None and result.best_model_value is not None
        feature_stage = result.best_feature_stage or result.best_stage
        reduced_names = (
            result.features_reduced.feature_names
            if feature_stage == "reduced" and result.features_reduced is not None
            else None
        )
        store = stage_store(
            concat_stores(result.features, result.holdout_store),
            result.y_transform,
            reduced_names,
        )
        cv_result = self._cv.run(
            self._get_model(result.best_model_name),
            store.X,
            store.y,
            best_params=result.best_params,
            y_true=store.y_true,
            inverse_fn=self._get_inverse_fn(store),
            train_target_fn=get_train_target_fn(store.y_transform),
            _splitter=FixedSplit(result.features.n_samples, store.n_samples),
        )
        if not cv_result.folds:
            result.cv_warnings.append("[holdout] Fitting the final configuration on the dev rows failed.")
            return result

        result.holdout_cv_result = cv_result
        metadata = self._metadata_with(stage="holdout")
        if result.y_transform:
            metadata = replace(metadata, y_transform=result.y_transform)
        result.holdout_df = aggregate_metrics(compute_metrics_all([cv_result], metadata=metadata), group_by_stage=False)
        if self.config.task_type == TaskType.CLASSIFICATION:
            holdout_y = result.holdout_store.y
            n_minority = min(int(np.sum(holdout_y == c)) for c in np.unique(store.y))
            if n_minority < MIN_HOLDOUT_MINORITY:
                result.cv_warnings.append(
                    f"[holdout] The holdout holds only {n_minority} row(s) of its rarest class, so its "
                    f"score and the conformal calibration built on it are unreliable. Consider "
                    "holdout_frac=0 for this target (CV scores and CV-residual conformal) or a larger holdout_frac."
                )

        holdout_value = float(result.holdout_df.iloc[0][self.config.metric_col])
        cv_value, cv_std = result.best_model_value, result.best_model_std
        self._log(f"  Holdout {self.config.metric}={holdout_value:.4f} (CV {cv_value:.4f})")
        if cv_std is not None and np.isfinite(cv_std):
            gap = holdout_value - cv_value if self.config.minimize_metric else cv_value - holdout_value
            if gap > 2 * cv_std:
                result.cv_warnings.append(
                    f"[holdout] {self.config.metric} on the holdout ({holdout_value:.4f}) is worse than the "
                    f"CV estimate ({cv_value:.4f} ± {cv_std:.4f}) by more than 2 fold SDs — the CV score "
                    "is likely optimistic (selection bias) or the holdout differs from the dev rows."
                )
        return result

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
            if minority_ratio < IMBALANCE_RATIO_THRESHOLD:
                suggestion = (
                    "Consider setting handle_imbalance=True or 'auto' in PipelineConfig."
                    if not self.config.handle_imbalance
                    else f"handle_imbalance={self.config.handle_imbalance!r} is active — "
                    "class_weight='balanced' will be applied."
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

    def _check_metric_suitability(self, y: np.ndarray, fold_indices: list[tuple[np.ndarray, np.ndarray]]) -> list[str]:
        """Warn when the data makes config.metric an unreliable selection criterion.

        Advisory only — the metric encodes how predictions will be used, which the data
        cannot tell, so it is never changed. Only conditions that affect the chosen metric
        are reported. y is the original-scale target; fold_indices are the step-1 folds.
        """
        metric = self.config.metric
        test_ys = [y[te] for _, te in fold_indices]
        msgs: list[str] = []

        if self.config.task_type == TaskType.CLASSIFICATION:
            classes, counts = np.unique(y, return_counts=True)
            minority_ratio = counts.min() / counts.sum()
            minority = classes[counts.argmin()]
            if metric == "AUC" and minority_ratio < IMBALANCE_RATIO_THRESHOLD:
                msgs.append(
                    f"AUC with minority class ratio={minority_ratio:.3f}: AUC can look optimistic "
                    "when negatives dominate. Consider 'AUCPR' or 'Brier'."
                )
            if metric in ("AUC", "AUCPR"):
                n_single = sum(len(np.unique(yt)) < 2 for yt in test_ys)
                if n_single:
                    msgs.append(
                        f"{metric}: {n_single}/{len(test_ys)} test fold(s) hold a single class, so "
                        f"{metric} is NaN there and the mean uses fewer folds. Consider 'Brier' or 'LogLoss'."
                    )
            if metric == "AUCPR":
                shares = [np.mean(yt == minority) for yt in test_ys]
                if max(shares) > PREVALENCE_SPREAD_RATIO * min(shares):
                    msgs.append(
                        f"AUCPR: minority-class share varies across test folds "
                        f"({min(shares):.3f}-{max(shares):.3f}); AUCPR's baseline is the prevalence, "
                        "so fold scores are not comparable. Check the fold std, or consider 'Brier'."
                    )
            if metric == "F1" and self._class_weight_active:
                msgs.append(
                    "F1 with class_weight='balanced': weighting shifts the effective threshold, so F1 "
                    "partly ranks thresholds rather than models. Consider a threshold-free metric."
                )
        else:
            if metric == "R2":
                total_var = np.var(y)
                n_low = sum(np.var(yt) < LOW_FOLD_VARIANCE_RATIO * total_var for yt in test_ys)
                if n_low:
                    msgs.append(
                        f"R2: {n_low}/{len(test_ys)} test fold(s) have y variance below "
                        f"{LOW_FOLD_VARIANCE_RATIO:.0%} of the overall variance, where R2 is unstable "
                        "and can be strongly negative. Consider 'RMSE' or 'MAE'."
                    )
            if metric == "RMSE":
                sq_dev = np.sort((y - y.mean()) ** 2)[::-1]
                n_top = max(1, int(np.ceil(0.01 * len(y))))
                tail_share = sq_dev[:n_top].sum() / sq_dev.sum() if sq_dev.sum() > 0 else 0.0
                if tail_share > TAIL_SSD_SHARE:
                    msgs.append(
                        f"RMSE: the top 1% of observations carry {tail_share:.0%} of the squared "
                        "deviation, so a few points dominate RMSE. Consider 'MAE' or a winsorized transform."
                    )
            if metric == "MAE":
                zero_share = np.mean(y == 0)
                if zero_share >= 0.5:
                    msgs.append(
                        f"MAE with {zero_share:.0%} zeros in y: the median is 0, so MAE rewards "
                        "predicting ~0 everywhere. Consider 'RMSE' or a delta (hurdle) model."
                    )

        for m in msgs:
            logger.warning(f"[metric] {m}")
        return msgs

    def _check_ranking_robustness(self, robust_df: pd.DataFrame, step: int) -> list[str]:
        """Warn when the winner is a paired-fold tie or loses on the counterpart-family metric.

        Advisory only — see h2ml.evaluation.ranking. Only prefer_reduced_on_tie (step 3)
        acts on these ties; the warnings themselves never change the selection.
        """
        msgs = [f"step {step}: {m}" for m in ranking_warnings(robust_df, self.config.metric)]
        for m in msgs:
            logger.warning(f"[ranking] {m}")
        return msgs

    def _prefer_reduced_on_tie(
        self,
        result: PipelineResult,
        robust_df: pd.DataFrame,
        all_stages_agg: pd.DataFrame,
        best_row: pd.Series,
    ) -> pd.Series:
        """1-SE rule: swap a default-stage winner for the best-ranked tied reduced candidate.

        "Tied" is the Indistinguishable flag of ranking_robustness (paired-fold gap within
        one corrected SE). Returns best_row unchanged when the winner is already reduced,
        the reduced set kept every feature (stages identical), or no reduced candidate ties.
        Otherwise records the switch in cv_warnings and returns the reduced agg row.
        """
        if best_row["Stage"] != "default":
            return best_row
        assert result.features is not None and result.features_reduced is not None
        if result.features_reduced.n_features == result.features.n_features:
            return best_row
        tied = robust_df.loc[(robust_df["Stage"] == "reduced") & robust_df["Indistinguishable"]]
        if tied.empty:
            return best_row
        pick = tied.iloc[0]  # robust_df is sorted by rank under the selection metric
        keys = [c for c in KEY_COLS if c in robust_df.columns]
        new_row = all_stages_agg.loc[(all_stages_agg[keys] == pick[keys]).all(axis=1)].iloc[0]
        msg = (
            f"step 3: prefer_reduced_on_tie → {pick['Model']} | reduced "
            f"(tied with {best_row['Model']} | default: gap {pick['Winner_Gap']:.4g} ≤ SE {pick['Gap_SE']:.4g})"
        )
        logger.info(msg)
        result.cv_warnings.append(msg)
        return new_row

    def _get_inverse_fn(self, store: PipelineData):
        """Return the inverse transform callable for the store's y_transform, or None."""
        if store.y_transform is None:
            return None
        from h2ml.preprocessing.transforms import INVERSE_TRANSFORMS

        return INVERSE_TRANSFORMS.get(store.y_transform)

    def _is_opt_enabled(self, estimator_name: str) -> bool:
        """Return True only if the model has a registry entry that run_study can tune.

        Mirrors get_entry(): an estimator outside the registry, or one with
        opt_enabled=False or no param_fn, has no search space and is skipped.
        """
        from h2ml.utils.registry import CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY

        registry = CLASSIFIER_REGISTRY if self.config.task_type == TaskType.CLASSIFICATION else REGRESSOR_REGISTRY
        entry = registry.get(estimator_name)
        return entry is not None and entry.opt_enabled and entry.param_fn is not None

    def _supports_class_weight(self, estimator_name: str) -> bool:
        """Return True if the classifier's registry entry has supports_class_weight=True."""
        from h2ml.utils.registry import CLASSIFIER_REGISTRY

        entry = CLASSIFIER_REGISTRY.get(estimator_name)
        return entry.supports_class_weight if entry is not None else False

    def _inject_class_weights(self, class_weight: Optional[str] = "balanced") -> None:
        """Set class_weight on every supporting classifier in self.models.

        Looked up by estimator class name — the same key step 4 uses — so a model
        given a custom display name (e.g. name="LR") is weighted consistently.
        """
        injected = []
        for model in self.models:
            if self._supports_class_weight(model.estimator.__class__.__name__):
                model.estimator.set_params(class_weight=class_weight)
                injected.append(model.name)
        if injected:
            logger.debug(f"handle_imbalance: class_weight={class_weight!r} set on {injected}")

    def _resolve_class_weights(self, y: np.ndarray, log: bool = True) -> None:
        """Resolve handle_imbalance="auto" against the run's target.

        Injects class_weight="balanced" when the minority-class ratio is below
        IMBALANCE_RATIO_THRESHOLD. If an earlier run on this pipeline injected
        weights and this target is balanced, they are reset to None so a reused
        pipeline does not carry a stale decision. No-op unless "auto" + classification.
        log=False leaves the decision unlogged (multi_metric logs one line for all metrics).
        """
        if self.config.handle_imbalance != "auto" or self.config.task_type != TaskType.CLASSIFICATION:
            return
        _, counts = np.unique(y, return_counts=True)
        minority_ratio = counts.min() / counts.sum()
        active = bool(minority_ratio < IMBALANCE_RATIO_THRESHOLD)
        if active:
            self._inject_class_weights("balanced")
        elif self._class_weight_active:
            self._inject_class_weights(None)
        self._class_weight_active = active
        if not log:
            return
        logger.info(
            f"handle_imbalance='auto': minority class ratio={minority_ratio:.3f} → "
            f"class_weight={'balanced' if active else 'not applied'}"
        )

    def _record_best_estimator(self, result: PipelineResult) -> None:
        """Store an unfitted clone of the winning model's estimator on result.

        build_final_model refits from this instead of rebuilding from the registry,
        so custom display names, non-registry estimators and constructor args all
        survive into the deployed model.
        """
        from sklearn.base import clone

        model = self._get_model(result.best_model_name)
        result.best_base_estimator = clone(model.estimator)
        result.best_requires_scaling = model.requires_scaling

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
