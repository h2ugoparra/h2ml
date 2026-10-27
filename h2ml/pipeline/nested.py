"""
h2ml/pipeline/nested.py

Nested cross-validation of the whole h2ml procedure: the full 4-step pipeline
runs inside each outer training fold and its final configuration is scored on
the outer test fold. Estimates how well the *procedure* generalises and how
stable its choices are; it does not produce a deployable model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

from h2ml.core.feature_store import PipelineData
from h2ml.evaluation.metrics import METRIC_COL, METRIC_MINIMIZE, compute_metrics
from h2ml.features.spatial_cv import build_splitter
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineResult


@dataclass
class NestedCVResult:
    """
    Output of nested_cv().

    Attributes:
        metric:            Short selection metric name, e.g. "AUC" or "R2".
        folds_df:          One row per completed outer fold: the inner pipeline's choice
                           (Model, Y_transform, Stage, Feature_Stage, N_features), its
                           inner CV estimate (Inner_<metric>, Inner_<metric>_Std), and
                           the outer test metrics (<metric>_Test, _Train, _Gap columns).
        selected_features: Feature names the final configuration used, per completed fold.
        results:           Inner PipelineResult per completed fold, with the outer test
                           fold attached as its holdout.
        failed_folds:      Outer fold ids whose inner run or scoring raised.
        cv_type:           "spatial" when outer folds were spatial, else "random".
    """

    metric: str
    folds_df: pd.DataFrame
    selected_features: list[list[str]]
    results: list[PipelineResult] = field(repr=False)
    failed_folds: list[int] = field(default_factory=list)
    cv_type: str = "random"

    @property
    def n_folds(self) -> int:
        """Number of completed outer folds."""
        return len(self.folds_df)

    def summary(self) -> pd.DataFrame:
        """
        One-row summary of the procedure's generalisation.

        Columns:
            Outer_<metric>_Mean / _Std: mean and std of the outer test score — the
                unbiased estimate of what running the pipeline on new data yields.
            Inner_<metric>_Mean: mean of the inner CV estimates the pipeline reported.
            Optimism_Mean: mean of (inner − outer), sign-adjusted so positive always
                means the inner CV estimate was too optimistic.
            N_folds, N_distinct_selections: completed folds and distinct
                (model, transform, stage) choices across them.
        """
        test_col = METRIC_COL[self.metric].replace("_Mean", "")
        outer = self.folds_df[test_col]
        inner = self.folds_df[f"Inner_{self.metric}"]
        optimism = (outer - inner) if METRIC_MINIMIZE[self.metric] else (inner - outer)
        return pd.DataFrame(
            [
                {
                    f"Outer_{self.metric}_Mean": outer.mean(),
                    f"Outer_{self.metric}_Std": outer.std(),
                    f"Inner_{self.metric}_Mean": inner.mean(),
                    "Optimism_Mean": optimism.mean(),
                    "N_folds": self.n_folds,
                    "N_distinct_selections": len(self.selection_counts()),
                }
            ]
        )

    def selection_counts(self) -> pd.DataFrame:
        """How often each (Model, Y_transform, Stage) was selected, most frequent first."""
        keys = ["Model", "Y_transform", "Stage"]
        counts = self.folds_df.fillna({"Y_transform": ""}).groupby(keys).size().reset_index(name="Count")
        counts["Fraction"] = counts["Count"] / self.n_folds
        return counts.sort_values("Count", ascending=False, kind="stable").reset_index(drop=True)

    def feature_frequency(self) -> pd.Series:
        """Fraction of completed folds whose final configuration used each feature."""
        flat = pd.Series([f for names in self.selected_features for f in names], dtype=object)
        return (flat.value_counts() / self.n_folds).rename("Fraction")


def nested_cv(
    pipeline: H2MLPipeline,
    store: PipelineData,
    outer_splits: int = 5,
    transforms: Optional[Iterable[str]] = None,
) -> NestedCVResult:
    """
    Run the full pipeline inside each outer fold and score its final configuration
    on the outer test fold.

    Outer folds use the pipeline's own splitter rules — spatial blocks when
    store.coords is set (PipelineConfig spatial-CV settings), else stratified
    (classification) or plain KFold, seeded by config.random_state. Each inner run
    builds its own splitter on the outer training rows, so no selection step ever
    sees the outer test rows. Outer folds run serially; config.n_jobs applies
    inside each inner run. Cost is about outer_splits × one pipeline.run().

    The pipeline's holdout_frac is ignored inside the loop — the outer test fold
    plays that role. A fold whose inner run raises is logged and skipped.

    Args:
        pipeline:     Configured H2MLPipeline (models, config, metadata are reused).
        store:        Full dataset.
        outer_splits: Number of outer folds (>= 2).
        transforms:   y-transform names swept inside each inner run (regression only).

    Returns:
        NestedCVResult.

    Raises:
        ValueError: If outer_splits < 2.
        RuntimeError: If every outer fold failed.
    """
    if outer_splits < 2:
        raise ValueError(f"outer_splits must be >= 2, got {outer_splits}")
    config = pipeline.config
    pipeline._validate_store(store)
    if config.holdout_frac > 0:
        logger.info("nested_cv: holdout_frac is ignored — each outer test fold acts as the holdout.")
        pipeline = H2MLPipeline(replace(config, holdout_frac=0.0), models=pipeline.models, metadata=pipeline.metadata)

    splitter = build_splitter(
        config.task_type,
        n_splits=outer_splits,
        random_state=config.random_state,
        coords=store.coords,
        X=store.X,
        y=store.y,
        spatial=config.spatial_cv,
    )
    cv_type = "spatial" if store.coords is not None else "random"
    transforms = list(transforms) if transforms is not None else None

    rows: list[dict] = []
    selected_features: list[list[str]] = []
    results: list[PipelineResult] = []
    failed: list[int] = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(store.X, store.y)):
        if test_idx.size == 0:
            logger.warning(f"nested_cv: outer fold {fold_id} skipped — empty test set.")
            failed.append(fold_id)
            continue
        if config.verbose:
            logger.info(f"nested_cv: outer fold {fold_id + 1}/{outer_splits} ({test_idx.size} test rows)")
        try:
            result = pipeline.run(store.take(train_idx), transforms=transforms)
            result.holdout_store = store.take(test_idx)
            result.holdout_idx = test_idx
            result.holdout_strategy = cv_type
            result = pipeline._score_holdout(result)
            if result.holdout_cv_result is None:
                raise RuntimeError("scoring the final configuration on the outer test fold failed")
        except Exception as e:
            logger.warning(f"nested_cv: outer fold {fold_id} failed — {e}")
            failed.append(fold_id)
            continue

        assert result.features is not None
        feature_stage = result.best_feature_stage or result.best_stage
        names = (
            result.features_reduced.feature_names
            if feature_stage == "reduced" and result.features_reduced is not None
            else result.features.feature_names
        )
        outer_metrics = compute_metrics(result.holdout_cv_result).iloc[0].drop(["Model", "Fold"])
        rows.append(
            {
                "Outer_Fold": fold_id,
                "Model": result.best_model_name,
                "Y_transform": result.y_transform,
                "Stage": result.best_stage,
                "Feature_Stage": feature_stage,
                "N_features": len(names),
                "N_test": int(test_idx.size),
                f"Inner_{config.metric}": result.best_model_value,
                f"Inner_{config.metric}_Std": result.best_model_std,
                **outer_metrics.to_dict(),
            }
        )
        selected_features.append(list(names))
        results.append(result)

    if not rows:
        raise RuntimeError(f"nested_cv: all {outer_splits} outer folds failed.")
    return NestedCVResult(
        metric=config.metric,
        folds_df=pd.DataFrame(rows),
        selected_features=selected_features,
        results=results,
        failed_folds=failed,
        cv_type=cv_type,
    )
