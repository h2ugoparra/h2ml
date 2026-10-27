"""
h2ml/pipeline/stability.py

Seed stability: rerun the pipeline under several random seeds and measure how
much its choices and its reported score depend on that one arbitrary number.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Literal, Optional

import pandas as pd
from loguru import logger

from h2ml.core.feature_store import PipelineData
from h2ml.evaluation.metrics import METRIC_COL
from h2ml.pipeline.nested import feature_frequency, final_feature_names, selection_counts
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineResult


@dataclass
class SeedStabilityResult:
    """
    Output of seed_stability().

    Attributes:
        metric:            Short selection metric name, e.g. "AUC" or "R2".
        runs_df:           One row per completed seed: Seed, Model, Y_transform, Stage,
                           N_features, the reported CV score (<metric>, <metric>_Std) and,
                           when a holdout was scored, Holdout_<metric>.
        selected_features: Feature names each seed's final configuration used.
        results:           PipelineResult per completed seed.
        failed_seeds:      Seeds whose run raised.
    """

    metric: str
    runs_df: pd.DataFrame
    selected_features: list[list[str]]
    results: list[PipelineResult] = field(repr=False)
    failed_seeds: list[int] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        """Number of completed seeds."""
        return len(self.runs_df)

    def summary(self) -> pd.DataFrame:
        """
        One-row summary across seeds.

        Columns:
            <metric>_Mean / _Std / _Min / _Max: spread of the reported CV score — how much
                of a difference between two runs could be the seed alone.
            Holdout_<metric>_Mean / _Std: same for the holdout score, when scored.
            Top_Selection, Top_Selection_Fraction: most frequent (model, transform, stage)
                and the share of seeds that picked it.
            N_runs, N_distinct_selections.
        """
        score = self.runs_df[self.metric]
        top = self.selection_counts().iloc[0]
        row = {
            f"{self.metric}_Mean": score.mean(),
            f"{self.metric}_Std": score.std(),
            f"{self.metric}_Min": score.min(),
            f"{self.metric}_Max": score.max(),
        }
        holdout_col = f"Holdout_{self.metric}"
        if holdout_col in self.runs_df:
            row[f"{holdout_col}_Mean"] = self.runs_df[holdout_col].mean()
            row[f"{holdout_col}_Std"] = self.runs_df[holdout_col].std()
        transform = f" | {top['Y_transform']}" if top["Y_transform"] else ""
        row.update(
            {
                "Top_Selection": f"{top['Model']}{transform} | {top['Stage']}",
                "Top_Selection_Fraction": top["Fraction"],
                "N_runs": self.n_runs,
                "N_distinct_selections": len(self.selection_counts()),
            }
        )
        return pd.DataFrame([row])

    def selection_counts(self) -> pd.DataFrame:
        """How often each (Model, Y_transform, Stage) was selected, most frequent first."""
        return selection_counts(self.runs_df)

    def feature_frequency(self) -> pd.Series:
        """Fraction of seeds whose final configuration used each feature."""
        return feature_frequency(self.selected_features)


def seed_stability(
    pipeline: H2MLPipeline,
    store: PipelineData,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    transforms: Optional[Iterable[str]] = None,
    through_step: Literal[1, 3, 4] = 3,
) -> SeedStabilityResult:
    """
    Rerun the pipeline with config.random_state set to each seed.

    The seed drives the CV fold split (and the spatial partition), SHAP sampling,
    the Optuna sampler and the holdout split. Seeds fixed inside the model
    constructors (e.g. a registry RandomForest's random_state) are not changed —
    the spread therefore isolates the evaluation and selection procedure.

    A winner picked by only some seeds, or a score spread comparable to the gaps
    between candidates, means one run's choice is partly luck: compare candidates
    with ranking_robustness() and prefer simpler ones among those tied.

    Args:
        pipeline:     Configured H2MLPipeline (models, config, metadata are reused).
        store:        Full dataset.
        seeds:        random_state values to run (at least two).
        transforms:   y-transform names swept in each run (regression only).
        through_step: 1 — model screening only (run_step1_only); 3 — selection without
                      HPO (run_step1_to_step3, default); 4 — the full run().

    Returns:
        SeedStabilityResult.

    Raises:
        ValueError: If fewer than two distinct seeds are given or through_step is invalid.
        RuntimeError: If every seed's run failed.
    """
    seeds = list(dict.fromkeys(seeds))
    if len(seeds) < 2:
        raise ValueError(f"seeds must hold at least two distinct values, got {seeds}")
    if through_step not in (1, 3, 4):
        raise ValueError(f"through_step must be 1, 3 or 4, got {through_step!r}")
    transforms = list(transforms) if transforms is not None else None
    metric = pipeline.config.metric

    rows: list[dict] = []
    selected_features: list[list[str]] = []
    results: list[PipelineResult] = []
    failed: list[int] = []
    for seed in seeds:
        run_pipeline = H2MLPipeline(
            replace(pipeline.config, random_state=seed), models=pipeline.models, metadata=pipeline.metadata
        )
        if pipeline.config.verbose:
            logger.info(f"seed_stability: seed {seed}")
        try:
            if through_step == 1:
                result = run_pipeline.run_step1_only(store, transforms=transforms)
            elif through_step == 3:
                result = run_pipeline.run_step1_to_step3(store, transforms=transforms)
            else:
                result = run_pipeline.run(store, transforms=transforms)
        except Exception as e:
            logger.warning(f"seed_stability: seed {seed} failed — {e}")
            failed.append(seed)
            continue

        names = final_feature_names(result)
        row = {
            "Seed": seed,
            "Model": result.best_model_name,
            "Y_transform": result.y_transform,
            "Stage": result.best_stage or "default",
            "N_features": len(names),
            metric: result.best_model_value,
            f"{metric}_Std": result.best_model_std,
        }
        if result.holdout_df is not None:
            row[f"Holdout_{metric}"] = float(result.holdout_df.iloc[0][METRIC_COL[metric]])
        rows.append(row)
        selected_features.append(names)
        results.append(result)

    if not rows:
        raise RuntimeError(f"seed_stability: all {len(seeds)} seeds failed.")
    return SeedStabilityResult(
        metric=metric,
        runs_df=pd.DataFrame(rows),
        selected_features=selected_features,
        results=results,
        failed_seeds=failed,
    )
