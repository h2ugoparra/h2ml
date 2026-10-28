"""
h2ml/pipeline/multi_metric.py

Multi-metric comparison: run the pipeline under several selection metrics while
running each metric-independent CV only once.

Steps 1 and 3 CV every model and compute every metric; step 2 depends only on
the step-1 winner (and the class weighting). Those are shared across metrics, and
only the selections — and step 4's metric-specific HPO — run per metric. With
reuse_step4="none" every metric's PipelineResult equals a separate run() with
that metric.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Literal, Mapping, Optional

import numpy as np
import pandas as pd
from loguru import logger

from h2ml.core.base import TaskType
from h2ml.core.feature_store import PipelineData
from h2ml.evaluation.metrics import METRIC_COL, resolve_metric
from h2ml.optimization.optimizer import CLF_METRICS, REG_METRICS
from h2ml.pipeline.nested import feature_frequency, final_feature_names, selection_counts
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineResult

TASK_METRICS: dict[TaskType, tuple[str, ...]] = {
    TaskType.CLASSIFICATION: tuple(CLF_METRICS),
    TaskType.REGRESSION: tuple(REG_METRICS),
}
REUSE_STEP4 = ("none", "model")

# PipelineResult fields each shared stage writes (see H2MLPipeline._cv_step1 etc.)
_STEP1_FIELDS = ("splitter", "step1_cv_result", "step1_fold_df", "step1_agg_df")
_STEP23_FIELDS = (
    "selector",
    "features_reduced",
    "step3_cv_result",
    "step3_fold_df",
    "step3_agg_df",
    "step3_reduced_stores",
    "step3_fold_features",
)
_STEP4_FIELDS = ("step4_cv_result", "step4_fold_df", "step4_agg_df")


@dataclass
class _Shared:
    """Output of one shared stage: the fields it set and the cv_warnings it added."""

    fields: dict
    warnings: list[str]
    extra: tuple = ()

    @classmethod
    def capture(cls, result: PipelineResult, names: tuple[str, ...], n_warnings: int, extra: tuple = ()) -> "_Shared":
        return cls({n: getattr(result, n) for n in names}, result.cv_warnings[n_warnings:], extra)

    def apply(self, result: PipelineResult) -> None:
        for name, value in self.fields.items():
            setattr(result, name, value)
        result.cv_warnings.extend(self.warnings)


@dataclass
class MultiMetricResult:
    """
    Output of multi_metric().

    Attributes:
        metrics:           Metrics in the order they ran.
        runs_df:           One row per completed metric: Metric, Model, Y_transform, Stage,
                           Feature_stage, N_features, Class_weight, the CV score (CV, CV_Std),
                           Holdout (when scored) — each under that row's metric — and
                           Step4_shared_with (the metric whose tuning was reused, else None).
        selected_features: Feature names each metric's final configuration used.
        results:           PipelineResult per completed metric.
        failed_metrics:    Metrics whose run raised.
        reuse_step4:       "none" or "model" (see multi_metric).
    """

    metrics: list[str]
    runs_df: pd.DataFrame
    selected_features: dict[str, list[str]]
    results: dict[str, PipelineResult] = field(repr=False)
    failed_metrics: list[str] = field(default_factory=list)
    reuse_step4: str = "none"

    def summary(self) -> pd.DataFrame:
        """
        One-row summary across metrics.

        Columns:
            N_metrics, N_distinct_selections: how many metrics completed and how many
                different (model, transform, stage) they chose.
            Top_Selection, Top_Selection_Fraction: most frequent choice and the share of
                metrics that picked it — 1.0 means the choice does not depend on the metric.
            N_hpo_runs: step-4 tunings actually run.
        """
        top = self.selection_counts().iloc[0]
        transform = f" | {top['Y_transform']}" if top["Y_transform"] else ""
        return pd.DataFrame(
            [
                {
                    "N_metrics": len(self.runs_df),
                    "N_distinct_selections": len(self.selection_counts()),
                    "Top_Selection": f"{top['Model']}{transform} | {top['Stage']}",
                    "Top_Selection_Fraction": top["Fraction"],
                    "N_hpo_runs": int(self.runs_df["Step4_shared_with"].isna().sum()),
                }
            ]
        )

    def selection_counts(self) -> pd.DataFrame:
        """How many metrics selected each (Model, Y_transform, Stage), most frequent first."""
        return selection_counts(self.runs_df)

    def feature_frequency(self) -> pd.Series:
        """Fraction of metrics whose final configuration used each feature."""
        return feature_frequency(list(self.selected_features.values()))

    def save(self, path: str | Path) -> None:
        """Save runs_df, the run settings and each metric's PipelineResult (under results/<metric>)."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self.runs_df.to_csv(root / "runs.csv", index=False)
        meta = {"metrics": self.metrics, "failed_metrics": self.failed_metrics, "reuse_step4": self.reuse_step4}
        (root / "multi_metric.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for metric, result in self.results.items():
            result.save(root / "results" / metric)

    @classmethod
    def load(cls, path: str | Path) -> "MultiMetricResult":
        """Reload a MultiMetricResult saved with save(path)."""
        root = Path(path)
        meta = json.loads((root / "multi_metric.json").read_text(encoding="utf-8"))
        runs_df = pd.read_csv(root / "runs.csv", keep_default_na=False, na_values=[""])
        runs_df = runs_df.astype(object).where(runs_df.notna(), None)
        results = {m: PipelineResult.load(str(root / "results" / m)) for m in runs_df["Metric"]}
        return cls(
            metrics=meta["metrics"],
            runs_df=runs_df,
            selected_features={m: final_feature_names(r) for m, r in results.items()},
            results=results,
            failed_metrics=meta["failed_metrics"],
            reuse_step4=meta["reuse_step4"],
        )


def _log_class_weights(pipelines: Mapping[str, H2MLPipeline], y) -> None:
    """One log line for every metric's class-weight decision, logged only if any metric uses "auto"."""
    if not any(p.config.handle_imbalance == "auto" for p in pipelines.values()):
        return
    groups: dict[str, list[str]] = {}
    for m, p in pipelines.items():
        if p._class_weight_active:
            label = "balanced"
        elif p.config.handle_imbalance == "auto":
            label = "not applied"
        else:
            label = "off"
        groups.setdefault(label, []).append(m)
    _, counts = np.unique(y, return_counts=True)
    decisions = "; ".join(f"{', '.join(ms)} → {label}" for label, ms in groups.items())
    logger.info(f"handle_imbalance (minority class ratio={counts.min() / counts.sum():.3f}): {decisions}")


def multi_metric(
    pipeline: H2MLPipeline,
    store: PipelineData,
    metrics: Optional[Iterable[str]] = None,
    transforms: Optional[Iterable[str]] = None,
    handle_imbalance: Optional[Mapping[str, bool | Literal["auto"]]] = None,
    reuse_step4: Literal["none", "model"] = "none",
) -> MultiMetricResult:
    """
    Run the full pipeline under each selection metric, sharing the metric-independent work.

    Shared, run once: the holdout split; step 1's CV per class-weighting setting; steps 2-3
    per (class weighting, step-1 winner, y-transform). Per metric: the step-1 and step-3
    selections, step 4 (Optuna optimises the metric) and the holdout score. Each metric
    runs under a copy of pipeline.config with that metric, so results[m] equals
    H2MLPipeline(replace(config, metric=m)).run(store) when reuse_step4="none".

    Args:
        pipeline:         Configured H2MLPipeline (models, config, metadata are reused;
                          config.metric is ignored). Its models are copied per metric, so
                          one metric's class weighting cannot leak into another's.
        store:            Full dataset.
        metrics:          Metrics to compare, in run order. Defaults to all metrics of the task.
        transforms:       y-transform names swept in step 1 (regression only).
        handle_imbalance: Per-metric override of config.handle_imbalance, e.g.
                          {"Brier": False, "LogLoss": False} to keep probability metrics
                          unweighted. Metrics not listed use the config value.
        reuse_step4:      "none" (default) tunes step 4 for every metric. "model" reuses an
                          earlier metric's tuning when the same (class weighting, model,
                          transform, feature stage) won; whether the tuned params beat
                          step 3 is still decided under each metric. Cheaper, but the
                          params were optimised for the earlier metric.

    Returns:
        MultiMetricResult.

    Raises:
        NotImplementedError: For regression pipelines (not supported yet).
        ValueError: If a metric is unknown or not a metric of the task, metrics is empty,
            handle_imbalance names a metric not compared, or reuse_step4 is invalid.
        RuntimeError: If every metric's run failed.
    """
    config = pipeline.config
    if config.task_type != TaskType.CLASSIFICATION:
        raise NotImplementedError("multi_metric supports classification pipelines only (regression is planned).")
    if reuse_step4 not in REUSE_STEP4:
        raise ValueError(f"reuse_step4 must be one of {REUSE_STEP4}, got {reuse_step4!r}")
    task_metrics = TASK_METRICS[config.task_type]
    metrics = list(dict.fromkeys(resolve_metric(m) for m in (task_metrics if metrics is None else metrics)))
    if not metrics:
        raise ValueError("metrics must hold at least one metric.")
    invalid = [m for m in metrics if m not in task_metrics]
    if invalid:
        raise ValueError(f"metrics {invalid} are not {config.task_type.value} metrics: {list(task_metrics)}")
    overrides = {resolve_metric(m): v for m, v in (handle_imbalance or {}).items()}
    unknown = [m for m in overrides if m not in metrics]
    if unknown:
        raise ValueError(f"handle_imbalance names metrics not compared: {unknown}")

    pipelines = {
        m: _metric_pipeline(
            pipeline, replace(config, metric=m, handle_imbalance=overrides.get(m, config.handle_imbalance))
        )
        for m in metrics
    }
    first = pipelines[metrics[0]]
    start, dev = first._start(store, log_class_weights=False)  # holdout split + validation, once
    transform_stores = first._build_transform_stores(dev, transforms)
    for p in list(pipelines.values())[1:]:
        p._resolve_class_weights(dev.y, log=False)
    _log_class_weights(pipelines, dev.y)

    step1: dict[bool, _Shared] = {}
    step23: dict[tuple, _Shared] = {}
    step4: dict[tuple, tuple[str, _Shared]] = {}
    rows: list[dict] = []
    selected: dict[str, list[str]] = {}
    results: dict[str, PipelineResult] = {}
    failed: list[str] = []
    for metric in metrics:
        p = pipelines[metric]
        if config.verbose:
            logger.info(f"multi_metric: {metric}")
        try:
            result = p._new_result(dev)
            for name in ("holdout_store", "holdout_idx", "holdout_strategy"):
                setattr(result, name, getattr(start, name))

            weighted = p._class_weight_active
            if weighted not in step1:
                n = len(result.cv_warnings)
                fold_indices = p._cv_step1(result, transform_stores)
                step1[weighted] = _Shared.capture(result, _STEP1_FIELDS, n, (fold_indices,))
            else:
                step1[weighted].apply(result)
            p._select_step1(result, *step1[weighted].extra)

            key = (weighted, result.best_model_name, result.y_transform)
            if key not in step23:
                n = len(result.cv_warnings)
                p._run_step2(result, transform_stores)
                p._cv_step3(result, transform_stores)
                step23[key] = _Shared.capture(result, _STEP23_FIELDS, n)
            else:
                step23[key].apply(result)
            p._select_step3(result)

            key4 = (weighted, result.best_model_name, result.y_transform, result.best_feature_stage)
            shared_with = None
            if reuse_step4 == "model" and key4 in step4:
                shared_with, tuned = step4[key4]
                tuned.apply(result)
            else:
                n = len(result.cv_warnings)
                params = p._tune_step4(result, transform_stores)
                tuned = _Shared.capture(result, _STEP4_FIELDS, n, params)
                step4.setdefault(key4, (metric, tuned))
            p._select_step4(result, *tuned.extra)
            p._score_holdout(result)
            p._log_summary(result)
        except Exception as e:
            logger.warning(f"multi_metric: {metric} failed — {e}")
            failed.append(metric)
            continue

        names = final_feature_names(result)
        row = {
            "Metric": metric,
            "Model": result.best_model_name,
            "Y_transform": result.y_transform,
            "Stage": result.best_stage,
            "Feature_stage": result.best_feature_stage,
            "N_features": len(names),
            "Class_weight": result.class_weight_balanced,
            "CV": result.best_model_value,
            "CV_Std": result.best_model_std,
        }
        if result.holdout_df is not None:
            row["Holdout"] = float(result.holdout_df.iloc[0][METRIC_COL[metric]])
        row["Step4_shared_with"] = shared_with
        rows.append(row)
        selected[metric] = names
        results[metric] = result

    if not rows:
        raise RuntimeError(f"multi_metric: all {len(metrics)} metrics failed.")
    return MultiMetricResult(
        metrics=metrics,
        runs_df=pd.DataFrame(rows),
        selected_features=selected,
        results=results,
        failed_metrics=failed,
        reuse_step4=reuse_step4,
    )


def _metric_pipeline(pipeline: H2MLPipeline, config) -> H2MLPipeline:
    """
    A pipeline for one metric with its own copy of the models. _inject_class_weights
    edits estimators in place, so shared models would let one metric's class weighting
    leak into another's; weights the source pipeline injected are undone first.
    """
    metric_pipeline = H2MLPipeline(config, models=copy.deepcopy(pipeline.models), metadata=pipeline.metadata)
    if pipeline._class_weight_active and not metric_pipeline._class_weight_active:
        metric_pipeline._inject_class_weights(None)
    return metric_pipeline
