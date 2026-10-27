"""
h2ml/pipeline/holdout.py

Outer holdout: rows set aside before step 1 and scored once on the final
selected configuration, giving a performance estimate that no selection
step has seen.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterator, Optional

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from h2ml.core.base import TaskType
from h2ml.core.feature_store import PipelineData
from h2ml.core.spatial_config import SpatialCVConfig
from h2ml.preprocessing.transforms import PRE_CLIP_TRANSFORMS, Y_TRANSFORMS

HOLDOUT_STRATEGIES = ("auto", "random", "time")
HOLDOUT_REFITS = ("dev", "all")
# Holdout sizes below this give a very noisy score — warned, not refused.
MIN_HOLDOUT_SAMPLES = 30


def split_holdout(
    store: PipelineData,
    task_type: TaskType,
    frac: float,
    strategy: str,
    random_state: int,
    spatial: Optional[SpatialCVConfig] = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Split store rows into development and holdout indices.

    Strategies:
        "auto"   — "spatial" when store.coords is set, else "random".
        "random" — stratified by y for classification, plain random for regression.
        "time"   — the latest ceil(frac * n) rows by store.times.
        A "spatial" holdout is one fold of the pipeline's spatial splitter built with
        round(1 / frac) folds — the fold whose size is closest to frac * n — so held-out
        rows are whole spatial blocks rather than neighbours of training rows.

    Returns:
        (dev_idx, test_idx, resolved_strategy) — sorted, disjoint, covering every row.

    Raises:
        ValueError: If "time" is requested without store.times, the holdout is empty,
            or a classification holdout or dev set holds a single class.
    """
    n = store.n_samples
    resolved = strategy
    if strategy == "auto":
        resolved = "spatial" if store.coords is not None else "random"

    all_idx = np.arange(n)
    if resolved == "time":
        if store.times is None:
            raise ValueError('holdout_strategy="time" requires store.times.')
        n_test = math.ceil(frac * n)
        order = np.argsort(np.asarray(store.times, dtype="datetime64[D]"), kind="stable")
        test_idx = order[n - n_test :]
    elif resolved == "spatial":
        from h2ml.features.spatial_cv import build_splitter

        splitter = build_splitter(
            task_type,
            n_splits=max(2, round(1 / frac)),
            random_state=random_state,
            coords=store.coords,
            X=store.X,
            y=store.y,
            spatial=spatial,
        )
        folds = [te for _, te in splitter.split(store.X, store.y) if te.size > 0]
        if not folds:
            raise ValueError("Spatial holdout split produced no non-empty fold.")
        test_idx = min(folds, key=lambda te: abs(te.size - frac * n))
    else:
        stratify = store.y if task_type == TaskType.CLASSIFICATION else None
        _, test_idx = train_test_split(all_idx, test_size=frac, random_state=random_state, stratify=stratify)

    test_idx = np.sort(test_idx)
    dev_idx = np.setdiff1d(all_idx, test_idx)
    if test_idx.size == 0 or dev_idx.size == 0:
        raise ValueError(f"Holdout split left an empty set (dev={dev_idx.size}, holdout={test_idx.size}).")
    if task_type == TaskType.CLASSIFICATION:
        for name, idx in (("holdout", test_idx), ("dev", dev_idx)):
            if np.unique(store.y[idx]).size < 2:
                raise ValueError(
                    f"The {name} set from holdout_strategy={resolved!r} holds a single class. "
                    "Use a larger holdout_frac or a different holdout_strategy."
                )
    if test_idx.size < MIN_HOLDOUT_SAMPLES:
        logger.warning(
            f"Holdout has only {test_idx.size} rows — its score will be noisy. Consider a larger holdout_frac."
        )
    return dev_idx, test_idx, resolved


def concat_stores(a: PipelineData, b: PipelineData) -> PipelineData:
    """Stack the rows of two stores with the same features (a first, then b)."""

    def _cat(x, y):
        return None if x is None or y is None else np.concatenate([x, y])

    return PipelineData(
        X=np.vstack([a.X, b.X]),
        feature_names=list(a.feature_names),
        y=np.concatenate([a.y, b.y]),
        y_true=_cat(a.y_true, b.y_true),
        y_transform=a.y_transform,
        coords=_cat(a.coords, b.coords),
        times=_cat(a.times, b.times),
    )


def stage_store(
    store: PipelineData,
    y_transform: Optional[str],
    feature_names: Optional[list[str]] = None,
) -> PipelineData:
    """
    Put a raw store into the y-space and feature set the winner was validated on.

    The y-transform is re-applied only when the store does not already carry one.
    Winsorize variants keep y before clipping — the limit is fitted on training rows
    by get_train_target_fn — matching build_transform_stores.
    """
    if y_transform and store.y_transform is None:
        if y_transform in PRE_CLIP_TRANSFORMS:
            y = PRE_CLIP_TRANSFORMS[y_transform](store.y)
        else:
            y = Y_TRANSFORMS[y_transform](store.y)
        store = replace(store, y=y, y_true=store.y, y_transform=y_transform)
    if feature_names is not None:
        store = store.select(feature_names)
    return store


class FixedSplit:
    """One-fold splitter: train on the first n_train rows, test on the rest."""

    def __init__(self, n_train: int, n_samples: int):
        self.n_train = n_train
        self.n_samples = n_samples

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return 1

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield np.arange(self.n_train), np.arange(self.n_train, self.n_samples)
