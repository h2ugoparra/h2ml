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
# Classification: fewer rows of the rarest class than this in the holdout make its score
# (and the conformal calibration built on it) unreliable, however many rows it has.
MIN_HOLDOUT_MINORITY = 20
# Regression: fewer non-zero rows than this leave a zero-heavy target's score with little to measure.
MIN_HOLDOUT_NONZERO = 20
# Warn when the holdout's mean or non-zero rate differs from dev by more than this factor.
MAX_HOLDOUT_DRIFT = 2.0
# Warn when whole blocks land further than this share from the requested holdout size.
MAX_HOLDOUT_SIZE_MISS = 0.25


def _draw_blocks(block_ids: np.ndarray, target: float, random_state: int) -> np.ndarray:
    """
    Row indices of whole blocks drawn at random until the total is closest to target rows.

    A block is kept only if it moves the total closer to target. The smallest block is
    used when every block alone overshoots too far for that to happen.
    """
    ids, counts = np.unique(block_ids, return_counts=True)
    chosen, size = [], 0
    for i in np.random.default_rng(random_state).permutation(ids.size):
        if abs(size + counts[i] - target) < abs(size - target):
            chosen.append(i)
            size += counts[i]
    return np.flatnonzero(np.isin(block_ids, ids[chosen or [counts.argmin()]]))


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
        A "spatial" holdout is whole blocks of the pipeline's spatial splitter, drawn at
        random until the total is closest to frac * n. Blocks come from coords alone, so
        for the same rows and seed the holdout does not depend on y or X.

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

        n_splits = max(2, round(1 / frac))
        # Labels the splitter's own debug lines, which step 1 repeats for its CV splitter on the dev rows.
        logger.debug(f"Holdout splitter: spatial blocks on all {n} rows; {frac:.0%} drawn at random.")
        splitter = build_splitter(
            task_type,
            n_splits=n_splits,
            random_state=random_state,
            coords=store.coords,
            X=store.X,
            y=store.y,
            spatial=replace(spatial or SpatialCVConfig(), use_target=False),
        )
        test_idx = _draw_blocks(splitter.block_id_, frac * n, random_state)
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
    return dev_idx, test_idx, resolved


def holdout_warnings(dev: PipelineData, holdout: PipelineData, task_type: TaskType, frac: float) -> list[str]:
    """
    [holdout] warnings about the rows set aside, for result.cv_warnings.

    Covers every task: too few rows, and a size far from the requested fraction. For
    regression also: too few non-zero rows on a target with zeros, and a mean or
    non-zero rate far from dev. Classification's rare-class check runs on the score.
    """
    warnings = []
    n_holdout = holdout.n_samples
    if n_holdout < MIN_HOLDOUT_SAMPLES:
        warnings.append(
            f"[holdout] The holdout has only {n_holdout} rows, so its score will be noisy. "
            "Consider a larger holdout_frac."
        )
    target = frac * (n_holdout + dev.n_samples)
    if abs(n_holdout - target) > MAX_HOLDOUT_SIZE_MISS * target:
        warnings.append(
            f"[holdout] The holdout has {n_holdout} rows against {target:.0f} requested: spatial blocks "
            "are too large to reach holdout_frac."
        )
    if task_type != TaskType.REGRESSION:
        return warnings

    def drifted(name: str, in_holdout: float, in_dev: float) -> None:
        if in_dev > 0 and not 1 / MAX_HOLDOUT_DRIFT <= in_holdout / in_dev <= MAX_HOLDOUT_DRIFT:
            warnings.append(
                f"[holdout] The holdout's {name} ({in_holdout:.3g}) differs from the dev rows' ({in_dev:.3g}) "
                f"by more than {MAX_HOLDOUT_DRIFT:g}×, so its score may not represent the dev distribution."
            )

    drifted("mean", float(holdout.y.mean()), float(dev.y.mean()))
    if (dev.y == 0).any() or (holdout.y == 0).any():
        n_nonzero = int(np.count_nonzero(holdout.y))
        if n_nonzero < MIN_HOLDOUT_NONZERO:
            warnings.append(
                f"[holdout] The holdout holds only {n_nonzero} non-zero row(s), so its score mostly reflects "
                "the zeros. Consider holdout_frac=0 for this target or a larger holdout_frac."
            )
        drifted("non-zero rate", n_nonzero / n_holdout, float(np.mean(dev.y != 0)))
    return warnings


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
