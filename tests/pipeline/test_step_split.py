"""
tests/pipeline/test_step_split.py

Step 1's CV does not depend on the selection metric: selecting under any metric
from one CV run must match a separate run under that metric.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from h2ml.core.feature_store import PipelineData
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig
from h2ml.pipeline.step import make_classifier

FEATURE_NAMES = [f"feat_{i}" for i in range(5)]
N = 150


def _store() -> PipelineData:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, len(FEATURE_NAMES)))
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.standard_normal(N) > 0.5).astype(int)
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _pipeline(metric: str) -> H2MLPipeline:
    config = PipelineConfig(task_type="classification", metric=metric, n_splits=3, n_jobs=1, verbose=False)
    models = [
        make_classifier(LogisticRegression(max_iter=500), name="LR"),
        make_classifier(RandomForestClassifier(n_estimators=20, random_state=0), name="RF"),
        make_classifier(GaussianNB(), name="NB"),
    ]
    return H2MLPipeline(config=config, models=models)


@pytest.fixture(scope="module")
def shared_step1():
    pipeline = _pipeline("Brier")
    result, _ = pipeline._start(_store())
    fold_indices = pipeline._cv_step1(result)
    return result, fold_indices


@pytest.mark.parametrize("metric", ["Brier", "LogLoss", "AUC", "AUCPR", "F1"])
def test_select_step1_from_shared_cv_matches_separate_run(shared_step1, metric):
    shared, fold_indices = shared_step1
    pipeline = _pipeline(metric)
    selected = pipeline._select_step1(replace(shared, cv_warnings=list(shared.cv_warnings)), fold_indices)
    separate = pipeline.run_step1_only(_store())

    assert selected.metric == separate.metric == metric
    assert selected.best_model_name == separate.best_model_name
    assert selected.best_model_value == pytest.approx(separate.best_model_value)
    assert selected.best_model_std == pytest.approx(separate.best_model_std)
    assert selected.cv_warnings == separate.cv_warnings


def test_shared_cv_left_untouched_by_selection(shared_step1):
    shared, fold_indices = shared_step1
    _pipeline("AUC")._select_step1(replace(shared, cv_warnings=list(shared.cv_warnings)), fold_indices)
    assert shared.best_model_name is None
    assert shared.metric is None
