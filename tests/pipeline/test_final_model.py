"""
tests/pipeline/test_final_model.py

Tests for FinalModel (inference, scaling, persistence) and build_final_model().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from h2ml.pipeline.base import TaskType
from h2ml.pipeline.cv import CVResult, FoldResult
from h2ml.pipeline.final_model import ConformalCalibration, FinalModel, build_final_model, _build_conformal_calibration
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig, PipelineResult
from h2ml.pipeline.step import make_classifier
from h2ml.features.feature_store import PipelineData
from h2ml.features.selector import FeatureSelector
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["a", "b", "c", "d", "e"]
N = 80


@pytest.fixture
def X() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((N, len(FEATURE_NAMES)))


@pytest.fixture
def y_clf() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 2, N)


@pytest.fixture
def y_reg() -> np.ndarray:
    return np.random.default_rng(0).standard_normal(N)


@pytest.fixture
def fitted_rf_clf(X, y_clf) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y_clf)
    return clf


@pytest.fixture
def fitted_rf_reg(X, y_reg) -> RandomForestRegressor:
    reg = RandomForestRegressor(n_estimators=10, random_state=42)
    reg.fit(X, y_reg)
    return reg


@pytest.fixture
def clf_model(fitted_rf_clf) -> FinalModel:
    return FinalModel(
        estimator=fitted_rf_clf,
        feature_names=FEATURE_NAMES,
        task_type=TaskType.CLASSIFICATION,
    )


@pytest.fixture
def reg_model(fitted_rf_reg) -> FinalModel:
    return FinalModel(
        estimator=fitted_rf_reg,
        feature_names=FEATURE_NAMES,
        task_type=TaskType.REGRESSION,
    )


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:

    def test_clf_predict_returns_array(self, clf_model, X):
        preds = clf_model.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_clf_predict_shape(self, clf_model, X):
        assert clf_model.predict(X).shape == (N,)

    def test_reg_predict_returns_array(self, reg_model, X):
        preds = reg_model.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_predict_accepts_dataframe(self, clf_model, X):
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        preds = clf_model.predict(df)
        assert preds.shape == (N,)

    def test_predict_dataframe_aligns_columns_by_name(self, clf_model, X):
        """Shuffled column order in DataFrame should still produce correct predictions."""
        df_normal = pd.DataFrame(X, columns=FEATURE_NAMES)
        df_shuffled = df_normal[FEATURE_NAMES[::-1]]  # reversed column order
        np.testing.assert_array_equal(
            clf_model.predict(df_normal),
            clf_model.predict(df_shuffled),
        )


# ---------------------------------------------------------------------------
# predict_proba()
# ---------------------------------------------------------------------------

class TestPredictProba:

    def test_clf_predict_proba_returns_array(self, clf_model, X):
        proba = clf_model.predict_proba(X)
        assert isinstance(proba, np.ndarray)

    def test_clf_predict_proba_shape(self, clf_model, X):
        assert clf_model.predict_proba(X).shape == (N,)

    def test_reg_predict_proba_raises(self, reg_model, X):
        with pytest.raises(ValueError, match="classification"):
            reg_model.predict_proba(X)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

class TestScaling:

    def test_scaling_applied_when_required(self, X, y_clf):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_scaled, y_clf)

        model = FinalModel(
            estimator=clf,
            feature_names=FEATURE_NAMES,
            task_type=TaskType.CLASSIFICATION,
            requires_scaling=True,
            scaler=scaler,
        )

        preds_via_model = model.predict(X)
        preds_manual = clf.predict(scaler.transform(X))
        np.testing.assert_array_equal(preds_via_model, preds_manual)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_creates_file(self, clf_model, tmp_path):
        out = tmp_path / "model.pkl"
        clf_model.save(out)
        assert out.exists()

    def test_load_returns_final_model(self, clf_model, tmp_path):
        out = tmp_path / "model.pkl"
        clf_model.save(out)
        loaded = FinalModel.load(out)
        assert isinstance(loaded, FinalModel)

    def test_round_trip_predictions_match(self, clf_model, X, tmp_path):
        out = tmp_path / "model.pkl"
        clf_model.save(out)
        loaded = FinalModel.load(out)
        np.testing.assert_array_equal(clf_model.predict(X), loaded.predict(X))

    def test_save_creates_parent_dirs(self, clf_model, tmp_path):
        out = tmp_path / "nested" / "dir" / "model.pkl"
        clf_model.save(out)
        assert out.exists()


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_model_name(self, clf_model):
        clf_model.best_model_name = "RF"
        assert "RF" in repr(clf_model)

    def test_repr_contains_n_features(self, clf_model):
        assert str(len(FEATURE_NAMES)) in repr(clf_model)

    def test_repr_contains_task_type(self, clf_model):
        assert "classification" in repr(clf_model)


# ---------------------------------------------------------------------------
# build_final_model()
# ---------------------------------------------------------------------------

class TestBuildFinalModel:

    def _run_pipeline_with_lr(self, seed: int = 0) -> tuple[PipelineResult, PipelineData]:
        """Helper: run a minimal LR pipeline (no custom name — registry name required)."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(seed)
        store = PipelineData(
            X=rng.standard_normal((100, 5)).astype(np.float32),
            feature_names=[f"f{i}" for i in range(5)],
            y=rng.integers(0, 2, 100).astype(np.float32),
        )
        # No custom name — defaults to "LogisticRegression" matching the registry key
        pipeline = H2MLPipeline(
            models=[make_classifier(LogisticRegression(max_iter=200, random_state=42))],
            config=PipelineConfig(task_type=TaskType.CLASSIFICATION, n_splits=3, n_trials=1),
        )
        selected = [f"f{i}" for i in range(3)]
        mock_sel = MagicMock(spec=FeatureSelector)
        mock_sel.fit_transform.side_effect = lambda s, y=None: s.select(selected)
        mock_sel.transform.side_effect = lambda s: s.select(selected)

        with patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=mock_sel):
            result = pipeline.run(store)
        return result, store

    def test_returns_final_model(self):
        result, _ = self._run_pipeline_with_lr()
        final = result.build_final_model()
        assert isinstance(final, FinalModel)

    def test_final_model_can_predict(self):
        result, store = self._run_pipeline_with_lr(seed=1)
        final = result.build_final_model()

        rng = np.random.default_rng(99)
        X_new = rng.standard_normal((10, 5)).astype(np.float32)
        feature_idx = [store.feature_names.index(f) for f in final.feature_names]
        preds = final.predict(X_new[:, feature_idx])
        assert preds.shape == (10,)


# ---------------------------------------------------------------------------
# ConformalCalibration
# ---------------------------------------------------------------------------

def _make_cv_result(task_type: TaskType, n: int = 50, seed: int = 0) -> CVResult:
    """Build a minimal CVResult with two folds of synthetic OOF predictions."""
    rng = np.random.default_rng(seed)
    half = n // 2
    if task_type == TaskType.REGRESSION:
        fold0 = FoldResult(
            fold_id=0, model_name="M",
            y_train=np.zeros(half), y_test=rng.standard_normal(half),
            y_pred_train=np.zeros(half), y_pred_test=rng.standard_normal(half),
        )
        fold1 = FoldResult(
            fold_id=1, model_name="M",
            y_train=np.zeros(half), y_test=rng.standard_normal(half),
            y_pred_train=np.zeros(half), y_pred_test=rng.standard_normal(half),
        )
    else:
        fold0 = FoldResult(
            fold_id=0, model_name="M",
            y_train=np.zeros(half), y_test=rng.integers(0, 2, half).astype(float),
            y_pred_train=np.zeros(half), y_pred_test=rng.integers(0, 2, half).astype(float),
            y_prob_train=rng.uniform(0, 1, half),
            y_prob_test=rng.uniform(0, 1, half),
        )
        fold1 = FoldResult(
            fold_id=1, model_name="M",
            y_train=np.zeros(half), y_test=rng.integers(0, 2, half).astype(float),
            y_pred_train=np.zeros(half), y_pred_test=rng.integers(0, 2, half).astype(float),
            y_prob_train=rng.uniform(0, 1, half),
            y_prob_test=rng.uniform(0, 1, half),
        )
    return CVResult(model_name="M", task_type=task_type, folds=[fold0, fold1])


class TestConformalCalibration:

    def test_threshold_returns_float(self):
        cal = ConformalCalibration(
            scores=np.linspace(0, 1, 100), n=100, task_type=TaskType.REGRESSION
        )
        assert isinstance(cal.threshold(0.10), float)

    def test_threshold_increases_with_coverage(self):
        cal = ConformalCalibration(
            scores=np.linspace(0, 1, 100), n=100, task_type=TaskType.REGRESSION
        )
        assert cal.threshold(0.20) <= cal.threshold(0.05)

    def test_threshold_clamps_to_max_score(self):
        cal = ConformalCalibration(
            scores=np.linspace(0, 1, 10), n=10, task_type=TaskType.REGRESSION
        )
        # alpha=0 → should return the maximum score (clamped at level=1.0)
        assert cal.threshold(0.0) == pytest.approx(1.0)

    def test_regression_scores_are_absolute_residuals(self):
        cv = _make_cv_result(TaskType.REGRESSION)

        class MockResult:
            best_cv_result = cv

        cal = _build_conformal_calibration(MockResult())
        assert cal is not None
        assert cal.n == 50
        assert np.all(cal.scores >= 0)
        assert np.all(np.diff(cal.scores) >= 0)  # sorted ascending

    def test_classification_scores_in_0_1(self):
        cv = _make_cv_result(TaskType.CLASSIFICATION)

        class MockResult:
            best_cv_result = cv

        cal = _build_conformal_calibration(MockResult())
        assert cal is not None
        assert np.all(cal.scores >= 0) and np.all(cal.scores <= 1)

    def test_returns_none_when_no_cv_result(self):
        class MockResult:
            best_cv_result = None

        assert _build_conformal_calibration(MockResult()) is None

    def test_returns_none_when_folds_empty(self):
        cv = CVResult(model_name="M", task_type=TaskType.REGRESSION, folds=[])

        class MockResult:
            best_cv_result = cv

        assert _build_conformal_calibration(MockResult()) is None


# ---------------------------------------------------------------------------
# predict_interval / predict_set
# ---------------------------------------------------------------------------

class TestPredictInterval:

    def test_returns_lower_upper_arrays(self, reg_model, X):
        cal = ConformalCalibration(
            scores=np.linspace(0, 1, 50), n=50, task_type=TaskType.REGRESSION
        )
        reg_model.conformal = cal
        lower, upper = reg_model.predict_interval(X)
        assert lower.shape == (N,)
        assert upper.shape == (N,)

    def test_upper_greater_than_lower(self, reg_model, X):
        cal = ConformalCalibration(
            scores=np.linspace(0, 1, 50), n=50, task_type=TaskType.REGRESSION
        )
        reg_model.conformal = cal
        lower, upper = reg_model.predict_interval(X)
        assert np.all(upper >= lower)

    def test_smaller_alpha_gives_wider_interval(self, reg_model, X):
        cal = ConformalCalibration(
            scores=np.linspace(0, 2, 100), n=100, task_type=TaskType.REGRESSION
        )
        reg_model.conformal = cal
        l90, u90 = reg_model.predict_interval(X, alpha=0.10)
        l80, u80 = reg_model.predict_interval(X, alpha=0.20)
        assert np.all((u90 - l90) >= (u80 - l80))

    def test_raises_for_classification_model(self, clf_model, X):
        clf_model.conformal = ConformalCalibration(
            scores=np.array([0.1, 0.2]), n=2, task_type=TaskType.CLASSIFICATION
        )
        with pytest.raises(ValueError, match="regression"):
            clf_model.predict_interval(X)

    def test_raises_without_calibration(self, reg_model, X):
        reg_model.conformal = None
        with pytest.raises(ValueError, match="conformal calibration"):
            reg_model.predict_interval(X)


class TestPredictSet:

    def test_returns_list_of_arrays(self, clf_model, X):
        cal = ConformalCalibration(
            scores=np.linspace(0, 1, 50), n=50, task_type=TaskType.CLASSIFICATION
        )
        clf_model.conformal = cal
        sets = clf_model.predict_set(X)
        assert len(sets) == N
        assert all(isinstance(s, np.ndarray) for s in sets)

    def test_high_coverage_includes_both_classes(self, clf_model, X):
        # alpha=0 → q=1.0, every sample gets {0, 1}
        cal = ConformalCalibration(
            scores=np.ones(100), n=100, task_type=TaskType.CLASSIFICATION
        )
        clf_model.conformal = cal
        sets = clf_model.predict_set(X, alpha=0.0)
        assert all(len(s) == 2 for s in sets)

    def test_labels_are_subset_of_0_1(self, clf_model, X):
        cal = ConformalCalibration(
            scores=np.linspace(0, 1, 50), n=50, task_type=TaskType.CLASSIFICATION
        )
        clf_model.conformal = cal
        sets = clf_model.predict_set(X)
        for s in sets:
            assert set(s.tolist()).issubset({0, 1})

    def test_raises_for_regression_model(self, reg_model, X):
        reg_model.conformal = ConformalCalibration(
            scores=np.array([0.1, 0.2]), n=2, task_type=TaskType.REGRESSION
        )
        with pytest.raises(ValueError, match="classification"):
            reg_model.predict_set(X)

    def test_raises_without_calibration(self, clf_model, X):
        clf_model.conformal = None
        with pytest.raises(ValueError, match="conformal calibration"):
            clf_model.predict_set(X)


class TestBuildFinalModelConformal:

    def test_conformal_is_set_after_pipeline_run(self):
        from sklearn.linear_model import LogisticRegression
        from unittest.mock import patch

        rng = np.random.default_rng(0)
        store = PipelineData(
            X=rng.standard_normal((100, 5)).astype(np.float32),
            feature_names=[f"f{i}" for i in range(5)],
            y=rng.integers(0, 2, 100).astype(np.float32),
        )
        pipeline = H2MLPipeline(
            models=[make_classifier(LogisticRegression(max_iter=200, random_state=42))],
            config=PipelineConfig(task_type=TaskType.CLASSIFICATION, n_splits=3, n_trials=1),
        )
        selected = [f"f{i}" for i in range(3)]
        mock_sel = MagicMock(spec=FeatureSelector)
        mock_sel.fit_transform.side_effect = lambda s, y=None: s.select(selected)
        mock_sel.transform.side_effect = lambda s: s.select(selected)

        with patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=mock_sel):
            result = pipeline.run(store)

        final = result.build_final_model()
        assert final.conformal is not None
        assert isinstance(final.conformal, ConformalCalibration)
        assert final.conformal.n > 0

    def test_conformal_persists_through_save_load(self, tmp_path):
        from sklearn.linear_model import LogisticRegression
        from unittest.mock import patch

        rng = np.random.default_rng(1)
        store = PipelineData(
            X=rng.standard_normal((100, 5)).astype(np.float32),
            feature_names=[f"f{i}" for i in range(5)],
            y=rng.integers(0, 2, 100).astype(np.float32),
        )
        pipeline = H2MLPipeline(
            models=[make_classifier(LogisticRegression(max_iter=200, random_state=42))],
            config=PipelineConfig(task_type=TaskType.CLASSIFICATION, n_splits=3, n_trials=1),
        )
        selected = [f"f{i}" for i in range(3)]
        mock_sel = MagicMock(spec=FeatureSelector)
        mock_sel.fit_transform.side_effect = lambda s, y=None: s.select(selected)
        mock_sel.transform.side_effect = lambda s: s.select(selected)

        with patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=mock_sel):
            result = pipeline.run(store)

        final = result.build_final_model()
        out = tmp_path / "final.pkl"
        final.save(out)
        loaded = FinalModel.load(out)
        assert loaded.conformal is not None
        assert loaded.conformal.n == final.conformal.n
