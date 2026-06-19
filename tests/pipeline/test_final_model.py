"""
tests/pipeline/test_final_model.py

Tests for FinalModel (inference, scaling, persistence) and build_final_model().
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from h2ml.pipeline.base import TaskType
from h2ml.pipeline.cv import CVResult, FoldResult
from h2ml.evaluation.conformal import (
    ConformalCalibration,
    LocalConformalCalibration,
    _conformal_quantile,
    _encode_times,
    _time_bin,
)
from h2ml.pipeline.final_model import (
    DeltaFinalModel,
    FinalModel,
    _build_conformal_calibration,
    _build_delta_conformal,
    build_delta_final_model,
)
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

    def test_raises_on_incomplete_result(self):
        """An empty/partial result raises a clear ValueError instead of an opaque
        AssertionError (which also vanishes under python -O)."""
        with pytest.raises(ValueError, match="step-1 CV result"):
            PipelineResult().build_final_model()


# ---------------------------------------------------------------------------
# ConformalCalibration
# ---------------------------------------------------------------------------


def _make_cv_result(task_type: TaskType, n: int = 50, seed: int = 0) -> CVResult:
    """Build a minimal CVResult with two folds of synthetic OOF predictions."""
    rng = np.random.default_rng(seed)
    half = n // 2
    if task_type == TaskType.REGRESSION:
        fold0 = FoldResult(
            fold_id=0,
            model_name="M",
            y_train=np.zeros(half),
            y_test=rng.standard_normal(half),
            y_pred_train=np.zeros(half),
            y_pred_test=rng.standard_normal(half),
        )
        fold1 = FoldResult(
            fold_id=1,
            model_name="M",
            y_train=np.zeros(half),
            y_test=rng.standard_normal(half),
            y_pred_train=np.zeros(half),
            y_pred_test=rng.standard_normal(half),
        )
    else:
        fold0 = FoldResult(
            fold_id=0,
            model_name="M",
            y_train=np.zeros(half),
            y_test=rng.integers(0, 2, half).astype(float),
            y_pred_train=np.zeros(half),
            y_pred_test=rng.integers(0, 2, half).astype(float),
            y_prob_train=rng.uniform(0, 1, half),
            y_prob_test=rng.uniform(0, 1, half),
        )
        fold1 = FoldResult(
            fold_id=1,
            model_name="M",
            y_train=np.zeros(half),
            y_test=rng.integers(0, 2, half).astype(float),
            y_pred_train=np.zeros(half),
            y_pred_test=rng.integers(0, 2, half).astype(float),
            y_prob_train=rng.uniform(0, 1, half),
            y_prob_test=rng.uniform(0, 1, half),
        )
    return CVResult(model_name="M", task_type=task_type, folds=[fold0, fold1])


class TestConformalCalibration:
    def test_threshold_returns_float(self):
        cal = ConformalCalibration(scores=np.linspace(0, 1, 100), n=100, task_type=TaskType.REGRESSION)
        assert isinstance(cal.threshold(0.10), float)

    def test_threshold_increases_with_coverage(self):
        cal = ConformalCalibration(scores=np.linspace(0, 1, 100), n=100, task_type=TaskType.REGRESSION)
        assert cal.threshold(0.20) <= cal.threshold(0.05)

    def test_threshold_clamps_to_max_score(self):
        cal = ConformalCalibration(scores=np.linspace(0, 1, 10), n=10, task_type=TaskType.REGRESSION)
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
        cal = ConformalCalibration(scores=np.linspace(0, 1, 50), n=50, task_type=TaskType.REGRESSION)
        reg_model.conformal = cal
        lower, upper = reg_model.predict_interval(X)
        assert lower.shape == (N,)
        assert upper.shape == (N,)

    def test_upper_greater_than_lower(self, reg_model, X):
        cal = ConformalCalibration(scores=np.linspace(0, 1, 50), n=50, task_type=TaskType.REGRESSION)
        reg_model.conformal = cal
        lower, upper = reg_model.predict_interval(X)
        assert np.all(upper >= lower)

    def test_smaller_alpha_gives_wider_interval(self, reg_model, X):
        cal = ConformalCalibration(scores=np.linspace(0, 2, 100), n=100, task_type=TaskType.REGRESSION)
        reg_model.conformal = cal
        l90, u90 = reg_model.predict_interval(X, alpha=0.10)
        l80, u80 = reg_model.predict_interval(X, alpha=0.20)
        assert np.all((u90 - l90) >= (u80 - l80))

    def test_raises_for_classification_model(self, clf_model, X):
        clf_model.conformal = ConformalCalibration(scores=np.array([0.1, 0.2]), n=2, task_type=TaskType.CLASSIFICATION)
        with pytest.raises(ValueError, match="regression"):
            clf_model.predict_interval(X)

    def test_raises_without_calibration(self, reg_model, X):
        reg_model.conformal = None
        with pytest.raises(ValueError, match="conformal calibration"):
            reg_model.predict_interval(X)


class TestPredictSet:
    def test_returns_list_of_arrays(self, clf_model, X):
        cal = ConformalCalibration(scores=np.linspace(0, 1, 50), n=50, task_type=TaskType.CLASSIFICATION)
        clf_model.conformal = cal
        sets = clf_model.predict_set(X)
        assert len(sets) == N
        assert all(isinstance(s, np.ndarray) for s in sets)

    def test_high_coverage_includes_both_classes(self, clf_model, X):
        # alpha=0 → q=1.0, every sample gets {0, 1}
        cal = ConformalCalibration(scores=np.ones(100), n=100, task_type=TaskType.CLASSIFICATION)
        clf_model.conformal = cal
        sets = clf_model.predict_set(X, alpha=0.0)
        assert all(len(s) == 2 for s in sets)

    def test_labels_are_subset_of_0_1(self, clf_model, X):
        cal = ConformalCalibration(scores=np.linspace(0, 1, 50), n=50, task_type=TaskType.CLASSIFICATION)
        clf_model.conformal = cal
        sets = clf_model.predict_set(X)
        for s in sets:
            assert set(s.tolist()).issubset({0, 1})

    def test_raises_for_regression_model(self, reg_model, X):
        reg_model.conformal = ConformalCalibration(scores=np.array([0.1, 0.2]), n=2, task_type=TaskType.REGRESSION)
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


# ---------------------------------------------------------------------------
# DeltaFinalModel
# ---------------------------------------------------------------------------

N_DELTA = 80
N_POS = 50
DELTA_FEATURES = ["a", "b", "c", "d", "e"]


def _make_delta_components(
    n: int = N_DELTA,
    n_pos: int = N_POS,
    seed: int = 0,
) -> tuple[FinalModel, FinalModel, np.ndarray, np.ndarray, np.ndarray]:
    """Return (clf_model, reg_model, X_full, y_full, positive_indices)."""
    rng = np.random.default_rng(seed)
    X_full = rng.standard_normal((n, len(DELTA_FEATURES)))
    y_full = rng.integers(0, 8, n).astype(float)
    positive_indices = np.where(y_full > 0)[0][:n_pos]  # cap at n_pos for reproducibility

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X_full, (y_full > 0).astype(int))
    clf_model = FinalModel(estimator=clf, feature_names=DELTA_FEATURES, task_type=TaskType.CLASSIFICATION)

    reg = RandomForestRegressor(n_estimators=5, random_state=42)
    reg.fit(X_full[positive_indices], y_full[positive_indices])
    reg_model = FinalModel(estimator=reg, feature_names=DELTA_FEATURES, task_type=TaskType.REGRESSION)

    return clf_model, reg_model, X_full, y_full, positive_indices


class TestDeltaFinalModel:
    def test_predict_shape(self):
        clf_model, reg_model, X_full, _, _ = _make_delta_components()
        model = DeltaFinalModel(clf=clf_model, reg=reg_model)
        assert model.predict(X_full).shape == (N_DELTA,)

    def test_predict_non_negative(self):
        clf_model, reg_model, X_full, _, _ = _make_delta_components()
        model = DeltaFinalModel(clf=clf_model, reg=reg_model)
        assert np.all(model.predict(X_full) >= 0)

    def test_predict_accepts_dataframe(self):
        clf_model, reg_model, X_full, _, _ = _make_delta_components()
        model = DeltaFinalModel(clf=clf_model, reg=reg_model)
        df = pd.DataFrame(X_full, columns=DELTA_FEATURES)
        np.testing.assert_array_almost_equal(model.predict(X_full), model.predict(df))

    def test_predict_interval_shape(self):
        clf_model, reg_model, X_full, _, _ = _make_delta_components()
        cal = ConformalCalibration(scores=np.linspace(0, 5, 100), n=100, task_type=TaskType.REGRESSION)
        model = DeltaFinalModel(clf=clf_model, reg=reg_model, conformal=cal)
        lower, upper = model.predict_interval(X_full)
        assert lower.shape == (N_DELTA,)
        assert upper.shape == (N_DELTA,)

    def test_predict_interval_upper_ge_lower(self):
        clf_model, reg_model, X_full, _, _ = _make_delta_components()
        cal = ConformalCalibration(scores=np.linspace(0, 5, 100), n=100, task_type=TaskType.REGRESSION)
        model = DeltaFinalModel(clf=clf_model, reg=reg_model, conformal=cal)
        lower, upper = model.predict_interval(X_full)
        assert np.all(upper >= lower)

    def test_predict_interval_lower_non_negative(self):
        clf_model, reg_model, X_full, _, _ = _make_delta_components()
        cal = ConformalCalibration(scores=np.linspace(0, 5, 100), n=100, task_type=TaskType.REGRESSION)
        model = DeltaFinalModel(clf=clf_model, reg=reg_model, conformal=cal)
        lower, _ = model.predict_interval(X_full)
        assert np.all(lower >= 0)

    def test_predict_interval_smaller_alpha_gives_wider_interval(self):
        clf_model, reg_model, X_full, _, _ = _make_delta_components()
        cal = ConformalCalibration(scores=np.linspace(0, 5, 200), n=200, task_type=TaskType.REGRESSION)
        model = DeltaFinalModel(clf=clf_model, reg=reg_model, conformal=cal)
        l90, u90 = model.predict_interval(X_full, alpha=0.10)
        l80, u80 = model.predict_interval(X_full, alpha=0.20)
        assert np.all((u90 - l90) >= (u80 - l80))

    def test_predict_interval_raises_without_calibration(self):
        clf_model, reg_model, X_full, _, _ = _make_delta_components()
        model = DeltaFinalModel(clf=clf_model, reg=reg_model)
        with pytest.raises(ValueError, match="conformal calibration"):
            model.predict_interval(X_full)

    def test_save_creates_pkl_files(self, tmp_path):
        clf_model, reg_model, _, _, _ = _make_delta_components()
        cal = ConformalCalibration(scores=np.linspace(0, 1, 50), n=50, task_type=TaskType.REGRESSION)
        model = DeltaFinalModel(clf=clf_model, reg=reg_model, conformal=cal)
        model.save(tmp_path / "delta")
        assert (tmp_path / "delta" / "clf.pkl").exists()
        assert (tmp_path / "delta" / "reg.pkl").exists()
        assert (tmp_path / "delta" / "conformal.pkl").exists()

    def test_save_no_conformal_file_when_uncalibrated(self, tmp_path):
        clf_model, reg_model, _, _, _ = _make_delta_components()
        DeltaFinalModel(clf=clf_model, reg=reg_model).save(tmp_path / "delta")
        assert not (tmp_path / "delta" / "conformal.pkl").exists()

    def test_load_roundtrip_predictions(self, tmp_path):
        clf_model, reg_model, X_full, _, _ = _make_delta_components()
        model = DeltaFinalModel(clf=clf_model, reg=reg_model)
        model.save(tmp_path / "delta")
        loaded = DeltaFinalModel.load(tmp_path / "delta")
        np.testing.assert_array_almost_equal(model.predict(X_full), loaded.predict(X_full))

    def test_load_preserves_calibration(self, tmp_path):
        clf_model, reg_model, _, _, _ = _make_delta_components()
        cal = ConformalCalibration(scores=np.linspace(0, 2, 80), n=80, task_type=TaskType.REGRESSION)
        model = DeltaFinalModel(clf=clf_model, reg=reg_model, conformal=cal)
        model.save(tmp_path / "delta")
        loaded = DeltaFinalModel.load(tmp_path / "delta")
        assert loaded.conformal is not None
        assert loaded.conformal.n == 80

    def test_load_without_calibration(self, tmp_path):
        clf_model, reg_model, _, _, _ = _make_delta_components()
        DeltaFinalModel(clf=clf_model, reg=reg_model).save(tmp_path / "delta")
        loaded = DeltaFinalModel.load(tmp_path / "delta")
        assert loaded.conformal is None

    def test_repr(self):
        clf_model, reg_model, _, _, _ = _make_delta_components()
        r = repr(DeltaFinalModel(clf=clf_model, reg=reg_model))
        assert "DeltaFinalModel" in r


# ---------------------------------------------------------------------------
# _build_delta_conformal / build_delta_final_model
# ---------------------------------------------------------------------------


def _make_indexed_cv_result(task_type: TaskType, n: int, seed: int = 0) -> CVResult:
    """CVResult with non-empty test_idx covering [0, n)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    idx0, idx1 = np.arange(0, half), np.arange(half, n)

    def make_fold(fold_id, test_idx, train_idx):
        size = len(test_idx)
        if task_type == TaskType.REGRESSION:
            return FoldResult(
                fold_id=fold_id,
                model_name="M",
                y_train=np.zeros(len(train_idx)),
                y_test=rng.uniform(1, 8, size),
                y_pred_train=np.zeros(len(train_idx)),
                y_pred_test=rng.uniform(1, 8, size),
                test_idx=test_idx,
                train_idx=train_idx,
            )
        return FoldResult(
            fold_id=fold_id,
            model_name="M",
            y_train=np.zeros(len(train_idx)),
            y_test=rng.integers(0, 2, size).astype(float),
            y_pred_train=np.zeros(len(train_idx)),
            y_pred_test=rng.integers(0, 2, size).astype(float),
            y_prob_train=rng.uniform(0, 1, len(train_idx)),
            y_prob_test=rng.uniform(0, 1, size),
            test_idx=test_idx,
            train_idx=train_idx,
        )

    return CVResult(
        model_name="M",
        task_type=task_type,
        folds=[make_fold(0, idx0, idx1), make_fold(1, idx1, idx0)],
    )


class MockResult:
    def __init__(self, cv: CVResult):
        self.best_cv_result = cv


class TestBuildDeltaConformal:
    def _setup(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        X_full = rng.standard_normal((n, len(DELTA_FEATURES)))
        y_full = rng.integers(0, 8, n).astype(float)
        positive_indices = np.where(y_full > 0)[0]
        n_pos = len(positive_indices)

        reg = RandomForestRegressor(n_estimators=5, random_state=42)
        reg.fit(X_full[positive_indices], y_full[positive_indices])
        reg_final = FinalModel(estimator=reg, feature_names=DELTA_FEATURES, task_type=TaskType.REGRESSION)

        clf_result = MockResult(cv=_make_indexed_cv_result(TaskType.CLASSIFICATION, n, seed))
        reg_result = MockResult(cv=_make_indexed_cv_result(TaskType.REGRESSION, n_pos, seed + 1))

        return clf_result, reg_result, reg_final, X_full, y_full, positive_indices

    def test_returns_calibration(self):
        clf_r, reg_r, reg_f, X, y, pos = self._setup()
        cal = _build_delta_conformal(clf_r, reg_r, reg_f, X, y, pos)
        assert cal is not None
        assert isinstance(cal, ConformalCalibration)

    def test_scores_non_negative(self):
        clf_r, reg_r, reg_f, X, y, pos = self._setup()
        cal = _build_delta_conformal(clf_r, reg_r, reg_f, X, y, pos)
        assert np.all(cal.scores >= 0)

    def test_scores_sorted_ascending(self):
        clf_r, reg_r, reg_f, X, y, pos = self._setup()
        cal = _build_delta_conformal(clf_r, reg_r, reg_f, X, y, pos)
        assert np.all(np.diff(cal.scores) >= 0)

    def test_n_equals_sample_count(self):
        n = 60
        clf_r, reg_r, reg_f, X, y, pos = self._setup(n=n)
        cal = _build_delta_conformal(clf_r, reg_r, reg_f, X, y, pos)
        assert cal.n == n

    def test_returns_none_when_no_clf_cv(self):
        _, reg_r, reg_f, X, y, pos = self._setup()

        class NoCv:
            best_cv_result = None

        assert _build_delta_conformal(NoCv(), reg_r, reg_f, X, y, pos) is None

    def test_returns_none_when_clf_length_mismatch(self):
        clf_r, reg_r, reg_f, X, y, pos = self._setup(n=60)
        X_wrong = X[:50]
        y_wrong = y[:50]
        assert _build_delta_conformal(clf_r, reg_r, reg_f, X_wrong, y_wrong, pos) is None


class TestBuildDeltaFinalModel:
    def test_returns_delta_final_model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestRegressor as RFR
        from h2ml.pipeline.step import make_classifier, make_regressor

        rng = np.random.default_rng(0)
        n = 60
        X_full = rng.standard_normal((n, len(DELTA_FEATURES))).astype(np.float32)
        y_full = rng.integers(0, 8, n).astype(float)
        positive_indices = np.where(y_full > 0)[0]

        # Classifier pipeline (all N samples, no feature reduction)
        clf_store = PipelineData(X=X_full, feature_names=DELTA_FEATURES, y=(y_full > 0).astype(np.float32))
        clf_pipeline = H2MLPipeline(
            models=[make_classifier(LogisticRegression(max_iter=200, random_state=42))],
            config=PipelineConfig(task_type=TaskType.CLASSIFICATION, n_splits=3, n_trials=1),
        )
        mock_sel = MagicMock(spec=FeatureSelector)
        mock_sel.fit_transform.side_effect = lambda s, y=None: s
        mock_sel.transform.side_effect = lambda s: s
        with patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=mock_sel):
            clf_result = clf_pipeline.run(clf_store)

        # Regressor pipeline (positive samples only, no feature reduction)
        X_pos = X_full[positive_indices]
        y_pos = y_full[positive_indices].astype(np.float32)
        reg_store = PipelineData(X=X_pos, feature_names=DELTA_FEATURES, y=y_pos)
        reg_pipeline = H2MLPipeline(
            models=[make_regressor(RFR(n_estimators=10, random_state=42))],
            config=PipelineConfig(task_type=TaskType.REGRESSION, metric="R2", n_splits=3, n_trials=1),
        )
        with patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=mock_sel):
            reg_result = reg_pipeline.run(reg_store)

        delta = build_delta_final_model(clf_result, reg_result, X_full, y_full, positive_indices)

        assert isinstance(delta, DeltaFinalModel)
        assert delta.conformal is not None
        assert delta.predict(X_full).shape == (n,)
        lower, upper = delta.predict_interval(X_full)
        assert np.all(upper >= lower)
        assert np.all(lower >= 0)


# ---------------------------------------------------------------------------
# LocalConformalCalibration
# ---------------------------------------------------------------------------


def _make_local_conformal(metric: str = "euclidean", min_block_n: int = 3) -> LocalConformalCalibration:
    """
    Two-block fixture: block 0 (low error, coords near origin) and
    block 1 (high error, coords near [10, 10]).
    """
    rng = np.random.default_rng(0)
    n_per_block = 20
    # block 0: low residuals
    scores_0 = rng.uniform(0.0, 0.2, n_per_block)
    coords_0 = rng.uniform(0.0, 1.0, (n_per_block, 2))
    # block 1: high residuals
    scores_1 = rng.uniform(0.8, 1.5, n_per_block)
    coords_1 = rng.uniform(9.0, 10.0, (n_per_block, 2))

    all_scores = np.concatenate([scores_0, scores_1])
    all_coords = np.vstack([coords_0, coords_1])
    all_block_idx = np.array([0] * n_per_block + [1] * n_per_block)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(all_coords)
    oof_context_scaled = scaler.transform(all_coords)

    return LocalConformalCalibration(
        scores_by_block=[np.sort(scores_0), np.sort(scores_1)],
        oof_context_scaled=oof_context_scaled,
        oof_block_indices=all_block_idx,
        context_mean=scaler.mean_,
        context_std=scaler.scale_,
        fallback_scores=np.sort(all_scores),
        metric=metric,
        min_block_n=min_block_n,
    )


class TestLocalConformalCalibration:
    def test_threshold_returns_array(self):
        lc = _make_local_conformal()
        coords = np.array([[0.5, 0.5], [0.2, 0.3], [9.5, 9.5]])
        q = lc.threshold(0.10, coords=coords)
        assert q.shape == (3,)
        assert np.all(q > 0)

    def test_block_locality_spatial(self):
        lc = _make_local_conformal()
        near_block0 = np.array([[0.5, 0.5]])
        near_block1 = np.array([[9.5, 9.5]])
        q0 = lc.threshold(0.10, coords=near_block0)[0]
        q1 = lc.threshold(0.10, coords=near_block1)[0]
        assert q1 > q0, "High-error block should produce larger threshold"

    def test_block_locality_temporal(self):
        """Points in different seasons should get different thresholds when
        blocks differ by seasonal residual distribution."""
        rng = np.random.default_rng(1)
        n = 20
        # block 0: winter dates (Jan), low error
        scores_0 = rng.uniform(0.0, 0.2, n)
        times_0 = np.array(["2021-01-15"] * n)
        # block 1: summer dates (Jul), high error
        scores_1 = rng.uniform(0.8, 1.5, n)
        times_1 = np.array(["2021-07-15"] * n)

        all_scores = np.concatenate([scores_0, scores_1])
        all_times = np.concatenate([times_0, times_1])
        all_block_idx = np.array([0] * n + [1] * n)

        encoded = _encode_times(all_times)
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(encoded)
        oof_context_scaled = scaler.transform(encoded)

        lc = LocalConformalCalibration(
            scores_by_block=[np.sort(scores_0), np.sort(scores_1)],
            oof_context_scaled=oof_context_scaled,
            oof_block_indices=all_block_idx,
            context_mean=scaler.mean_,
            context_std=scaler.scale_,
            fallback_scores=np.sort(all_scores),
            metric="euclidean",
            min_block_n=3,
            has_coords=False,
            has_times=True,
        )

        q_winter = lc.threshold(0.10, times=np.array(["2021-01-20"]))[0]
        q_summer = lc.threshold(0.10, times=np.array(["2021-07-20"]))[0]
        assert q_summer > q_winter

    def test_fallback_on_small_block(self):
        lc = _make_local_conformal(min_block_n=50)  # both blocks < 50 samples
        coords = np.array([[0.5, 0.5]])
        q = lc.threshold(0.10, coords=coords)[0]
        # Should equal global fallback threshold
        n = len(lc.fallback_scores)
        level = min(np.ceil(0.9 * (n + 1)) / n, 1.0)
        expected = float(np.quantile(lc.fallback_scores, level))
        assert abs(q - expected) < 1e-10

    def test_combined_spatio_temporal(self):
        coords = np.array([[0.5, 0.5], [9.5, 9.5]])
        times = np.array(["2021-03-15", "2021-08-20"])
        # Build a combined local conformal that uses both
        rng = np.random.default_rng(2)
        n = 20
        scores_0 = rng.uniform(0.0, 0.2, n)
        scores_1 = rng.uniform(0.8, 1.5, n)
        coords_0 = rng.uniform(0.0, 1.0, (n, 2))
        coords_1 = rng.uniform(9.0, 10.0, (n, 2))
        times_0 = np.array(["2021-01-15"] * n)
        times_1 = np.array(["2021-07-15"] * n)

        all_coords = np.vstack([coords_0, coords_1])
        all_times = np.concatenate([times_0, times_1])
        all_block_idx = np.array([0] * n + [1] * n)

        from h2ml.evaluation.conformal import _build_context
        from sklearn.preprocessing import StandardScaler

        ctx = _build_context(all_coords, all_times)
        scaler = StandardScaler().fit(ctx)
        oof_ctx = scaler.transform(ctx)

        lc_st = LocalConformalCalibration(
            scores_by_block=[np.sort(scores_0), np.sort(scores_1)],
            oof_context_scaled=oof_ctx,
            oof_block_indices=all_block_idx,
            context_mean=scaler.mean_,
            context_std=scaler.scale_,
            fallback_scores=np.sort(np.concatenate([scores_0, scores_1])),
            metric="euclidean",
            min_block_n=3,
            has_coords=True,
            has_times=True,
        )

        q = lc_st.threshold(0.10, coords=coords, times=times)
        assert q.shape == (2,)
        assert q[1] > q[0]

    def test_predict_interval_uses_local_conformal(self):
        """predict_interval with coords returns varying widths; without coords is scalar."""
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 4))
        y = rng.standard_normal(50)
        est = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
        global_cal = ConformalCalibration(
            scores=np.sort(np.abs(rng.standard_normal(50))),
            n=50,
            task_type=TaskType.REGRESSION,
        )
        lc = _make_local_conformal()

        model = FinalModel(
            estimator=est,
            feature_names=[f"f{i}" for i in range(4)],
            task_type=TaskType.REGRESSION,
            conformal=global_cal,
            local_conformal=lc,
        )

        X_test = rng.standard_normal((10, 4))
        coords = np.column_stack(
            [
                rng.uniform(0, 1, 5).tolist() + rng.uniform(9, 10, 5).tolist(),
                rng.uniform(0, 1, 5).tolist() + rng.uniform(9, 10, 5).tolist(),
            ]
        )

        lower, upper = model.predict_interval(X_test, alpha=0.10, coords=coords)
        widths = upper - lower
        assert lower.shape == upper.shape == (10,)
        assert np.all(upper >= lower)
        assert widths.std() > 0, "Widths should vary with local calibration"

        # Without coords → scalar q, constant width
        lower_g, upper_g = model.predict_interval(X_test, alpha=0.10)
        widths_g = upper_g - lower_g
        assert np.allclose(widths_g, widths_g[0])

    def test_local_conformal_none_without_spatial_cv(self):
        """FinalModel built without spatial CV should have local_conformal=None."""
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(4)
        X = rng.standard_normal((50, 3))
        y = rng.standard_normal(50)
        est = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
        model = FinalModel(
            estimator=est,
            feature_names=["a", "b", "c"],
            task_type=TaskType.REGRESSION,
        )
        assert model.local_conformal is None


class TestTimeBinHelper:
    def test_month_range(self):
        dates = np.array([f"2021-{m:02d}-15" for m in range(1, 13)])
        bins = _time_bin(dates, "month")
        assert list(bins) == list(range(1, 13))

    def test_month_december(self):
        bins = _time_bin(np.array(["2021-12-01"]), "month")
        assert int(bins[0]) == 12

    def test_season_djf(self):
        for month in ["01", "02", "12"]:
            bins = _time_bin(np.array([f"2021-{month}-15"]), "season")
            assert int(bins[0]) == 0, f"Month {month} should be DJF (0)"

    def test_season_mam(self):
        for month in ["03", "04", "05"]:
            bins = _time_bin(np.array([f"2021-{month}-15"]), "season")
            assert int(bins[0]) == 1, f"Month {month} should be MAM (1)"

    def test_season_jja(self):
        for month in ["06", "07", "08"]:
            bins = _time_bin(np.array([f"2021-{month}-15"]), "season")
            assert int(bins[0]) == 2, f"Month {month} should be JJA (2)"

    def test_season_son(self):
        for month in ["09", "10", "11"]:
            bins = _time_bin(np.array([f"2021-{month}-15"]), "season")
            assert int(bins[0]) == 3, f"Month {month} should be SON (3)"


def _make_compound_local_conformal(min_block_n: int = 3) -> LocalConformalCalibration:
    """
    Two spatial blocks × two seasons (winter/summer).
    Block 0 winter: low error. Block 0 summer: medium error.
    Block 1 winter: medium error. Block 1 summer: high error.
    """
    rng = np.random.default_rng(10)
    n = 25

    scores_b0_win = rng.uniform(0.0, 0.1, n)
    scores_b0_sum = rng.uniform(0.3, 0.5, n)
    scores_b1_win = rng.uniform(0.3, 0.5, n)
    scores_b1_sum = rng.uniform(0.8, 1.2, n)

    coords_b0 = rng.uniform(0.0, 1.0, (n * 2, 2))
    coords_b1 = rng.uniform(9.0, 10.0, (n * 2, 2))
    times_win = np.array(["2021-01-15"] * n + ["2021-01-15"] * n)
    times_sum = np.array(["2021-07-15"] * n + ["2021-07-15"] * n)

    all_coords = np.vstack([coords_b0, coords_b1])  # (4n, 2)
    all_scores = np.concatenate([scores_b0_win, scores_b0_sum, scores_b1_win, scores_b1_sum])
    all_block_idx = np.array([0] * (2 * n) + [1] * (2 * n))
    all_times = np.concatenate([times_win[:n], times_sum[:n], times_win[n:], times_sum[n:]])

    from sklearn.preprocessing import StandardScaler
    from h2ml.evaluation.conformal import _build_context, _time_bin as tb

    ctx = _build_context(all_coords, all_times)
    scaler = StandardScaler().fit(ctx)
    oof_ctx = scaler.transform(ctx)

    bins = tb(all_times, "season")
    raw_compound: dict = {}
    for bi, tbin, score in zip(all_block_idx, bins, all_scores):
        raw_compound.setdefault((int(bi), int(tbin)), []).append(float(score))
    compound = {k: np.sort(np.array(v)) for k, v in raw_compound.items()}

    scores_by_block = [
        np.sort(np.concatenate([scores_b0_win, scores_b0_sum])),
        np.sort(np.concatenate([scores_b1_win, scores_b1_sum])),
    ]

    return LocalConformalCalibration(
        scores_by_block=scores_by_block,
        oof_context_scaled=oof_ctx,
        oof_block_indices=all_block_idx,
        context_mean=scaler.mean_,
        context_std=scaler.scale_,
        fallback_scores=np.sort(all_scores),
        metric="euclidean",
        min_block_n=min_block_n,
        has_coords=True,
        has_times=True,
        compound_scores=compound,
        time_bin_resolution="season",
    )


class TestTemporalBlockPartitioning:
    def test_same_location_different_season(self):
        lc = _make_compound_local_conformal()
        coords = np.array([[9.5, 9.5]])  # near block 1 (high error)
        q_win = lc.threshold(0.10, coords=coords, times=np.array(["2021-01-15"]))[0]
        q_sum = lc.threshold(0.10, coords=coords, times=np.array(["2021-07-15"]))[0]
        assert q_sum > q_win, "Summer should have higher threshold in high-error block"

    def test_fallback_to_spatial_when_compound_small(self):
        lc = _make_compound_local_conformal(min_block_n=30)  # spatial blocks have 50 samples
        lc = LocalConformalCalibration(**{**lc.__dict__, "min_compound_n": 30})  # compound cells (25) < 30
        coords = np.array([[0.5, 0.5]])  # block 0
        times = np.array(["2021-01-15"])
        q = lc.threshold(0.10, coords=coords, times=times)[0]
        # Should use spatial block scores (pooled across seasons), not compound
        spatial_scores = lc.scores_by_block[0]
        expected = float(
            np.quantile(spatial_scores, min(np.ceil(0.9 * (len(spatial_scores) + 1)) / len(spatial_scores), 1.0))
        )
        assert abs(q - expected) < 1e-9

    def test_fallback_to_global_when_spatial_small(self):
        lc = _make_compound_local_conformal(min_block_n=200)
        lc = LocalConformalCalibration(**{**lc.__dict__, "min_compound_n": 200})  # all cells < 200
        coords = np.array([[0.5, 0.5]])
        times = np.array(["2021-01-15"])
        q = lc.threshold(0.10, coords=coords, times=times)[0]
        fb = lc.fallback_scores
        expected = float(np.quantile(fb, min(np.ceil(0.9 * (len(fb) + 1)) / len(fb), 1.0)))
        assert abs(q - expected) < 1e-9

    def test_no_times_at_inference_uses_spatial_only(self):
        lc = _make_compound_local_conformal()
        coords = np.array([[0.5, 0.5]])
        # has_times=True but no times passed → time_bin=None → spatial fallback
        q = lc.threshold(0.10, coords=coords)[0]
        spatial_scores = lc.scores_by_block[0]
        expected = float(
            np.quantile(spatial_scores, min(np.ceil(0.9 * (len(spatial_scores) + 1)) / len(spatial_scores), 1.0))
        )
        assert abs(q - expected) < 1e-9


class TestLocalConformalSummary:
    def test_columns_and_spatial_rows(self):
        lc = _make_compound_local_conformal()
        df = lc.summary()
        assert list(df.columns) == ["block", "level", "time_bin", "bin_name", "n", "q", "used"]
        spatial = df[df["level"] == "spatial"]
        assert len(spatial) == len(lc.scores_by_block)
        # spatial rows carry no time bin
        assert spatial["time_bin"].isna().all()
        assert (spatial["bin_name"] == "all").all()

    def test_compound_rows_present_and_labelled(self):
        lc = _make_compound_local_conformal()
        df = lc.summary()
        compound = df[df["level"] == "compound"]
        # 2 blocks × 2 populated seasons (DJF, JJA)
        assert len(compound) == len(lc.compound_scores)
        assert set(compound["bin_name"]) == {"DJF", "JJA"}

    def test_no_compound_when_scores_absent(self):
        lc = _make_compound_local_conformal()
        lc = LocalConformalCalibration(**{**lc.__dict__, "compound_scores": None})
        df = lc.summary()
        assert (df["level"] == "spatial").all()

    def test_q_matches_conformal_quantile(self):
        lc = _make_compound_local_conformal()
        df = lc.summary(alpha=0.10)
        row = df[df["level"] == "spatial"].iloc[0]
        expected = _conformal_quantile(lc.scores_by_block[int(row["block"])], 0.10)
        assert abs(row["q"] - expected) < 1e-12

    def test_used_reflects_fallback_levels(self):
        # Cells (25) ≥ min_compound_n(5); blocks (50) ≥ min_block_n(3) → all "compound"/"block"
        lc = _make_compound_local_conformal()
        df = lc.summary()
        assert set(df[df["level"] == "compound"]["used"]) == {"compound"}
        assert set(df[df["level"] == "spatial"]["used"]) == {"block"}

        # Raise thresholds above every cell/block size → everything falls to global
        lc_global = LocalConformalCalibration(**{**lc.__dict__, "min_block_n": 200, "min_compound_n": 200})
        assert set(lc_global.summary()["used"]) == {"global"}

    def test_max_blocks_truncates(self):
        lc = _make_compound_local_conformal()
        df = lc.summary(max_blocks=1)
        assert set(df["block"]) == {0}


class TestEncodeTimesHelper:
    def test_shape_and_range(self):
        times = np.array(["2021-01-01", "2021-07-01", "2022-12-31"])
        enc = _encode_times(times)
        assert enc.shape == (3, 3)
        # sin/cos columns in [-1, 1]
        assert np.all(np.abs(enc[:, :2]) <= 1.0 + 1e-9)
        # year column
        assert list(enc[:, 2]) == [2021.0, 2021.0, 2022.0]

    def test_circular_adjacency(self):
        jan1 = _encode_times(np.array(["2021-01-01"]))
        dec31 = _encode_times(np.array(["2021-12-31"]))
        jul1 = _encode_times(np.array(["2021-07-01"]))
        d_jan_dec = np.linalg.norm(jan1[:, :2] - dec31[:, :2])
        d_jan_jul = np.linalg.norm(jan1[:, :2] - jul1[:, :2])
        assert d_jan_dec < d_jan_jul, "Jan and Dec should be closer than Jan and Jul"
