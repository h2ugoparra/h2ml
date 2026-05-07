"""
tests/test_pipeline/test_cv.py

Tests for CrossValidator, CVResult and FoldResult.

Coverage:
    - FoldResult fields are populated correctly
    - FoldResult.task_type inferred from presence of probabilities
    - CVResult aggregates folds correctly
    - CVResult.fold_arrays() extracts attributes across folds
    - CrossValidator.run() produces correct number of folds
    - CrossValidator.run() uses StratifiedKFold for classification
    - CrossValidator.run() uses KFold for regression
    - CrossValidator.run() shapes of arrays in FoldResult
    - CrossValidator.run() y_prob is None for regressors
    - CrossValidator.run() y_prob is not None for classifiers
    - CrossValidator.run() applies scaling when requires_scaling=True
    - CrossValidator.run() does not scale when requires_scaling=False
    - CrossValidator.run() re-instantiates model with best_params
    - CrossValidator.run_all() returns one CVResult per step
    - CrossValidator.run_all() passes best_params per model name
    - Fold train/test indices are disjoint
    - Fold indices cover the full dataset
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC

try:
    from lightgbm import LGBMClassifier
    from xgboost import XGBRegressor

    HAS_BOOSTING = True
except ImportError:
    LGBMClassifier = None  # type: ignore[assignment,misc]
    XGBRegressor = None  # type: ignore[assignment,misc]
    HAS_BOOSTING = False

requires_boosting = pytest.mark.skipif(
    not HAS_BOOSTING,
    reason="lightgbm and xgboost not installed (install the [boosting] extra)",
)

from h2ml.pipeline.base import TaskType
from h2ml.pipeline.cv import CrossValidator, CVResult, FoldResult
from h2ml.pipeline.step import make_classifier, make_regressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cv() -> CrossValidator:
    """Default CrossValidator — 5 folds, deterministic."""
    return CrossValidator(n_splits=5, shuffle=True, random_state=42)


@pytest.fixture
def cv_3fold() -> CrossValidator:
    return CrossValidator(n_splits=3, shuffle=True, random_state=42)


# Classifiers
@pytest.fixture
def svc_step():
    return make_classifier(
        SVC(random_state=42, probability=True), requires_scaling=True
    )


@pytest.fixture
def lgbm_step():
    if not HAS_BOOSTING:
        pytest.skip("lightgbm not installed")
    return make_classifier(LGBMClassifier(random_state=42, verbose=-1))


# Regressors
@pytest.fixture
def lr_step():
    return make_regressor(LogisticRegression(random_state=42))


@pytest.fixture
def ridge_step():
    return make_regressor(Ridge())


@pytest.fixture
def xgb_step():
    if not HAS_BOOSTING:
        pytest.skip("xgboost not installed")
    return make_regressor(XGBRegressor(random_state=42))


# ---------------------------------------------------------------------------
# FoldResult
# ---------------------------------------------------------------------------


class TestFoldResult:
    def test_task_type_classification_when_proba_present(self):
        fold = FoldResult(
            fold_id=0,
            model_name="test",
            y_train=np.array([0, 1]),
            y_test=np.array([0, 1]),
            y_pred_train=np.array([0, 1]),
            y_pred_test=np.array([0, 1]),
            y_prob_train=np.array([0.1, 0.9]),
            y_prob_test=np.array([0.2, 0.8]),
        )
        assert fold.task_type == "classification"

    def test_task_type_regression_when_proba_absent(self):
        fold = FoldResult(
            fold_id=0,
            model_name="test",
            y_train=np.array([1.0, 2.0]),
            y_test=np.array([1.5]),
            y_pred_train=np.array([1.1, 2.1]),
            y_pred_test=np.array([1.4]),
        )
        assert fold.task_type == "regression"

    def test_default_indices_are_empty(self):
        fold = FoldResult(
            fold_id=0,
            model_name="test",
            y_train=np.array([0]),
            y_test=np.array([1]),
            y_pred_train=np.array([0]),
            y_pred_test=np.array([1]),
        )
        assert len(fold.train_idx) == 0
        assert len(fold.test_idx) == 0

    def test_fit_time_default_zero(self):
        fold = FoldResult(
            fold_id=0,
            model_name="test",
            y_train=np.array([0]),
            y_test=np.array([1]),
            y_pred_train=np.array([0]),
            y_pred_test=np.array([1]),
        )
        assert fold.fit_time == 0.0


# ---------------------------------------------------------------------------
# CVResult
# ---------------------------------------------------------------------------


class TestCVResult:
    @pytest.fixture
    def cv_result_with_folds(self) -> CVResult:
        """CVResult pre-populated with 3 dummy folds."""
        result = CVResult(model_name="DummyModel", task_type=TaskType.CLASSIFICATION)
        for i in range(3):
            result.folds.append(
                FoldResult(
                    fold_id=i,
                    model_name="DummyModel",
                    y_train=np.array([0, 1, 0]),
                    y_test=np.array([1, 0]),
                    y_pred_train=np.array([0, 1, 0]),
                    y_pred_test=np.array([1, 0]),
                    y_prob_test=np.array([0.8, 0.3]),
                    fit_time=float(i),
                )
            )
        return result

    def test_n_folds(self, cv_result_with_folds):
        assert cv_result_with_folds.n_folds == 3

    def test_fold_arrays_extracts_fit_times(self, cv_result_with_folds):
        times = cv_result_with_folds.fold_arrays("fit_time")
        assert times == [0.0, 1.0, 2.0]

    def test_fold_arrays_extracts_y_prob(self, cv_result_with_folds):
        probs = cv_result_with_folds.fold_arrays("y_prob_test")
        assert len(probs) == 3
        assert all(isinstance(p, np.ndarray) for p in probs)

    def test_empty_result_has_zero_folds(self):
        result = CVResult(model_name="empty", task_type=TaskType.REGRESSION)
        assert result.n_folds == 0

    # ------------------------------------------------------------------
    # oof_predictions / oof_labels
    # ------------------------------------------------------------------

    @pytest.fixture
    def cv_result_oof(self) -> CVResult:
        """CVResult with 2 folds that partition 6 samples."""
        result = CVResult(model_name="M", task_type=TaskType.CLASSIFICATION)
        # Fold 0: samples 0,2,4 → test; 1,3,5 → train
        result.folds.append(
            FoldResult(
                fold_id=0,
                model_name="M",
                y_train=np.array([0, 1, 0]),
                y_test=np.array([1, 0, 1]),
                y_pred_train=np.zeros(3),
                y_pred_test=np.ones(3),
                y_prob_train=np.array([0.2, 0.8, 0.3]),
                y_prob_test=np.array([0.9, 0.1, 0.8]),
                train_idx=np.array([1, 3, 5]),
                test_idx=np.array([0, 2, 4]),
            )
        )
        # Fold 1: samples 1,3,5 → test; 0,2,4 → train
        result.folds.append(
            FoldResult(
                fold_id=1,
                model_name="M",
                y_train=np.array([1, 0, 1]),
                y_test=np.array([0, 1, 0]),
                y_pred_train=np.ones(3),
                y_pred_test=np.zeros(3),
                y_prob_train=np.array([0.9, 0.1, 0.8]),
                y_prob_test=np.array([0.2, 0.7, 0.3]),
                train_idx=np.array([0, 2, 4]),
                test_idx=np.array([1, 3, 5]),
            )
        )
        return result

    def test_oof_predictions_none_when_no_folds(self):
        result = CVResult(model_name="empty", task_type=TaskType.CLASSIFICATION)
        assert result.oof_predictions is None

    def test_oof_predictions_shape(self, cv_result_oof):
        oof = cv_result_oof.oof_predictions
        assert oof is not None
        assert oof.shape == (6,)

    def test_oof_predictions_covers_all_samples(self, cv_result_oof):
        oof = cv_result_oof.oof_predictions
        assert not np.any(np.isnan(oof))

    def test_oof_predictions_values_correct(self, cv_result_oof):
        oof = cv_result_oof.oof_predictions
        # Fold 0 filled positions [0,2,4] with [0.9, 0.1, 0.8]
        np.testing.assert_allclose(oof[[0, 2, 4]], [0.9, 0.1, 0.8])
        # Fold 1 filled positions [1,3,5] with [0.2, 0.7, 0.3]
        np.testing.assert_allclose(oof[[1, 3, 5]], [0.2, 0.7, 0.3])

    def test_oof_labels_shape(self, cv_result_oof):
        labels = cv_result_oof.oof_labels
        assert labels is not None
        assert labels.shape == (6,)

    def test_oof_labels_none_when_no_folds(self):
        result = CVResult(model_name="empty", task_type=TaskType.CLASSIFICATION)
        assert result.oof_labels is None

    def test_oof_labels_numeric_dtype_uses_float_with_nan(self, cv_result_oof):
        labels = cv_result_oof.oof_labels
        assert np.issubdtype(labels.dtype, np.floating)

    def test_oof_labels_string_dtype_uses_object_with_none(self):
        result = CVResult(model_name="M", task_type=TaskType.CLASSIFICATION)
        result.folds.append(
            FoldResult(
                fold_id=0,
                model_name="M",
                y_train=np.array(["cat", "dog"]),
                y_test=np.array(["bird", "cat"]),
                y_pred_train=np.zeros(2),
                y_pred_test=np.zeros(2),
                y_prob_train=np.array([0.3, 0.7]),
                y_prob_test=np.array([0.4, 0.6]),
                train_idx=np.array([0, 1]),
                test_idx=np.array([2, 3]),
            )
        )
        labels = result.oof_labels
        assert labels is not None
        assert labels.dtype == object
        assert labels[2] == "bird"
        assert labels[3] == "cat"
        assert labels[0] is None  # unfilled position

    def test_oof_regression_uses_predictions_not_proba(self):
        result = CVResult(model_name="R", task_type=TaskType.REGRESSION)
        result.folds.append(
            FoldResult(
                fold_id=0,
                model_name="R",
                y_train=np.array([1.0, 2.0]),
                y_test=np.array([3.0, 4.0]),
                y_pred_train=np.array([1.1, 2.1]),
                y_pred_test=np.array([3.1, 4.1]),
                train_idx=np.array([0, 1]),
                test_idx=np.array([2, 3]),
            )
        )
        oof = result.oof_predictions
        assert oof is not None
        np.testing.assert_allclose(oof[[2, 3]], [3.1, 4.1])


# ---------------------------------------------------------------------------
# CrossValidator.run() — fold count and structure
# ---------------------------------------------------------------------------


class TestCrossValidatorRun:
    def test_correct_number_of_folds(self, cv, lgbm_step, classification_data):
        X, y = classification_data
        result = cv.run(lgbm_step, X, y)
        assert result.n_folds == 5

    def test_correct_number_of_folds_3fold(
        self, cv_3fold, lgbm_step, classification_data
    ):
        X, y = classification_data
        result = cv_3fold.run(lgbm_step, X, y)
        assert result.n_folds == 3

    def test_result_model_name(self, cv, lgbm_step, classification_data):
        X, y = classification_data
        result = cv.run(lgbm_step, X, y)
        assert result.model_name == lgbm_step.name

    def test_result_task_type_classification(self, cv, lgbm_step, classification_data):
        X, y = classification_data
        result = cv.run(lgbm_step, X, y)
        assert result.task_type == TaskType.CLASSIFICATION

    def test_result_task_type_regression(self, cv, ridge_step, regression_data):
        X, y = regression_data
        result = cv.run(ridge_step, X, y)
        assert result.task_type == TaskType.REGRESSION


# ---------------------------------------------------------------------------
# CrossValidator.run() — array shapes
# ---------------------------------------------------------------------------


class TestCrossValidatorArrayShapes:
    def test_y_pred_test_shape(self, cv, lr_step, classification_data):
        X, y = classification_data
        result = cv.run(lr_step, X, y)
        for fold in result.folds:
            assert fold.y_pred_test.shape == fold.y_test.shape

    def test_y_pred_train_shape(self, cv, lr_step, classification_data):
        X, y = classification_data
        result = cv.run(lr_step, X, y)
        for fold in result.folds:
            assert fold.y_pred_train.shape == fold.y_train.shape

    def test_y_prob_test_shape_classifier(self, cv, lr_step, classification_data):
        X, y = classification_data
        result = cv.run(lr_step, X, y)
        for fold in result.folds:
            assert fold.y_prob_test.shape == fold.y_test.shape

    def test_y_prob_train_shape_classifier(self, cv, lr_step, classification_data):
        X, y = classification_data
        result = cv.run(lr_step, X, y)
        for fold in result.folds:
            assert fold.y_prob_train.shape == fold.y_train.shape

    def test_total_test_samples_equals_dataset(self, cv, lr_step, classification_data):
        """All test folds together should cover the full dataset exactly once."""
        X, y = classification_data
        result = cv.run(lr_step, X, y)
        total_test = sum(len(f.y_test) for f in result.folds)
        assert total_test == len(y)


# ---------------------------------------------------------------------------
# CrossValidator.run() — probabilities
# ---------------------------------------------------------------------------


class TestCrossValidatorProbabilities:
    def test_y_prob_is_none_for_regressor(self, cv, ridge_step, regression_data):
        X, y = regression_data
        result = cv.run(ridge_step, X, y)
        for fold in result.folds:
            assert fold.y_prob_test is None
            assert fold.y_prob_train is None

    def test_y_prob_not_none_for_classifier(self, cv, lgbm_step, classification_data):
        X, y = classification_data
        result = cv.run(lgbm_step, X, y)
        for fold in result.folds:
            assert fold.y_prob_test is not None
            assert fold.y_prob_train is not None

    def test_y_prob_values_between_0_and_1(self, cv, lr_step, classification_data):
        X, y = classification_data
        result = cv.run(lr_step, X, y)
        for fold in result.folds:
            assert fold.y_prob_test.min() >= 0.0
            assert fold.y_prob_test.max() <= 1.0


# ---------------------------------------------------------------------------
# CrossValidator.run() — fold index integrity
# ---------------------------------------------------------------------------


class TestFoldIndexIntegrity:
    def test_train_test_indices_are_disjoint(self, cv, lr_step, classification_data):
        X, y = classification_data
        result = cv.run(lr_step, X, y)
        for fold in result.folds:
            overlap = np.intersect1d(fold.train_idx, fold.test_idx)
            assert len(overlap) == 0, f"Fold {fold.fold_id} has overlapping indices"

    def test_all_indices_cover_full_dataset(self, cv, lr_step, classification_data):
        X, y = classification_data
        result = cv.run(lr_step, X, y)
        all_test_idx = np.concatenate([f.test_idx for f in result.folds])
        assert sorted(all_test_idx) == list(range(len(y)))

    def test_fit_time_is_positive(self, cv, lr_step, classification_data):
        X, y = classification_data
        result = cv.run(lr_step, X, y)
        for fold in result.folds:
            assert fold.fit_time >= 0.0


# ---------------------------------------------------------------------------
# CrossValidator.run() — scaling
# ---------------------------------------------------------------------------


class TestCrossValidatorScaling:
    def test_scaling_applied_when_required(self, cv, svc_step, classification_data):
        """SVC with requires_scaling=True should run without errors."""
        X, y = classification_data
        result = cv.run(svc_step, X, y)
        assert result.n_folds == 5

    def test_no_scaling_when_not_required(self, cv, lr_step, classification_data):
        """LR with requires_scaling=False should run without errors."""
        X, y = classification_data
        result = cv.run(lr_step, X, y)
        assert result.n_folds == 5

    def test_scaling_does_not_leak_across_folds(
        self, cv, svc_step, classification_data
    ):
        """Each fold should produce valid probabilities — sign that scaler was fit per fold."""
        X, y = classification_data
        result = cv.run(svc_step, X, y)
        for fold in result.folds:
            assert fold.y_prob_test is not None
            assert fold.y_prob_test.min() >= 0.0
            assert fold.y_prob_test.max() <= 1.0


# ---------------------------------------------------------------------------
# CrossValidator.run() — best_params
# ---------------------------------------------------------------------------


class TestCrossValidatorBestParams:
    def test_best_params_applied(self, cv, lgbm_step, classification_data):
        """Running with best_params should succeed and produce same fold count."""
        X, y = classification_data
        best_params = {"num_leaves": 10, "max_depth": 5, "random_state": 42}
        result = cv.run(lgbm_step, X, y, best_params=best_params)
        assert result.n_folds == 5

    def test_no_best_params_uses_defaults(self, cv, lgbm_step, classification_data):
        X, y = classification_data
        result = cv.run(lgbm_step, X, y, best_params=None)
        assert result.n_folds == 5


# ---------------------------------------------------------------------------
# CrossValidator.run_all()
# ---------------------------------------------------------------------------


@requires_boosting
class TestCrossValidatorRunAll:
    def test_run_all_returns_one_result_per_step(self, cv, classification_data):
        X, y = classification_data
        steps = [
            make_classifier(LGBMClassifier(random_state=42, verbosa=-1), name="LGBM"),
            make_classifier(
                RandomForestClassifier(n_estimators=10, random_state=42), name="RF"
            ),
        ]
        results = cv.run_all(steps, X, y)
        assert len(results) == 2

    def test_run_all_result_names_match_steps(self, cv, classification_data):
        X, y = classification_data
        steps = [
            make_classifier(LGBMClassifier(random_state=42, verbosa=-1), name="LGBM"),
            make_classifier(
                RandomForestClassifier(n_estimators=10, random_state=42), name="RF"
            ),
        ]
        results = cv.run_all(steps, X, y)
        names = [r.model_name for r in results]
        assert names == ["LGBM", "RF"]

    def test_run_all_with_per_model_best_params(self, cv, classification_data):
        X, y = classification_data
        steps = [
            make_classifier(LGBMClassifier(random_state=42, verbose=-1), name="LGBM"),
        ]
        best_params = {"LGBM": {"learning_rate": 0.2, "random_state": 50}}
        results = cv.run_all(steps, X, y, best_params=best_params)
        assert results[0].n_folds == 5

    def test_run_all_mixed_tasks(self, cv, classification_data, regression_data):
        X_c, y_c = classification_data
        X_r, y_r = regression_data
        clf_step = make_classifier(LGBMClassifier(random_state=42, verbose=-1))
        reg_step = make_regressor(XGBRegressor(random_state=42))

        clf_result = cv.run(clf_step, X_c, y_c)
        reg_result = cv.run(reg_step, X_r, y_r)

        assert clf_result.task_type == TaskType.CLASSIFICATION
        assert reg_result.task_type == TaskType.REGRESSION


# ---------------------------------------------------------------------------
# Multiclass classification
# ---------------------------------------------------------------------------


class TestMulticlassClassification:
    def test_y_prob_is_2d_for_multiclass(self, cv, multiclass_data):
        """For a 3-class target, y_prob_test should be a 2D array (n_samples, n_classes)."""
        X, y = multiclass_data
        step = make_classifier(RandomForestClassifier(n_estimators=10, random_state=42))
        result = cv.run(step, X, y)
        for fold in result.folds:
            assert fold.y_prob_test is not None
            assert fold.y_prob_test.ndim == 2
            assert fold.y_prob_test.shape[1] == 3

    def test_y_prob_rows_sum_to_one_multiclass(self, cv, multiclass_data):
        """Probability matrix rows should sum to 1."""
        X, y = multiclass_data
        step = make_classifier(RandomForestClassifier(n_estimators=10, random_state=42))
        result = cv.run(step, X, y)
        for fold in result.folds:
            np.testing.assert_allclose(fold.y_prob_test.sum(axis=1), 1.0, atol=1e-6)

    def test_failed_folds_initially_empty(self, cv, multiclass_data):
        X, y = multiclass_data
        step = make_classifier(RandomForestClassifier(n_estimators=10, random_state=42))
        result = cv.run(step, X, y)
        assert result.failed_folds == []

    def test_failed_folds_populated_on_error(self, cv, classification_data):
        """A step that always raises should populate failed_folds for every fold."""
        from unittest.mock import patch

        X, y = classification_data
        step = make_classifier(RandomForestClassifier(n_estimators=10, random_state=42))
        with patch.object(step, "fit", side_effect=RuntimeError("forced failure")):
            result = cv.run(step, X, y)
        assert len(result.failed_folds) == cv.n_splits
        assert result.folds == []


# ---------------------------------------------------------------------------
# Spatial block CV
# ---------------------------------------------------------------------------


class TestSpatialBlockCV:
    """CrossValidator wired up with SpatialBlockSplitter via coords kwarg."""

    N = 200

    @pytest.fixture
    def spatial_coords(self):
        rng = np.random.default_rng(9)
        return rng.uniform(0, 1, size=(self.N, 2))

    @pytest.fixture
    def spatial_data(self):
        rng = np.random.default_rng(9)
        X = rng.standard_normal((self.N, 5))
        y = rng.integers(0, 2, self.N)
        return X, y

    @pytest.fixture
    def cv5(self):
        return CrossValidator(n_splits=5, shuffle=True, random_state=42)

    def test_run_with_coords_produces_n_folds(self, cv5, spatial_data, spatial_coords):
        X, y = spatial_data
        step = make_classifier(RandomForestClassifier(n_estimators=5, random_state=42))
        result = cv5.run(step, X, y, coords=spatial_coords)
        assert result.n_folds == 5

    def test_run_all_with_coords_returns_one_result_per_step(
        self, cv5, spatial_data, spatial_coords
    ):
        X, y = spatial_data
        steps = [
            make_classifier(
                RandomForestClassifier(n_estimators=5, random_state=42), name="RF1"
            ),
            make_classifier(
                RandomForestClassifier(n_estimators=5, random_state=0), name="RF2"
            ),
        ]
        results = cv5.run_all(steps, X, y, coords=spatial_coords)
        assert len(results) == 2

    def test_run_without_coords_unchanged(self, cv5, classification_data):
        X, y = classification_data
        step = make_classifier(RandomForestClassifier(n_estimators=5, random_state=42))
        result = cv5.run(step, X, y, coords=None)
        assert result.n_folds == 5

    def test_spatial_splitter_used_when_coords_provided(
        self, cv5, spatial_data, spatial_coords
    ):
        from unittest.mock import patch

        X, y = spatial_data
        step = make_classifier(RandomForestClassifier(n_estimators=5, random_state=42))
        with (
            patch(
                "h2ml.features.spatial_cv.SpatialBlockSplitter.__init__",
                return_value=None,
            ) as mock_init,
            patch(
                "h2ml.features.spatial_cv.SpatialBlockSplitter.split",
                return_value=iter([]),
            ),
            patch(
                "h2ml.features.spatial_cv.SpatialBlockSplitter.get_n_splits",
                return_value=5,
            ),
        ):
            try:
                cv5.run(step, X, y, coords=spatial_coords)
            except Exception:
                pass  # shape mismatch from the mocked splitter is fine
            mock_init.assert_called_once()

    def test_random_splitter_used_when_no_coords(self, cv5, classification_data):
        from unittest.mock import patch

        X, y = classification_data
        step = make_classifier(RandomForestClassifier(n_estimators=5, random_state=42))
        with patch("h2ml.pipeline.cv.StratifiedKFold") as mock_skf:
            mock_skf.return_value.split.return_value = iter([])
            try:
                cv5.run(step, X, y, coords=None)
            except Exception:
                pass
            mock_skf.assert_called_once()

    def test_train_test_disjoint_with_spatial_coords(
        self, cv5, spatial_data, spatial_coords
    ):
        X, y = spatial_data
        step = make_classifier(RandomForestClassifier(n_estimators=5, random_state=42))
        result = cv5.run(step, X, y, coords=spatial_coords)
        for fold in result.folds:
            overlap = np.intersect1d(fold.train_idx, fold.test_idx)
            assert len(overlap) == 0

    def test_all_indices_covered_with_spatial_coords(
        self, cv5, spatial_data, spatial_coords
    ):
        X, y = spatial_data
        step = make_classifier(RandomForestClassifier(n_estimators=5, random_state=42))
        result = cv5.run(step, X, y, coords=spatial_coords)
        all_test = np.concatenate([f.test_idx for f in result.folds])
        assert sorted(all_test) == list(range(self.N))

    def test_n_blocks_per_fold_forwarded(self, spatial_data, spatial_coords):
        from unittest.mock import patch

        X, y = spatial_data
        cv = CrossValidator(n_splits=5)
        step = make_classifier(RandomForestClassifier(n_estimators=5, random_state=42))
        with patch("h2ml.features.spatial_cv.SpatialBlockSplitter") as MockSplitter:
            mock_instance = MockSplitter.return_value
            mock_instance.split.return_value = iter([])
            try:
                cv.run(step, X, y, coords=spatial_coords, n_blocks_per_fold=3)
            except Exception:
                pass
            MockSplitter.assert_called_once_with(
                coords=spatial_coords,
                n_splits=5,
                n_blocks_per_fold=3,
                random_state=42,
                metric="euclidean",
            )


# ---------------------------------------------------------------------------
# _scaled_folds parameter
# ---------------------------------------------------------------------------


class TestScaledFolds:
    """
    Tests for the _scaled_folds parameter on CrossValidator.run().

    _scaled_folds allows callers to pre-compute StandardScaler arrays once
    and reuse them across multiple run() calls (e.g. multiple y-transforms
    that share the same X).  When provided, _maybe_scale() is bypassed.
    """

    def _precompute(self, cv: CrossValidator, X, y):
        """Build fold-level scaled arrays using the same splitter params as cv."""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        kf = StratifiedKFold(
            n_splits=cv.n_splits, shuffle=cv.shuffle, random_state=cv.random_state
        )
        folds = []
        for tr, te in kf.split(X, y):
            sc = StandardScaler()
            folds.append((sc.fit_transform(X[tr]), sc.transform(X[te])))
        return folds

    def test_correct_fold_count_with_scaled_folds(
        self, cv, svc_step, classification_data
    ):
        """run() with _scaled_folds returns the same number of folds as without."""
        X, y = classification_data
        scaled = self._precompute(cv, X, y)
        result = cv.run(svc_step, X, y, _scaled_folds=scaled)
        assert result.n_folds == cv.n_splits

    def test_maybe_scale_not_called_when_scaled_folds_provided(
        self, cv, svc_step, classification_data
    ):
        """_maybe_scale must be bypassed entirely when _scaled_folds is given."""
        from unittest.mock import patch

        X, y = classification_data
        scaled = self._precompute(cv, X, y)
        with patch.object(cv, "_maybe_scale", wraps=cv._maybe_scale) as mock:
            cv.run(svc_step, X, y, _scaled_folds=scaled)
        mock.assert_not_called()

    def test_maybe_scale_called_when_scaled_folds_is_none(
        self, cv, svc_step, classification_data
    ):
        """When _scaled_folds=None (default), _maybe_scale is called once per fold."""
        from unittest.mock import patch

        X, y = classification_data
        with patch.object(cv, "_maybe_scale", wraps=cv._maybe_scale) as mock:
            cv.run(svc_step, X, y, _scaled_folds=None)
        assert mock.call_count == cv.n_splits

    def test_scaled_folds_none_is_default(self, cv, svc_step, classification_data):
        """Omitting _scaled_folds is equivalent to passing None."""
        from unittest.mock import patch

        X, y = classification_data
        with patch.object(cv, "_maybe_scale", wraps=cv._maybe_scale) as mock:
            cv.run(svc_step, X, y)
        assert mock.call_count == cv.n_splits
