"""
tests/optimization/test_optimizer.py

Tests for:
    - run_study()      — public API: returns a correct Optuna study
    - _build_objective() — internal: scaling, fold count, intermediate reporting
    - optimize_all()   — batch wrapper: skips unknown/disabled models
"""

from __future__ import annotations

import numpy as np
import optuna
import pytest

from h2ml.optimization.optimizer import (
    _build_objective,
    _score_auc,
    _score_r2,
    _score_rmse,
    CLF_METRICS,
    REG_METRICS,
    optimize_all,
    run_study,
)
from h2ml.optimization.opt_params import ModelEntry
from h2ml.core.spatial_config import SpatialCVConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 80  # enough samples for StratifiedKFold with 2 splits on a binary target


@pytest.fixture
def clf_data() -> tuple[np.ndarray, np.ndarray]:
    """Small binary classification dataset."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((N, 4)), rng.integers(0, 2, N)


@pytest.fixture
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    """Small regression dataset."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((N, 4)), rng.standard_normal(N)


def _fast_clf_entry() -> ModelEntry:
    """ModelEntry backed by a tiny RandomForest — fast for testing."""
    from sklearn.ensemble import RandomForestClassifier

    return ModelEntry(
        model_cls=RandomForestClassifier,
        requires_scaling=False,
        param_fn=lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 5, 10, step=5),
            "max_depth": trial.suggest_int("max_depth", 2, 3),
            "random_state": 42,
        },
        opt_enabled=True,
    )


def _fast_reg_entry() -> ModelEntry:
    """ModelEntry backed by a tiny RandomForestRegressor — fast for testing."""
    from sklearn.ensemble import RandomForestRegressor

    return ModelEntry(
        model_cls=RandomForestRegressor,
        requires_scaling=False,
        param_fn=lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 5, 10, step=5),
            "max_depth": trial.suggest_int("max_depth", 2, 3),
            "random_state": 42,
        },
        opt_enabled=True,
    )


# ---------------------------------------------------------------------------
# TestRunStudy — integration tests for the public API
# ---------------------------------------------------------------------------


class TestRunStudy:
    def test_returns_optuna_study_classification(self, clf_data):
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=3,
            n_splits=2,
        )
        assert isinstance(study, optuna.Study)

    def test_returns_optuna_study_regression(self, reg_data):
        X, y = reg_data
        study = run_study("RandomForestRegressor", X, y, task="regression", n_trials=3, n_splits=2)
        assert isinstance(study, optuna.Study)

    def test_best_params_is_non_empty_dict(self, clf_data):
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=3,
            n_splits=2,
        )
        assert isinstance(study.best_params, dict)
        assert len(study.best_params) > 0

    def test_best_value_is_float(self, clf_data):
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=3,
            n_splits=2,
        )
        assert isinstance(study.best_value, float)

    def test_classification_best_value_in_auc_range(self, clf_data):
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=3,
            n_splits=2,
        )
        assert 0.0 <= study.best_value <= 1.0

    def test_regression_default_metric_is_r2(self, reg_data):
        """Default regression metric is R², which can range from -∞ to 1."""
        X, y = reg_data
        study = run_study("RandomForestRegressor", X, y, task="regression", n_trials=3, n_splits=2)
        assert study.best_value <= 1.0

    def test_study_direction_is_maximize(self, clf_data):
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=2,
            n_splits=2,
        )
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_default_study_name(self, clf_data):
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=2,
            n_splits=2,
        )
        assert study.study_name == "RandomForestClassifier_classification"

    def test_custom_study_name(self, clf_data):
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=2,
            n_splits=2,
            study_name="my_study",
        )
        assert study.study_name == "my_study"

    def test_n_trials_total_count(self, clf_data):
        """Total trials in the study (complete + pruned) must equal n_trials."""
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=4,
            n_splits=2,
        )
        assert len(study.trials) == 4

    def test_invalid_task_raises_value_error(self, clf_data):
        X, y = clf_data
        with pytest.raises(ValueError, match="task must be"):
            run_study("RandomForestClassifier", X, y, task="bad_task")

    def test_unknown_model_raises_key_error(self, clf_data):
        X, y = clf_data
        with pytest.raises(KeyError):
            run_study("NonExistentModel", X, y, task="classification")

    def test_opt_disabled_model_raises_value_error(self, clf_data):
        """LogisticRegression has opt_enabled=False in the registry."""
        X, y = clf_data
        with pytest.raises(ValueError, match="optimization disabled"):
            run_study("LogisticRegression", X, y, task="classification")

    def test_reproducibility_same_random_state(self, clf_data):
        X, y = clf_data
        kwargs = dict(task="classification", n_trials=5, n_splits=2, random_state=7)
        study_a = run_study("RandomForestClassifier", X, y, **kwargs)
        study_b = run_study("RandomForestClassifier", X, y, **kwargs)
        assert study_a.best_value == study_b.best_value
        assert study_a.best_params == study_b.best_params

    def test_different_random_states_may_differ(self, clf_data):
        """Sanity check: different seeds are not guaranteed identical."""
        X, y = clf_data
        study_a = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=5,
            n_splits=2,
            random_state=1,
        )
        study_b = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=5,
            n_splits=2,
            random_state=99,
        )
        # best_params may or may not differ — just confirm both finished
        assert study_a.best_params is not None
        assert study_b.best_params is not None

    def test_n_hpo_repeats_returns_study(self, clf_data):
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=4,
            n_splits=2,
            n_hpo_repeats=2,
        )
        assert isinstance(study, optuna.Study)
        assert isinstance(study.best_value, float)

    def test_n_hpo_repeats_splits_trials_evenly(self, clf_data):
        """Each repeat gets n_trials // n_hpo_repeats trials; returned study has that count."""
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=6,
            n_splits=2,
            n_hpo_repeats=3,
        )
        assert len(study.trials) == 2  # 6 // 3

    def test_n_hpo_repeats_study_name_has_suffix(self, clf_data):
        """With n_hpo_repeats > 1, each repeat's study name gets _r{i} suffix."""
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=4,
            n_splits=2,
            n_hpo_repeats=2,
        )
        assert "_r" in study.study_name

    def test_n_hpo_repeats_1_study_name_no_suffix(self, clf_data):
        """Default n_hpo_repeats=1 must not add a repeat suffix to the study name."""
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=4,
            n_splits=2,
            n_hpo_repeats=1,
            study_name="my_study",
        )
        assert study.study_name == "my_study"

    def test_n_hpo_repeats_reproducible(self, clf_data):
        X, y = clf_data
        kwargs = dict(
            task="classification",
            n_trials=4,
            n_splits=2,
            n_hpo_repeats=2,
            random_state=7,
        )
        study_a = run_study("RandomForestClassifier", X, y, **kwargs)
        study_b = run_study("RandomForestClassifier", X, y, **kwargs)
        assert study_a.best_value == study_b.best_value
        assert study_a.best_params == study_b.best_params


# ---------------------------------------------------------------------------
# TestBuildObjective — unit tests for the internal closure
# ---------------------------------------------------------------------------


class TestBuildObjective:
    def test_returns_callable(self, clf_data):
        X, y = clf_data
        obj = _build_objective(
            _fast_clf_entry(),
            X,
            y,
            "classification",
            n_splits=2,
            random_state=42,
            score_fn=_score_auc,
        )
        assert callable(obj)

    def test_objective_returns_float(self, clf_data):
        X, y = clf_data
        study = optuna.create_study(direction="maximize")
        study.optimize(
            _build_objective(
                _fast_clf_entry(),
                X,
                y,
                "classification",
                n_splits=2,
                random_state=42,
                score_fn=_score_auc,
            ),
            n_trials=1,
        )
        assert isinstance(study.best_value, float)

    def test_classification_score_in_unit_range(self, clf_data):
        X, y = clf_data
        study = optuna.create_study(direction="maximize")
        study.optimize(
            _build_objective(
                _fast_clf_entry(),
                X,
                y,
                "classification",
                n_splits=2,
                random_state=42,
                score_fn=_score_auc,
            ),
            n_trials=1,
        )
        assert 0.0 <= study.best_value <= 1.0

    def test_regression_r2_score_at_most_one(self, reg_data):
        """Default regression metric is R², bounded above by 1."""
        X, y = reg_data
        study = optuna.create_study(direction="maximize")
        study.optimize(
            _build_objective(
                _fast_reg_entry(),
                X,
                y,
                "regression",
                n_splits=2,
                random_state=42,
                score_fn=_score_r2,
            ),
            n_trials=1,
        )
        assert study.best_value <= 1.0

    def test_intermediate_values_reported_once_per_fold(self, clf_data):
        """trial.intermediate_values must have exactly n_splits entries."""
        X, y = clf_data
        n_splits = 3
        study = optuna.create_study(direction="maximize")
        study.optimize(
            _build_objective(
                _fast_clf_entry(),
                X,
                y,
                "classification",
                n_splits=n_splits,
                random_state=42,
                score_fn=_score_auc,
            ),
            n_trials=1,
        )
        assert len(study.trials[0].intermediate_values) == n_splits

    def test_intermediate_values_are_running_means(self, clf_data):
        """Each reported value is the mean of fold scores so far — must be in (0, 1]."""
        X, y = clf_data
        study = optuna.create_study(direction="maximize")
        study.optimize(
            _build_objective(
                _fast_clf_entry(),
                X,
                y,
                "classification",
                n_splits=3,
                random_state=42,
                score_fn=_score_auc,
            ),
            n_trials=1,
        )
        for v in study.trials[0].intermediate_values.values():
            assert 0.0 <= v <= 1.0

    def test_fit_called_n_splits_times_per_trial(self, clf_data):
        """The model must be fit exactly n_splits times per trial."""
        X, y = clf_data
        fit_calls = []

        from sklearn.ensemble import RandomForestClassifier

        class CountingRF(RandomForestClassifier):
            def fit(self, X, y, **kw):
                fit_calls.append(1)
                return super().fit(X, y, **kw)

        entry = ModelEntry(
            model_cls=CountingRF,
            requires_scaling=False,
            param_fn=lambda trial: {
                "n_estimators": 5,
                "max_depth": 2,
                "random_state": 42,
            },
            opt_enabled=True,
        )
        n_splits, n_trials = 3, 2
        study = optuna.create_study(direction="maximize")
        study.optimize(
            _build_objective(
                entry,
                X,
                y,
                "classification",
                n_splits=n_splits,
                random_state=42,
                score_fn=_score_auc,
            ),
            n_trials=n_trials,
        )
        assert sum(fit_calls) == n_splits * n_trials

    def test_scaling_applied_to_x_train_when_required(self, clf_data):
        """When requires_scaling=True, the X_train seen by the model should be standardised."""
        X, y = clf_data
        seen_X_trains: list[np.ndarray] = []

        class CapturingClassifier:
            def __init__(self, **kw):
                pass

            def fit(self, X, y):
                seen_X_trains.append(X.copy())
                return self

            def predict_proba(self, X):
                n = len(X)
                return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        entry = ModelEntry(
            model_cls=CapturingClassifier,
            requires_scaling=True,
            param_fn=lambda trial: {},
            opt_enabled=True,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(
            _build_objective(
                entry,
                X,
                y,
                "classification",
                n_splits=2,
                random_state=42,
                score_fn=_score_auc,
            ),
            n_trials=1,
        )

        assert len(seen_X_trains) == 2  # one per fold
        for X_train in seen_X_trains:
            assert abs(X_train.mean()) < 0.1  # near-zero mean after StandardScaler
            assert abs(X_train.std() - 1.0) < 0.1  # near-unit variance

    def test_no_scaling_when_not_required(self, clf_data):
        """When requires_scaling=False, raw X values must be passed to the model."""
        X, y = clf_data
        X_shifted = X + 100.0  # large shift so unscaled mean is clearly non-zero
        seen_X_trains: list[np.ndarray] = []

        from sklearn.ensemble import RandomForestClassifier

        class SpyRF(RandomForestClassifier):
            def fit(self, X, y, **kw):
                seen_X_trains.append(X.copy())
                return super().fit(X, y, **kw)

        entry = ModelEntry(
            model_cls=SpyRF,
            requires_scaling=False,
            param_fn=lambda trial: {
                "n_estimators": 5,
                "max_depth": 2,
                "random_state": 42,
            },
            opt_enabled=True,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(
            _build_objective(
                entry,
                X_shifted,
                y,
                "classification",
                n_splits=2,
                random_state=42,
                score_fn=_score_auc,
            ),
            n_trials=1,
        )

        assert len(seen_X_trains) > 0
        for X_train in seen_X_trains:
            assert X_train.mean() > 50.0  # raw offset preserved — not centred

    def test_none_param_fn_with_opt_enabled_raises_at_construction(self):
        """opt_enabled=True + param_fn=None is invalid — must raise at ModelEntry construction."""
        from sklearn.ensemble import RandomForestClassifier

        with pytest.raises(ValueError, match="param_fn"):
            ModelEntry(model_cls=RandomForestClassifier, param_fn=None, opt_enabled=True)


# ---------------------------------------------------------------------------
# TestOptimizeAll — batch wrapper
# ---------------------------------------------------------------------------


class TestOptimizeAll:
    def test_returns_dict(self, clf_data):
        X, y = clf_data
        result = optimize_all(
            ["RandomForestClassifier"],
            X,
            y,
            task="classification",
            n_trials=2,
            n_splits=2,
        )
        assert isinstance(result, dict)

    def test_known_model_present_in_result(self, clf_data):
        X, y = clf_data
        result = optimize_all(
            ["RandomForestClassifier"],
            X,
            y,
            task="classification",
            n_trials=2,
            n_splits=2,
        )
        assert "RandomForestClassifier" in result

    def test_result_values_are_studies(self, clf_data):
        X, y = clf_data
        result = optimize_all(
            ["RandomForestClassifier"],
            X,
            y,
            task="classification",
            n_trials=2,
            n_splits=2,
        )
        assert isinstance(result["RandomForestClassifier"], optuna.Study)

    def test_unknown_model_skipped(self, clf_data):
        X, y = clf_data
        result = optimize_all(
            ["RandomForestClassifier", "FakeModel"],
            X,
            y,
            task="classification",
            n_trials=2,
            n_splits=2,
        )
        assert "FakeModel" not in result
        assert "RandomForestClassifier" in result

    def test_opt_disabled_model_skipped(self, clf_data):
        X, y = clf_data
        result = optimize_all(
            ["RandomForestClassifier", "LogisticRegression"],
            X,
            y,
            task="classification",
            n_trials=2,
            n_splits=2,
        )
        assert "LogisticRegression" not in result
        assert "RandomForestClassifier" in result

    def test_multiple_models_all_returned(self, clf_data):
        X, y = clf_data
        result = optimize_all(
            ["RandomForestClassifier", "GradientBoostingClassifier"],
            X,
            y,
            task="classification",
            n_trials=2,
            n_splits=2,
        )
        assert set(result.keys()) == {
            "RandomForestClassifier",
            "GradientBoostingClassifier",
        }

    def test_empty_names_returns_empty_dict(self, clf_data):
        X, y = clf_data
        result = optimize_all([], X, y, task="classification")
        assert result == {}


# ---------------------------------------------------------------------------
# TestMetricSelection — metric parameter behaviour
# ---------------------------------------------------------------------------


class TestMetricSelection:
    def test_clf_metrics_registry_contains_expected_keys(self):
        assert set(CLF_METRICS) == {"AUC", "AUC_PR", "LogLoss", "F1", "Brier"}

    def test_reg_metrics_registry_contains_expected_keys(self):
        assert set(REG_METRICS) == {"R2", "MAE", "RMSE"}

    def test_invalid_clf_metric_raises_value_error(self, clf_data):
        X, y = clf_data
        with pytest.raises(ValueError, match="not valid for task"):
            run_study(
                "RandomForestClassifier",
                X,
                y,
                task="classification",
                metric="R2",
                n_trials=1,
            )

    def test_invalid_reg_metric_raises_value_error(self, reg_data):
        X, y = reg_data
        with pytest.raises(ValueError, match="not valid for task"):
            run_study(
                "RandomForestRegressor",
                X,
                y,
                task="regression",
                metric="AUC",
                n_trials=1,
            )

    def test_unknown_metric_raises_value_error(self, clf_data):
        X, y = clf_data
        with pytest.raises(ValueError, match="not valid for task"):
            run_study(
                "RandomForestClassifier",
                X,
                y,
                task="classification",
                metric="RMSE",
                n_trials=1,
            )

    def test_clf_auc_pr_returns_study(self, clf_data):
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            metric="AUC_PR",
            n_trials=3,
            n_splits=2,
        )
        assert isinstance(study, optuna.Study)
        assert 0.0 <= study.best_value <= 1.0

    def test_clf_f1_returns_study(self, clf_data):
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            metric="F1",
            n_trials=3,
            n_splits=2,
        )
        assert isinstance(study, optuna.Study)
        assert 0.0 <= study.best_value <= 1.0

    def test_clf_logloss_best_value_is_negative(self, clf_data):
        """LogLoss is negated internally — best_value must be ≤ 0."""
        X, y = clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            metric="LogLoss",
            n_trials=3,
            n_splits=2,
        )
        assert study.best_value <= 0.0

    def test_reg_rmse_best_value_is_non_positive(self, reg_data):
        """RMSE is negated internally — best_value must be ≤ 0."""
        X, y = reg_data
        study = run_study(
            "RandomForestRegressor",
            X,
            y,
            task="regression",
            metric="RMSE",
            n_trials=3,
            n_splits=2,
        )
        assert study.best_value <= 0.0

    def test_reg_mae_best_value_is_non_positive(self, reg_data):
        """MAE is negated internally — best_value must be ≤ 0."""
        X, y = reg_data
        study = run_study(
            "RandomForestRegressor",
            X,
            y,
            task="regression",
            metric="MAE",
            n_trials=3,
            n_splits=2,
        )
        assert study.best_value <= 0.0

    def test_optimize_all_passes_metric(self, clf_data):
        X, y = clf_data
        result = optimize_all(
            ["RandomForestClassifier"],
            X,
            y,
            task="classification",
            metric="F1",
            n_trials=2,
            n_splits=2,
        )
        assert "RandomForestClassifier" in result

    def test_score_rmse_is_non_positive(self, reg_data):
        """_score_rmse must always return a non-positive value."""
        X, y = reg_data
        study = optuna.create_study(direction="maximize")
        study.optimize(
            _build_objective(
                _fast_reg_entry(),
                X,
                y,
                "regression",
                n_splits=2,
                random_state=42,
                score_fn=_score_rmse,
            ),
            n_trials=1,
        )
        assert study.best_value <= 0.0


# ---------------------------------------------------------------------------
# TestSpatialCV — optimizer with spatial block splitter
# ---------------------------------------------------------------------------


class TestSpatialCV:
    N = 100

    @pytest.fixture
    def spatial_coords(self):
        rng = np.random.default_rng(5)
        return rng.uniform(0, 1, size=(self.N, 2))

    @pytest.fixture
    def small_clf_data(self):
        rng = np.random.default_rng(5)
        return rng.standard_normal((self.N, 4)), rng.integers(0, 2, self.N)

    @pytest.fixture
    def small_reg_data(self):
        rng = np.random.default_rng(5)
        return rng.standard_normal((self.N, 4)), rng.standard_normal(self.N)

    def test_build_objective_with_coords_returns_callable(self, small_clf_data, spatial_coords):
        X, y = small_clf_data
        obj = _build_objective(
            _fast_clf_entry(),
            X,
            y,
            "classification",
            n_splits=2,
            random_state=42,
            score_fn=_score_auc,
            coords=spatial_coords,
            spatial=SpatialCVConfig(n_blocks_per_fold=2),
        )
        assert callable(obj)

    def test_build_objective_with_coords_returns_float(self, small_clf_data, spatial_coords):
        X, y = small_clf_data
        study = optuna.create_study(direction="maximize")
        study.optimize(
            _build_objective(
                _fast_clf_entry(),
                X,
                y,
                "classification",
                n_splits=2,
                random_state=42,
                score_fn=_score_auc,
                coords=spatial_coords,
                spatial=SpatialCVConfig(n_blocks_per_fold=2),
            ),
            n_trials=1,
        )
        assert isinstance(study.best_value, float)

    def test_spatial_splitter_instantiated_when_coords_provided(self, small_clf_data, spatial_coords):
        from unittest.mock import patch

        X, y = small_clf_data
        with patch("h2ml.features.spatial_cv.SpatialBlockSplitter") as MockSplitter:
            mock_instance = MockSplitter.return_value
            # Provide enough fold tuples to prevent StopIteration
            n_splits = 2
            indices = np.arange(self.N)
            half = self.N // 2
            mock_instance.split.return_value = iter(
                [
                    (indices[:half], indices[half:]),
                    (indices[half:], indices[:half]),
                ]
            )
            _build_objective(
                _fast_clf_entry(),
                X,
                y,
                "classification",
                n_splits=n_splits,
                random_state=42,
                score_fn=_score_auc,
                coords=spatial_coords,
                spatial=SpatialCVConfig(n_blocks_per_fold=2),
            )
            MockSplitter.assert_called_once_with(
                coords=spatial_coords,
                n_splits=n_splits,
                n_blocks_per_fold=2,
                random_state=42,
                metric="euclidean",
            )

    def test_random_splitter_used_when_no_coords(self, small_clf_data):
        """When coords=None, _SPLITTER[task] is used instead of SpatialBlockSplitter."""
        from unittest.mock import patch, MagicMock
        import h2ml.optimization.optimizer as opt_module

        X, y = small_clf_data
        mock_cls = MagicMock()
        n_splits = 2
        indices = np.arange(self.N)
        half = self.N // 2
        mock_cls.return_value.split.return_value = iter(
            [
                (indices[:half], indices[half:]),
                (indices[half:], indices[:half]),
            ]
        )
        patched_splitter = {
            "classification": mock_cls,
            "regression": opt_module._SPLITTER["regression"],
        }
        with patch.object(opt_module, "_SPLITTER", patched_splitter):
            _build_objective(
                _fast_clf_entry(),
                X,
                y,
                "classification",
                n_splits=n_splits,
                random_state=42,
                score_fn=_score_auc,
                coords=None,
            )
        mock_cls.assert_called_once_with(n_splits=n_splits, shuffle=True, random_state=42)

    def test_run_study_with_coords_completes(self, small_clf_data, spatial_coords):
        X, y = small_clf_data
        study = run_study(
            "RandomForestClassifier",
            X,
            y,
            task="classification",
            n_trials=2,
            n_splits=2,
            coords=spatial_coords,
            spatial=SpatialCVConfig(n_blocks_per_fold=2),
        )
        assert isinstance(study, optuna.Study)
        assert isinstance(study.best_value, float)

    def test_run_study_regression_with_coords_completes(self, small_reg_data, spatial_coords):
        X, y = small_reg_data
        study = run_study(
            "RandomForestRegressor",
            X,
            y,
            task="regression",
            n_trials=2,
            n_splits=2,
            coords=spatial_coords,
            spatial=SpatialCVConfig(n_blocks_per_fold=2),
        )
        assert isinstance(study, optuna.Study)
