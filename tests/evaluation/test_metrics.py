"""
tests/test_evaluation/test_metrics.py

Tests for evaluation/metrics.py.

Coverage:
    RunMetadata:
        - to_dict() excludes None values
        - to_dict() includes only set fields

    _classification_metrics:
        - returns expected keys
        - raises when probabilities are missing

    _regression_metrics:
        - returns expected keys
        - RMSE is sqrt of MSE

    compute_metrics:
        - returns one row per fold for classifier
        - returns one row per fold for regressor
        - attaches metadata columns when provided
        - does not attach metadata columns when not provided
        - adds Gap columns for classification
        - adds Gap columns for regression
        - raises on unsupported task type

    compute_metrics_all:
        - returns one row per fold per model
        - concatenates correctly across models

    aggregate_metrics:
        - returns one row per model
        - produces _Mean and _Std suffixed columns
        - excludes Fold column from aggregation
        - includes metadata grouping columns when present

    select_best:
        - returns correct model name by max metric
        - returns correct model name by min metric
        - raises when metric column is missing
        - returns dict with expected keys
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from h2ml.pipeline.base import TaskType
from h2ml.pipeline.cv import CVResult, FoldResult
from h2ml.evaluation.metrics import (
    RunMetadata,
    aggregate_metrics,
    compute_metrics,
    compute_metrics_all,
    select_best,
)


# ---------------------------------------------------------------------------
# Helpers — build deterministic CVResult objects
# ---------------------------------------------------------------------------


def _make_classification_fold(fold_id: int, model_name: str = "TestClf") -> FoldResult:
    rng = np.random.default_rng(fold_id)
    n_test, n_train = 20, 80
    return FoldResult(
        fold_id=fold_id,
        model_name=model_name,
        y_train=rng.integers(0, 2, size=n_train),
        y_test=rng.integers(0, 2, size=n_test),
        y_pred_train=rng.integers(0, 2, size=n_train),
        y_pred_test=rng.integers(0, 2, size=n_test),
        y_prob_train=rng.uniform(0, 1, size=n_train),
        y_prob_test=rng.uniform(0, 1, size=n_test),
        fit_time=0.1 * (fold_id + 1),
    )


def _make_regression_fold(fold_id: int, model_name: str = "TestReg") -> FoldResult:
    rng = np.random.default_rng(fold_id)
    n_test, n_train = 20, 80
    return FoldResult(
        fold_id=fold_id,
        model_name=model_name,
        y_train=rng.standard_normal(n_train),
        y_test=rng.standard_normal(n_test),
        y_pred_train=rng.standard_normal(n_train),
        y_pred_test=rng.standard_normal(n_test),
        fit_time=0.1 * (fold_id + 1),
    )


def _make_cv_result(
    task: TaskType, n_folds: int = 5, model_name: Optional[str] = None
) -> CVResult:
    name = model_name or ("TestClf" if task == TaskType.CLASSIFICATION else "TestReg")
    make_fold = (
        _make_classification_fold
        if task == TaskType.CLASSIFICATION
        else _make_regression_fold
    )
    result = CVResult(model_name=name, task_type=task)
    for i in range(n_folds):
        result.folds.append(make_fold(i, model_name=name))
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clf_cv_result() -> CVResult:
    return _make_cv_result(TaskType.CLASSIFICATION)


@pytest.fixture
def reg_cv_result() -> CVResult:
    return _make_cv_result(TaskType.REGRESSION)


@pytest.fixture
def metadata() -> RunMetadata:
    return RunMetadata(stage="default", target="species_a", batch="batch_01")


@pytest.fixture
def metadata_partial() -> RunMetadata:
    """Metadata with some None fields — to_dict() should exclude them."""
    return RunMetadata(stage="optimized")


# ---------------------------------------------------------------------------
# RunMetadata
# ---------------------------------------------------------------------------


class TestRunMetadata:
    def test_to_dict_excludes_none_values(self, metadata_partial):
        d = metadata_partial.to_dict()
        assert "Target" not in d
        assert "Batch" not in d
        assert "Schema" not in d

    def test_to_dict_includes_set_fields(self, metadata):
        d = metadata.to_dict()
        assert d["Stage"] == "default"
        assert d["Target"] == "species_a"
        assert d["Batch"] == "batch_01"

    def test_to_dict_empty_when_all_none(self):
        m = RunMetadata()
        d = m.to_dict()
        # stage has a default value so it will be present
        assert "Stage" in d

    def test_default_stage_is_default(self):
        m = RunMetadata()
        assert m.stage == "default"


# ---------------------------------------------------------------------------
# compute_metrics — classification
# ---------------------------------------------------------------------------


class TestComputeMetricsClassification:
    def test_returns_one_row_per_fold(self, clf_cv_result):
        df = compute_metrics(clf_cv_result)
        assert len(df) == clf_cv_result.n_folds

    def test_contains_expected_columns(self, clf_cv_result):
        df = compute_metrics(clf_cv_result)
        expected = {
            "Model",
            "Fold",
            "AUC_Test",
            "AUC_Train",
            "AUC_PR_Test",
            "AUC_PR_Train",
            "LogLoss_Test",
            "LogLoss_Train",
            "F1_Test",
            "F1_Train",
            "Fit_Time",
        }
        assert expected.issubset(set(df.columns))

    def test_model_column_matches_name(self, clf_cv_result):
        df = compute_metrics(clf_cv_result)
        assert (df["Model"] == "TestClf").all()

    def test_fold_ids_are_sequential(self, clf_cv_result):
        df = compute_metrics(clf_cv_result)
        assert list(df["Fold"]) == list(range(clf_cv_result.n_folds))

    def test_auc_values_in_valid_range(self, clf_cv_result):
        df = compute_metrics(clf_cv_result)
        assert df["AUC_Test"].between(0, 1).all()
        assert df["AUC_Train"].between(0, 1).all()

    def test_logloss_values_are_positive(self, clf_cv_result):
        df = compute_metrics(clf_cv_result)
        assert (df["LogLoss_Test"] >= 0).all()

    def test_gap_columns_added(self, clf_cv_result):
        df = compute_metrics(clf_cv_result)
        for col in ("AUC_Gap", "AUC_PR_Gap", "F1_Gap", "LogLoss_Gap", "Brier_Gap"):
            assert col in df.columns, f"Missing gap column: {col}"

    def test_score_gap_is_train_minus_test(self, clf_cv_result):
        """For maximize metrics (AUC, F1): gap = train - test."""
        df = compute_metrics(clf_cv_result)
        pd.testing.assert_series_equal(
            df["AUC_Gap"].reset_index(drop=True),
            (df["AUC_Train"] - df["AUC_Test"]).reset_index(drop=True),
            check_names=False,
        )

    def test_error_gap_is_test_minus_train(self, clf_cv_result):
        """For minimize metrics (LogLoss, Brier): gap = test - train so positive = overfitting."""
        df = compute_metrics(clf_cv_result)
        pd.testing.assert_series_equal(
            df["LogLoss_Gap"].reset_index(drop=True),
            (df["LogLoss_Test"] - df["LogLoss_Train"]).reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            df["Brier_Gap"].reset_index(drop=True),
            (df["Brier_Test"] - df["Brier_Train"]).reset_index(drop=True),
            check_names=False,
        )

    def test_brier_test_column_present(self, clf_cv_result):
        df = compute_metrics(clf_cv_result)
        assert "Brier_Test" in df.columns
        assert "Brier_Train" in df.columns

    def test_brier_values_in_valid_range(self, clf_cv_result):
        df = compute_metrics(clf_cv_result)
        assert (df["Brier_Test"] >= 0).all()
        assert (df["Brier_Test"] <= 1).all()

    def test_brier_perfect_predictor_is_zero(self):
        """A model predicting exact probabilities should have Brier score 0."""
        from h2ml.pipeline.cv import CVResult, FoldResult
        from h2ml.evaluation.metrics import compute_metrics

        result = CVResult(model_name="Perfect", task_type=TaskType.CLASSIFICATION)
        result.folds.append(
            FoldResult(
                fold_id=0,
                model_name="Perfect",
                y_train=np.array([0, 1]),
                y_test=np.array([0, 1]),
                y_pred_train=np.array([0, 1]),
                y_pred_test=np.array([0, 1]),
                y_prob_train=np.array([0.0, 1.0]),
                y_prob_test=np.array([0.0, 1.0]),
                train_idx=np.array([0, 1]),
                test_idx=np.array([0, 1]),
                fit_time=0.0,
            )
        )
        df = compute_metrics(result)
        assert df["Brier_Test"].iloc[0] == pytest.approx(0.0)

    def test_metadata_columns_attached(self, clf_cv_result, metadata):
        df = compute_metrics(clf_cv_result, metadata=metadata)
        assert "Stage" in df.columns
        assert "Target" in df.columns
        assert "Batch" in df.columns

    def test_metadata_values_correct(self, clf_cv_result, metadata):
        df = compute_metrics(clf_cv_result, metadata=metadata)
        assert (df["Stage"] == "default").all()
        assert (df["Target"] == "species_a").all()

    def test_no_metadata_columns_when_not_provided(self, clf_cv_result):
        df = compute_metrics(clf_cv_result)
        assert "Stage" not in df.columns
        assert "Batch" not in df.columns

    def test_raises_when_probabilities_missing(self):
        """CVResult with classification type but no y_prob should raise."""
        result = CVResult(model_name="NoProb", task_type=TaskType.CLASSIFICATION)
        result.folds.append(
            FoldResult(
                fold_id=0,
                model_name="NoProb",
                y_train=np.array([0, 1, 0, 1]),
                y_test=np.array([0, 1]),
                y_pred_train=np.array([0, 1, 0, 1]),
                y_pred_test=np.array([0, 1]),
                y_prob_test=None,  # missing
                y_prob_train=None,  # missing
            )
        )
        with pytest.raises(ValueError, match="no probabilities"):
            compute_metrics(result)


# ---------------------------------------------------------------------------
# compute_metrics — regression
# ---------------------------------------------------------------------------


class TestComputeMetricsRegression:
    def test_returns_one_row_per_fold(self, reg_cv_result):
        df = compute_metrics(reg_cv_result)
        assert len(df) == reg_cv_result.n_folds

    def test_contains_expected_columns(self, reg_cv_result):
        df = compute_metrics(reg_cv_result)
        expected = {
            "Model",
            "Fold",
            "R2_Test",
            "R2_Train",
            "MAE_Test",
            "MAE_Train",
            "RMSE_Test",
            "RMSE_Train",
            "Fit_Time",
        }
        assert expected.issubset(set(df.columns))

    def test_rmse_is_sqrt_of_mse(self, reg_cv_result):
        """RMSE should equal sqrt(MSE) — verify against sklearn directly."""
        from sklearn.metrics import mean_squared_error

        df = compute_metrics(reg_cv_result)
        fold = reg_cv_result.folds[0]
        expected_rmse = np.sqrt(mean_squared_error(fold.y_test, fold.y_pred_test))
        assert abs(df.iloc[0]["RMSE_Test"] - expected_rmse) < 1e-9

    def test_mae_is_non_negative(self, reg_cv_result):
        df = compute_metrics(reg_cv_result)
        assert (df["MAE_Test"] >= 0).all()

    def test_gap_columns_added(self, reg_cv_result):
        df = compute_metrics(reg_cv_result)
        for col in ("R2_Gap", "MAE_Gap", "RMSE_Gap"):
            assert col in df.columns, f"Missing gap column: {col}"

    def test_r2_gap_is_train_minus_test(self, reg_cv_result):
        """R2 is a score (maximize): gap = train - test."""
        df = compute_metrics(reg_cv_result)
        pd.testing.assert_series_equal(
            df["R2_Gap"].reset_index(drop=True),
            (df["R2_Train"] - df["R2_Test"]).reset_index(drop=True),
            check_names=False,
        )

    def test_rmse_gap_is_test_minus_train(self, reg_cv_result):
        """RMSE is an error (minimize): gap = test - train, positive = overfitting."""
        df = compute_metrics(reg_cv_result)
        pd.testing.assert_series_equal(
            df["RMSE_Gap"].reset_index(drop=True),
            (df["RMSE_Test"] - df["RMSE_Train"]).reset_index(drop=True),
            check_names=False,
        )

    def test_mae_gap_is_test_minus_train(self, reg_cv_result):
        """MAE is an error (minimize): gap = test - train."""
        df = compute_metrics(reg_cv_result)
        pd.testing.assert_series_equal(
            df["MAE_Gap"].reset_index(drop=True),
            (df["MAE_Test"] - df["MAE_Train"]).reset_index(drop=True),
            check_names=False,
        )

    def test_fit_time_matches_fold(self, reg_cv_result):
        df = compute_metrics(reg_cv_result)
        for i, fold in enumerate(reg_cv_result.folds):
            assert df.iloc[i]["Fit_Time"] == pytest.approx(fold.fit_time)


# ---------------------------------------------------------------------------
# compute_metrics — edge cases
# ---------------------------------------------------------------------------


class TestComputeMetricsEdgeCases:
    def test_raises_on_transform_task_type(self):
        result = CVResult(model_name="bad", task_type=TaskType.TRANSFORM)
        result.folds.append(_make_classification_fold(0, "bad"))
        with pytest.raises(ValueError, match="Unsupported task type"):
            compute_metrics(result)


# ---------------------------------------------------------------------------
# compute_metrics_all
# ---------------------------------------------------------------------------


class TestComputeMetricsAll:
    def test_returns_rows_for_all_models(self):
        results = [
            _make_cv_result(TaskType.CLASSIFICATION, n_folds=5, model_name="ModelA"),
            _make_cv_result(TaskType.CLASSIFICATION, n_folds=5, model_name="ModelB"),
        ]
        df = compute_metrics_all(results)
        assert len(df) == 10  # 5 folds × 2 models

    def test_model_names_present(self):
        results = [
            _make_cv_result(TaskType.CLASSIFICATION, model_name="ModelA"),
            _make_cv_result(TaskType.CLASSIFICATION, model_name="ModelB"),
        ]
        df = compute_metrics_all(results)
        assert set(df["Model"]) == {"ModelA", "ModelB"}

    def test_metadata_attached_to_all_rows(self, metadata):
        results = [
            _make_cv_result(TaskType.CLASSIFICATION, model_name="ModelA"),
            _make_cv_result(TaskType.CLASSIFICATION, model_name="ModelB"),
        ]
        df = compute_metrics_all(results, metadata=metadata)
        assert (df["Stage"] == "default").all()

    def test_single_model_matches_compute_metrics(self, clf_cv_result):
        df_all = compute_metrics_all([clf_cv_result])
        df_single = compute_metrics(clf_cv_result)
        pd.testing.assert_frame_equal(
            df_all.reset_index(drop=True),
            df_single.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_returns_one_row_per_model(self):
        results = [
            _make_cv_result(TaskType.CLASSIFICATION, model_name="ModelA"),
            _make_cv_result(TaskType.CLASSIFICATION, model_name="ModelB"),
        ]
        fold_df = compute_metrics_all(results)
        agg_df = aggregate_metrics(fold_df)
        assert len(agg_df) == 2

    def test_mean_columns_exist(self):
        fold_df = compute_metrics(_make_cv_result(TaskType.CLASSIFICATION))
        agg_df = aggregate_metrics(fold_df)
        assert "AUC_Test_Mean" in agg_df.columns
        assert "AUC_Train_Mean" in agg_df.columns

    def test_std_columns_exist(self):
        fold_df = compute_metrics(_make_cv_result(TaskType.CLASSIFICATION))
        agg_df = aggregate_metrics(fold_df)
        assert "AUC_Test_Std" in agg_df.columns

    def test_fold_column_not_in_aggregated(self):
        fold_df = compute_metrics(_make_cv_result(TaskType.CLASSIFICATION))
        agg_df = aggregate_metrics(fold_df)
        assert "Fold" not in agg_df.columns

    def test_mean_value_matches_manual(self):
        fold_df = compute_metrics(_make_cv_result(TaskType.CLASSIFICATION))
        agg_df = aggregate_metrics(fold_df)
        expected_mean = fold_df["AUC_Test"].mean()
        assert agg_df.iloc[0]["AUC_Test_Mean"] == pytest.approx(expected_mean)

    def test_std_value_matches_manual(self):
        fold_df = compute_metrics(_make_cv_result(TaskType.CLASSIFICATION))
        agg_df = aggregate_metrics(fold_df)
        expected_std = fold_df["AUC_Test"].std()
        assert agg_df.iloc[0]["AUC_Test_Std"] == pytest.approx(expected_std)

    def test_metadata_columns_used_as_groupby(self, metadata):
        fold_df = compute_metrics(
            _make_cv_result(TaskType.CLASSIFICATION, model_name="ModelA"),
            metadata=metadata,
        )
        agg_df = aggregate_metrics(fold_df)
        assert "Stage" in agg_df.columns

    def test_regression_aggregation(self):
        fold_df = compute_metrics(_make_cv_result(TaskType.REGRESSION))
        agg_df = aggregate_metrics(fold_df)
        assert "R2_Test_Mean" in agg_df.columns
        assert "RMSE_Test_Mean" in agg_df.columns


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------


class TestSelectBest:
    @pytest.fixture
    def agg_df(self) -> pd.DataFrame:
        """Aggregated DataFrame with two models, known AUC values."""
        return pd.DataFrame(
            {
                "Model": ["ModelA", "ModelB"],
                "AUC_Test_Mean": [0.85, 0.91],
                "AUC_Test_Std": [0.02, 0.03],
                "RMSE_Test_Mean": [0.40, 0.35],
            }
        )

    def test_selects_max_by_default(self, agg_df):
        result = select_best(agg_df, metric="AUC_Test_Mean")
        assert result["model_name"] == "ModelB"

    def test_selects_min_when_minimize_true(self, agg_df):
        result = select_best(agg_df, metric="RMSE_Test_Mean", minimize=True)
        assert result["model_name"] == "ModelB"

    def test_returns_correct_metric_value(self, agg_df):
        result = select_best(agg_df, metric="AUC_Test_Mean")
        assert result["value"] == pytest.approx(0.91)

    def test_returns_expected_keys(self, agg_df):
        result = select_best(agg_df, metric="AUC_Test_Mean")
        assert set(result.keys()) == {"model_name", "metric", "value", "row"}

    def test_metric_field_matches_input(self, agg_df):
        result = select_best(agg_df, metric="AUC_Test_Mean")
        assert result["metric"] == "AUC_Test_Mean"

    def test_raises_when_metric_not_found(self, agg_df):
        with pytest.raises(ValueError, match="not found"):
            select_best(agg_df, metric="NonExistentMetric")

    # LCB / UCB selection

    def test_lcb_flips_winner_when_std_dominates(self):
        """High-mean model with much larger std should lose under LCB."""
        df = pd.DataFrame(
            {
                "Model": ["Stable", "Volatile"],
                "AUC_Test_Mean": [0.90, 0.92],
                "AUC_Test_Std": [0.01, 0.10],
            }
        )
        # Without LCB: Volatile wins (0.92 > 0.90)
        assert select_best(df, metric="AUC_Test_Mean")["model_name"] == "Volatile"
        # LCB scores: Stable=0.90-0.01/√5≈0.8955, Volatile=0.92-0.10/√5≈0.8753
        assert (
            select_best(df, metric="AUC_Test_Mean", n_folds=5)["model_name"] == "Stable"
        )

    def test_lcb_preserves_winner_when_std_equal(self):
        """When std is equal, higher mean still wins."""
        df = pd.DataFrame(
            {
                "Model": ["A", "B"],
                "AUC_Test_Mean": [0.85, 0.91],
                "AUC_Test_Std": [0.02, 0.02],
            }
        )
        assert select_best(df, metric="AUC_Test_Mean", n_folds=5)["model_name"] == "B"

    def test_ucb_minimize_flips_winner(self):
        """Low-error model with much larger std should lose under UCB (minimize)."""
        df = pd.DataFrame(
            {
                "Model": ["Stable", "Volatile"],
                "RMSE_Test_Mean": [0.40, 0.36],
                "RMSE_Test_Std": [0.02, 0.15],
            }
        )
        # Without UCB: Volatile wins (0.36 < 0.40)
        assert (
            select_best(df, metric="RMSE_Test_Mean", minimize=True)["model_name"]
            == "Volatile"
        )
        # UCB scores: Stable=0.40+0.02/√5≈0.409, Volatile=0.36+0.15/√5≈0.427
        assert (
            select_best(df, metric="RMSE_Test_Mean", minimize=True, n_folds=5)[
                "model_name"
            ]
            == "Stable"
        )

    def test_lcb_value_is_raw_mean(self):
        """Returned value is still the raw mean, not the adjusted score."""
        df = pd.DataFrame(
            {
                "Model": ["A"],
                "AUC_Test_Mean": [0.88],
                "AUC_Test_Std": [0.05],
            }
        )
        result = select_best(df, metric="AUC_Test_Mean", n_folds=5)
        assert result["value"] == pytest.approx(0.88)

    def test_lcb_no_std_col_falls_back_to_mean(self):
        """When the _Std column is absent, n_folds is ignored and raw mean is used."""
        df = pd.DataFrame(
            {
                "Model": ["A", "B"],
                "AUC_Test_Mean": [0.85, 0.91],
            }
        )
        result = select_best(df, metric="AUC_Test_Mean", n_folds=5)
        assert result["model_name"] == "B"

    def test_lcb_n_folds_none_uses_raw_mean(self, agg_df):
        """n_folds=None (default) uses raw mean — same as omitting n_folds."""
        assert (
            select_best(agg_df, metric="AUC_Test_Mean", n_folds=None)["model_name"]
            == select_best(agg_df, metric="AUC_Test_Mean")["model_name"]
        )


# ---------------------------------------------------------------------------
# _classification_metrics — multiclass
# ---------------------------------------------------------------------------


def _make_multiclass_fold(fold_id: int, n_classes: int = 3) -> FoldResult:
    """Build a FoldResult with multiclass 2D probability arrays."""
    rng = np.random.default_rng(fold_id)
    n_test, n_train = 20, 80
    classes = np.arange(n_classes)

    y_train = rng.integers(0, n_classes, size=n_train)
    y_test = rng.integers(0, n_classes, size=n_test)

    # Raw probabilities — normalise to sum to 1 per row
    raw_train = rng.uniform(0, 1, size=(n_train, n_classes))
    raw_test = rng.uniform(0, 1, size=(n_test, n_classes))
    prob_train = raw_train / raw_train.sum(axis=1, keepdims=True)
    prob_test = raw_test / raw_test.sum(axis=1, keepdims=True)

    return FoldResult(
        fold_id=fold_id,
        model_name="MulticlassModel",
        y_train=y_train,
        y_test=y_test,
        y_pred_train=rng.integers(0, n_classes, size=n_train),
        y_pred_test=rng.integers(0, n_classes, size=n_test),
        y_prob_train=prob_train,
        y_prob_test=prob_test,
        fit_time=0.1,
    )


class TestMulticlassClassificationMetrics:
    @pytest.fixture
    def multiclass_cv_result(self) -> CVResult:
        result = CVResult(
            model_name="MulticlassModel", task_type=TaskType.CLASSIFICATION
        )
        for i in range(5):
            result.folds.append(_make_multiclass_fold(i))
        return result

    def test_compute_metrics_returns_expected_keys(self, multiclass_cv_result):
        df = compute_metrics(multiclass_cv_result)
        expected = {
            "Model",
            "Fold",
            "AUC_Test",
            "AUC_Train",
            "AUC_PR_Test",
            "AUC_PR_Train",
            "LogLoss_Test",
            "LogLoss_Train",
            "F1_Test",
            "F1_Train",
            "Fit_Time",
        }
        assert expected.issubset(set(df.columns))

    def test_compute_metrics_returns_one_row_per_fold(self, multiclass_cv_result):
        df = compute_metrics(multiclass_cv_result)
        assert len(df) == multiclass_cv_result.n_folds

    def test_auc_values_in_valid_range(self, multiclass_cv_result):
        df = compute_metrics(multiclass_cv_result)
        assert df["AUC_Test"].between(0, 1).all()

    def test_f1_values_in_valid_range(self, multiclass_cv_result):
        df = compute_metrics(multiclass_cv_result)
        assert df["F1_Test"].between(0, 1).all()

    def test_logloss_is_positive(self, multiclass_cv_result):
        df = compute_metrics(multiclass_cv_result)
        assert (df["LogLoss_Test"] > 0).all()
