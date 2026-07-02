"""
tests/evaluation/test_compare.py

Tests for compare_results().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import pandas as pd

from h2ml.evaluation.compare import compare_results
from h2ml.pipeline.pipeline import PipelineResult
from h2ml.features.feature_store import PipelineData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(n: int = 50, f: int = 3) -> PipelineData:
    rng = np.random.default_rng(0)
    return PipelineData(
        X=rng.standard_normal((n, f)),
        feature_names=[f"feat_{i}" for i in range(f)],
        y=rng.integers(0, 2, n),
    )


def _make_agg_df(model: str = "RFC", stage: str = "default", auc: float = 0.85, brier: float = 0.15) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "Model": model,
                "Stage": stage,
                "AUC_Test_Mean": auc,
                "AUC_Test_Std": 0.02,
                "Brier_Test_Mean": brier,
                "Brier_Test_Std": 0.01,
            }
        ]
    )


def _make_fold_df(model: str = "RFC", stage: str = "default", n_folds: int = 5) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame([{"Model": model, "Stage": stage, "Fold": i, "AUC_Test": 0.85} for i in range(n_folds)])


def _make_result(
    model: str = "RFC",
    stage: str = "default",
    score_mean: float = 0.85,
    score_std: float = 0.03,
    n_folds: int = 5,
    n_features: int = 10,
    brier: float = 0.15,
    metric: str | None = "AUC",
) -> PipelineResult:
    result = PipelineResult(
        features=_make_store(f=n_features),
        best_model_name=model,
        best_stage=stage,
        best_feature_stage=stage if stage != "optimized" else "reduced",
        best_model_value=score_mean,
        best_model_std=score_std,
        step1_fold_df=_make_fold_df(model, "default", n_folds),
        step1_agg_df=_make_agg_df(model, "default", score_mean, brier),
        metric=metric,
    )
    return result


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestCompareResultsStructure:
    def test_returns_dataframe(self):
        df = compare_results([_make_result()])
        import pandas as pd

        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_result(self):
        df = compare_results([_make_result(), _make_result(score_mean=0.90)])
        assert len(df) == 2

    def test_expected_columns_present(self):
        df = compare_results([_make_result()])
        for col in (
            "Run",
            "Metric",
            "Best_Model",
            "Best_Stage",
            "Score_Mean",
            "Score_Std",
            "Conservative_Bound",
            "N_Features",
            "Completed_Steps",
        ):
            assert col in df.columns, f"Missing column: {col}"

    def test_empty_list_returns_empty_df(self):
        df = compare_results([])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestLabels:
    def test_default_labels(self):
        df = compare_results([_make_result(), _make_result()])
        assert list(df["Run"]) == ["Run_0", "Run_1"] or set(df["Run"]) == {
            "Run_0",
            "Run_1",
        }

    def test_custom_labels(self):
        df = compare_results(
            [_make_result(), _make_result()],
            labels=["baseline", "tuned"],
        )
        assert set(df["Run"]) == {"baseline", "tuned"}

    def test_mismatched_labels_raises(self):
        with pytest.raises(ValueError, match="labels"):
            compare_results([_make_result(), _make_result()], labels=["only_one"])


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSorting:
    def test_sorted_descending_by_default(self):
        r1 = _make_result(score_mean=0.80)
        r2 = _make_result(score_mean=0.92)
        df = compare_results([r1, r2], labels=["low", "high"])
        assert df.iloc[0]["Score_Mean"] >= df.iloc[1]["Score_Mean"]

    def test_sorted_ascending_for_minimize_metric(self):
        # RMSE is lower-is-better; METRIC_MINIMIZE derives ascending sort automatically
        r1 = _make_result(score_mean=0.80, metric="RMSE")
        r2 = _make_result(score_mean=0.92, metric="RMSE")
        df = compare_results([r1, r2], labels=["low", "high"])
        assert df.iloc[0]["Score_Mean"] <= df.iloc[1]["Score_Mean"]


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------


class TestFieldExtraction:
    def test_best_model_correct(self):
        df = compare_results([_make_result(model="LGBMClassifier")])
        assert df.iloc[0]["Best_Model"] == "LGBMClassifier"

    def test_best_stage_correct(self):
        df = compare_results([_make_result(stage="reduced")])
        assert df.iloc[0]["Best_Stage"] == "reduced"

    def test_score_mean_correct(self):
        df = compare_results([_make_result(score_mean=0.93)])
        assert df.iloc[0]["Score_Mean"] == pytest.approx(0.93)

    def test_score_std_correct(self):
        df = compare_results([_make_result(score_mean=0.85, score_std=0.04)])
        assert df.iloc[0]["Score_Std"] == pytest.approx(0.04)

    def test_n_features_from_default_store(self):
        df = compare_results([_make_result(n_features=7)])
        assert df.iloc[0]["N_Features"] == 7

    def test_n_features_from_reduced_store(self):
        result = _make_result(n_features=10, stage="reduced")
        result.features_reduced = _make_store(f=4)
        result.best_feature_stage = "reduced"
        df = compare_results([result])
        assert df.iloc[0]["N_Features"] == 4

    def test_completed_steps_for_step1_only(self):
        df = compare_results([_make_result()])
        assert 1 in df.iloc[0]["Completed_Steps"]

    def test_brier_mean_extracted(self):
        df = compare_results([_make_result(brier=0.12)])
        assert df.iloc[0]["Brier_Mean"] == pytest.approx(0.12)

    def test_brier_mean_none_when_column_absent(self):
        import pandas as pd

        result = _make_result()
        result.step1_agg_df = pd.DataFrame([{"Model": "RFC", "Stage": "default", "AUC_Test_Mean": 0.85}])
        df = compare_results([result])
        assert df.iloc[0]["Brier_Mean"] is None or (
            isinstance(df.iloc[0]["Brier_Mean"], float) and np.isnan(df.iloc[0]["Brier_Mean"])
        )


# ---------------------------------------------------------------------------
# Conservative_Bound computation
# ---------------------------------------------------------------------------


class TestConservativeBound:
    def test_cb_computed_correctly_maximize(self):
        # AUC is higher-is-better: CB = mean - std/sqrt(n)
        result = _make_result(score_mean=0.85, score_std=0.05, n_folds=5, metric="AUC")
        df = compare_results([result])
        expected = 0.85 - 0.05 / np.sqrt(5)
        assert df.iloc[0]["Conservative_Bound"] == pytest.approx(expected)

    def test_cb_computed_correctly_minimize(self):
        # RMSE is lower-is-better: CB = mean + std/sqrt(n)
        result = _make_result(score_mean=5.82, score_std=0.95, n_folds=5, metric="RMSE")
        df = compare_results([result])
        expected = 5.82 + 0.95 / np.sqrt(5)
        assert df.iloc[0]["Conservative_Bound"] == pytest.approx(expected)

    def test_cb_none_when_std_missing(self):
        result = _make_result()
        result.best_model_std = None
        df = compare_results([result])
        assert df.iloc[0]["Conservative_Bound"] is None or (
            isinstance(df.iloc[0]["Conservative_Bound"], float) and np.isnan(df.iloc[0]["Conservative_Bound"])
        )

    def test_cb_penalises_higher_variance(self):
        # For a higher-is-better metric, stable model should have higher CB
        stable = _make_result(score_mean=0.88, score_std=0.01, n_folds=5)
        volatile = _make_result(score_mean=0.88, score_std=0.10, n_folds=5)
        df = compare_results([stable, volatile], labels=["stable", "volatile"])
        cb = df.set_index("Run")["Conservative_Bound"]
        assert cb["stable"] > cb["volatile"]

    def test_cb_minimize_selects_stable_over_volatile(self):
        # For RMSE: B (mean=5.60, std=2.30) should lose to A (mean=5.82, std=0.95)
        r_a = _make_result(score_mean=5.82, score_std=0.95, n_folds=5, metric="RMSE")
        r_b = _make_result(score_mean=5.60, score_std=2.30, n_folds=5, metric="RMSE")
        df = compare_results([r_a, r_b], labels=["A", "B"])
        assert df.iloc[0]["Run"] == "A"


# ---------------------------------------------------------------------------
# OOF Brier
# ---------------------------------------------------------------------------


class TestOOFBrier:
    def test_oof_brier_none_when_no_cv_result(self):
        df = compare_results([_make_result()])
        val = df.iloc[0]["OOF_Brier"]
        assert val is None or (isinstance(val, float) and np.isnan(val))

    def test_oof_brier_computed_from_predictions(self):
        from h2ml.pipeline.cv import CVResult, FoldResult
        from h2ml.pipeline.base import TaskType

        result = _make_result()
        cv = CVResult(model_name="RFC", task_type=TaskType.CLASSIFICATION)
        cv.folds.append(
            FoldResult(
                fold_id=0,
                model_name="RFC",
                y_train=np.array([0, 1, 0, 1]),
                y_test=np.array([0, 1]),
                y_pred_train=np.array([0, 1, 0, 1]),
                y_pred_test=np.array([0, 1]),
                y_prob_train=np.array([0.1, 0.9, 0.2, 0.8]),
                y_prob_test=np.array([0.1, 0.9]),
                train_idx=np.array([0, 1, 2, 3]),
                test_idx=np.array([4, 5]),
                fit_time=0.0,
            )
        )
        result.step4_cv_result = cv
        df = compare_results([result])
        val = df.iloc[0]["OOF_Brier"]
        assert val is not None and not np.isnan(val)
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Metric column and mixed-metric warning
# ---------------------------------------------------------------------------


class TestMetricColumn:
    def test_metric_column_present(self):
        df = compare_results([_make_result(metric="AUC")])
        assert "Metric" in df.columns

    def test_metric_value_matches_result(self):
        df = compare_results([_make_result(metric="R2")], labels=["reg"])
        assert df.iloc[0]["Metric"] == "R2"

    def test_metric_none_when_not_set(self):
        df = compare_results([_make_result(metric=None)])
        assert df.iloc[0]["Metric"] is None

    def test_same_metrics_no_warning(self):
        from loguru import logger

        warned = []
        handler = logger.add(lambda msg: warned.append(str(msg)), level="WARNING")
        try:
            compare_results(
                [_make_result(metric="AUC"), _make_result(metric="AUC")],
                labels=["a", "b"],
            )
        finally:
            logger.remove(handler)
        assert not any("different metrics" in w for w in warned)

    def test_mixed_metrics_raises(self):
        with pytest.raises(ValueError, match="different metrics"):
            compare_results(
                [_make_result(metric="AUC"), _make_result(metric="R2")],
                labels=["clf", "reg"],
            )

    def test_mixed_metrics_error_names_the_metrics(self):
        with pytest.raises(ValueError) as exc_info:
            compare_results(
                [_make_result(metric="AUC"), _make_result(metric="RMSE")],
                labels=["a", "b"],
            )
        msg = str(exc_info.value)
        assert "AUC" in msg and "RMSE" in msg

    def test_mixed_metrics_error_suggests_metric_param(self):
        with pytest.raises(ValueError, match="metric="):
            compare_results(
                [_make_result(metric="AUC"), _make_result(metric="R2")],
                labels=["a", "b"],
            )
