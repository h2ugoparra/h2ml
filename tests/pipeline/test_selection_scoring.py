"""
tests/pipeline/test_selection_scoring.py

Tests for pooled regression selection and the constant-prediction guard: pooled
scores, zero-variance fold R2, select_best, ranking_robustness, the pipeline steps,
the Optuna objective, and resuming results saved before the option.

Step 2 (SHAP) is mocked as in test_holdout.py unless a test exercises it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from h2ml.core.base import TaskType
from h2ml.core.cv_result import CVResult, FoldResult
from h2ml.core.feature_store import PipelineData
from h2ml.evaluation.metrics import (
    POOLED_COL,
    _is_flat,
    aggregate_metrics,
    compute_metrics,
    select_best,
    selection_col,
)
from h2ml.evaluation.ranking import flat_warnings, ranking_robustness
from h2ml.features.selector import FeatureSelector
from h2ml.optimization.opt_params import ModelEntry
from h2ml.optimization.optimizer import _build_objective, _neg_rmse, _r2, _score_r2, _score_rmse
from h2ml.persistence.result_io import load_result, save_result
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig
from h2ml.pipeline.step import make_regressor

FEATURE_NAMES = [f"feat_{i}" for i in range(4)]
N = 120


def _fold(y_test, y_pred, fold_id=0, model="M", y_train=None) -> FoldResult:
    y_test, y_pred = np.asarray(y_test, float), np.asarray(y_pred, float)
    y_train = np.array([0.0, 1.0, 2.0, 3.0]) if y_train is None else np.asarray(y_train, float)
    return FoldResult(
        fold_id=fold_id, model_name=model, y_train=y_train, y_test=y_test, y_pred_train=y_train, y_pred_test=y_pred
    )


def _cv(folds: list[FoldResult], model="M") -> CVResult:
    return CVResult(model_name=model, task_type=TaskType.REGRESSION, folds=folds)


def _store(seed: int = 0) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, len(FEATURE_NAMES)))
    y = 2 * X[:, 0] + 0.1 * rng.standard_normal(N)
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _mock_selector(selected: list[str]):
    mock_sel = MagicMock(spec=FeatureSelector)
    mock_sel.selected_features_ = selected
    mock_sel.fit_transform.side_effect = lambda store, y=None: store.select(selected)
    mock_sel.transform.side_effect = lambda store: store.select(selected)
    return patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=mock_sel)


def _pipeline(models: list, metric: str = "RMSE", **kwargs) -> H2MLPipeline:
    config = PipelineConfig(task_type="regression", metric=metric, n_splits=3, n_trials=1, n_jobs=1, **kwargs)
    return H2MLPipeline(config=config, models=models)


def _ridge():
    return make_regressor(Ridge(), name="Ridge")


def _mean():
    return make_regressor(DummyRegressor(), name="Mean")


# ---------------------------------------------------------------------------
# Pooled scores and fold R2
# ---------------------------------------------------------------------------


class TestPooledScores:
    def test_match_a_direct_computation_on_all_rows(self):
        rng = np.random.default_rng(0)
        y = rng.exponential(size=60)
        p = y + rng.normal(size=60)
        folds = [_fold(y[:5], p[:5], 0), _fold(y[5:20], p[5:20], 1), _fold(y[20:], p[20:], 2)]
        row = aggregate_metrics(compute_metrics(_cv(folds))).iloc[0]
        assert row[POOLED_COL["R2"]] == pytest.approx(r2_score(y, p))
        assert row[POOLED_COL["MAE"]] == pytest.approx(mean_absolute_error(y, p))
        assert row[POOLED_COL["RMSE"]] == pytest.approx(np.sqrt(mean_squared_error(y, p)))

    def test_bookkeeping_columns_stay_out_of_mean_and_std(self):
        agg = aggregate_metrics(compute_metrics(_cv([_fold([1, 2], [1, 2]), _fold([3, 4], [3, 4], 1)])))
        assert not [c for c in agg.columns if c.startswith(("N_Test_", "Y_Mean_Test_", "Y_SS_Test_", "Is_Flat_"))]

    def test_constant_target_has_no_pooled_r2(self):
        agg = aggregate_metrics(compute_metrics(_cv([_fold([1, 1], [1, 1]), _fold([1, 1], [1, 2], 1)])))
        assert np.isnan(agg.iloc[0][POOLED_COL["R2"]])

    @pytest.mark.parametrize("prediction", [0.0, 5.0])
    def test_zero_variance_fold_r2_is_nan_whatever_the_error(self, prediction):
        df = compute_metrics(_cv([_fold([0, 0, 0], [prediction] * 3)]))
        assert np.isnan(df.iloc[0]["R2_Test"])

    def test_zero_variance_fold_is_skipped_by_the_mean_but_pooled_keeps_its_rows(self):
        y, p = np.array([1.0, 2.0, 3.0, 0.0, 0.0]), np.array([1.5, 2.5, 3.5, 0.5, 0.5])
        row = aggregate_metrics(compute_metrics(_cv([_fold(y[:3], p[:3]), _fold(y[3:], p[3:], 1)]))).iloc[0]
        assert row["R2_Test_Mean"] == pytest.approx(r2_score(y[:3], p[:3]))
        assert row[POOLED_COL["R2"]] == pytest.approx(r2_score(y, p))


class TestSelectionCol:
    @pytest.mark.parametrize("metric", ["R2", "MAE", "RMSE"])
    def test_regression(self, metric):
        assert selection_col(metric, "pooled") == f"{metric}_Test_Pooled"
        assert selection_col(metric, "fold_mean") == f"{metric}_Test_Mean"

    def test_classification_always_uses_the_fold_mean(self):
        assert selection_col("AUC", "pooled") == "AUC_Test_Mean"


# ---------------------------------------------------------------------------
# Is_Flat
# ---------------------------------------------------------------------------


def _flag(folds: list[FoldResult]) -> bool:
    return bool(aggregate_metrics(compute_metrics(_cv(folds))).iloc[0]["Is_Flat"])


class TestIsFlat:
    def test_constant_predictions(self):
        assert _flag([_fold([1, 2], [0, 0]), _fold([1, 2], [0, 0], 1)])

    def test_constant_within_each_fold_counts_even_if_folds_differ(self):
        assert _flag([_fold([1, 2], [3, 3]), _fold([1, 2], [7, 7], 1)])

    def test_tiny_spread_counts(self):
        assert _flag([_fold([1, 2], [0, 1e-9])])

    def test_one_varying_fold_clears_the_flag(self):
        assert not _flag([_fold([1, 2], [0, 0]), _fold([1, 2], [0, 1], 1)])

    def test_all_zero_training_target_with_zero_predictions(self):
        assert _flag([_fold([0, 1], [0, 0], y_train=[0, 0, 0])])

    def test_classification_uses_probabilities(self):
        fold = FoldResult(
            fold_id=0,
            model_name="M",
            y_train=np.array([0, 1]),
            y_test=np.array([0, 1, 0, 1]),
            y_pred_train=np.array([0, 1]),
            y_pred_test=np.zeros(4),
            y_prob_train=np.full(2, 0.5),
            y_prob_test=np.full(4, 0.5),
        )
        cv = CVResult(model_name="M", task_type=TaskType.CLASSIFICATION, folds=[fold])
        assert compute_metrics(cv).iloc[0]["Is_Flat"] == 1.0
        fold.y_prob_test = np.array([0.8, 0.3, 0.5, 0.6])
        assert compute_metrics(cv).iloc[0]["Is_Flat"] == 0.0

    def test_multiclass_probabilities(self):
        fold = _fold([0, 1], [0, 0])
        fold.y_prob_test = np.full((2, 3), 1 / 3)
        assert _is_flat(fold)
        fold.y_prob_test = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]])
        assert not _is_flat(fold)


def _agg(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Model", "RMSE_Test_Pooled", "RMSE_Test_Mean", "RMSE_Test_Std", "Is_Flat"])


class TestSelectBest:
    def test_pooled_ranks_by_the_score_without_a_fold_penalty(self):
        agg = _agg([("steady", 2.0, 2.0, 0.0, False), ("spread", 1.9, 1.9, 5.0, False)])
        assert select_best(agg, "RMSE_Test_Pooled", minimize=True, n_folds=5)["model_name"] == "spread"
        assert select_best(agg, "RMSE_Test_Mean", minimize=True, n_folds=5)["model_name"] == "steady"

    def test_flat_candidates_are_excluded(self):
        agg = _agg([("flat", 1.0, 1.0, 0.0, True), ("real", 2.0, 2.0, 0.0, False)])
        assert select_best(agg, "RMSE_Test_Pooled", minimize=True)["model_name"] == "real"

    def test_the_best_flat_candidate_is_kept_when_all_are_flat(self):
        agg = _agg([("flat_a", 2.0, 2.0, 0.0, True), ("flat_b", 1.0, 1.0, 0.0, True)])
        assert select_best(agg, "RMSE_Test_Pooled", minimize=True)["model_name"] == "flat_b"

    def test_rows_saved_without_the_flag_count_as_not_flat(self):
        agg = _agg([("old", 2.0, 2.0, 0.0, np.nan), ("flat", 1.0, 1.0, 0.0, True)])
        assert select_best(agg, "RMSE_Test_Pooled", minimize=True)["model_name"] == "old"


# ---------------------------------------------------------------------------
# ranking_robustness / flat_warnings
# ---------------------------------------------------------------------------


def _fold_df() -> pd.DataFrame:
    """B wins most small folds, A wins the big one: fold means and pooled scores disagree."""
    y = [np.array([0.0, 2.0]), np.array([0.0, 2.0]), np.arange(40.0)]
    a = [yi + e for yi, e in zip(y, (1.0, 1.0, 0.1), strict=True)]
    b = [yi + e for yi, e in zip(y, (0.2, 0.2, 0.9), strict=True)]
    frames = []
    for name, preds in (("A", a), ("B", b)):
        folds = [_fold(yi, p, k, name) for k, (yi, p) in enumerate(zip(y, preds, strict=True))]
        frames.append(compute_metrics(_cv(folds, name)))
    return pd.concat(frames, ignore_index=True)


class TestRanking:
    def test_winner_follows_the_scoring_rule(self):
        df = _fold_df()
        pooled = ranking_robustness(df, "RMSE", selection_scoring="pooled")
        fold_mean = ranking_robustness(df, "RMSE", selection_scoring="fold_mean")
        assert pooled.loc[pooled["Is_Winner"], "Model"].item() == "A"
        assert fold_mean.loc[fold_mean["Is_Winner"], "Model"].item() == "B"
        assert pooled.iloc[0]["Model"] == "A"  # sorted by rank under the same rule

    def test_paired_gaps_stay_fold_level(self):
        df = _fold_df()
        pooled = ranking_robustness(df, "RMSE", selection_scoring="pooled").set_index("Model")
        assert pooled.loc["B", "Winner_Gap"] < 0  # the pooled winner lags on unweighted folds
        assert pooled.loc["B", "Paired_Folds"] == 3

    def test_reports_flat_candidates(self):
        df = _fold_df()
        df.loc[df["Model"] == "B", "Is_Flat"] = 1.0
        robust = ranking_robustness(df, "RMSE", selection_scoring="pooled").set_index("Model")
        assert robust["Is_Flat"].to_dict() == {"A": False, "B": True}

    def test_fold_tables_from_older_saves_still_work(self):
        df = _fold_df().drop(columns=["N_Test", "Y_Mean_Test", "Y_SS_Test", "Is_Flat"])
        robust = ranking_robustness(df, "RMSE", selection_scoring="pooled")
        assert robust.loc[robust["Is_Winner"], "Model"].item() == "B"
        assert not robust["Is_Flat"].any()


class TestFlatWarnings:
    def _warnings(self, df: pd.DataFrame) -> list[str]:
        return flat_warnings(aggregate_metrics(df), "RMSE", "pooled", 3, df)

    def test_none_when_nothing_is_flat(self):
        assert self._warnings(_fold_df()) == []

    def test_none_without_the_flag(self):
        agg = _agg([("A", 1.0, 1.0, 0.0, False)]).drop(columns="Is_Flat")
        assert flat_warnings(agg, "RMSE", "pooled", 3, _fold_df()) == []

    def test_names_excluded_candidates(self):
        df = _fold_df()
        df.loc[df["Model"] == "B", "Is_Flat"] = 1.0
        msg = self._warnings(df)[0]
        assert "Excluded 1 constant-prediction candidate(s): B." in msg
        assert "would have won" not in msg

    def test_says_when_a_flat_candidate_would_have_won(self):
        df = _fold_df()
        df.loc[df["Model"] == "A", "Is_Flat"] = 1.0
        assert "would have won on RMSE" in self._warnings(df)[0]

    def test_all_flat(self):
        df = _fold_df()
        df["Is_Flat"] = 1.0
        assert "no usable signal" in self._warnings(df)[0]


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


class TestSelector:
    def _fit(self, keep_all: bool) -> tuple[FeatureSelector, MagicMock]:
        store = _store()
        # feat_0 and feat_1 are identical, so any real reduction drops one
        store.X[:, 1] = store.X[:, 0]
        values = pd.Series([4.0, 3.0, 2.0, 1.0], index=FEATURE_NAMES)
        selector = FeatureSelector(step=_ridge(), task_type=TaskType.REGRESSION, keep_all=keep_all)
        with patch("h2ml.features.selector.get_oof_shap_values", return_value=(None, values)) as shap:
            return selector.fit(store), shap

    def test_keep_all_skips_shap_and_keeps_every_feature(self):
        selector, shap = self._fit(keep_all=True)
        shap.assert_not_called()
        assert selector.selected_features_ == FEATURE_NAMES
        assert (selector.feature_importance_ == 0).all()

    def test_default_still_reduces(self):
        selector, shap = self._fit(keep_all=False)
        shap.assert_called_once()
        assert selector.n_selected == 3


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    @pytest.mark.parametrize(
        ("scoring", "col"), [("pooled", "RMSE_Test_Pooled"), ("fold_mean", "RMSE_Test_Mean")], ids=["pooled", "mean"]
    )
    def test_best_value_comes_from_the_selection_column(self, scoring, col):
        with _mock_selector(FEATURE_NAMES[:2]):
            result = _pipeline([_ridge()], selection_scoring=scoring).run_step1_to_step3(_store())
        assert result.selection_scoring == scoring
        assert result.best_model_value in set(pd.concat([result.step1_agg_df, result.step3_agg_df])[col])

    def test_invalid_scoring_raises(self):
        with pytest.raises(ValueError, match="selection_scoring"):
            PipelineConfig(selection_scoring="median")

    def test_flat_candidate_with_the_best_score_is_excluded(self):
        base = _store()
        noise = np.random.default_rng(1).normal(size=base.X.shape)  # features carry no signal
        store = PipelineData(X=noise, feature_names=FEATURE_NAMES, y=base.y)
        with _mock_selector(FEATURE_NAMES[:2]):
            result = _pipeline([_mean(), make_regressor(LinearRegression(), name="Linear")]).run_step1_only(store)
        assert result.best_model_name == "Linear"
        assert any("Excluded 1 constant-prediction candidate(s): Mean." in w for w in result.cv_warnings)
        assert any("would have won" in w for w in result.cv_warnings)

    def test_all_flat_candidates_warn(self):
        with _mock_selector(FEATURE_NAMES[:2]):
            result = _pipeline([_mean()]).run_step1_only(_store())
        assert result.best_model_name == "Mean"
        assert any("no usable signal" in w for w in result.cv_warnings)

    def test_all_flat_candidates_keep_every_feature_without_shap(self):
        result = _pipeline([_mean()]).run_step1_to_step3(_store())  # real selector: SHAP would raise here
        assert result.features_reduced.feature_names == FEATURE_NAMES

    def test_prefer_reduced_on_tie_skips_flat_candidates(self):
        pipeline = _pipeline([_ridge()], prefer_reduced_on_tie=True)
        result = SimpleNamespace(
            features=SimpleNamespace(n_features=4), features_reduced=SimpleNamespace(n_features=2), cv_warnings=[]
        )
        robust = pd.DataFrame(
            [("Ridge", "reduced", True, True, 0.0, 1.0)],
            columns=["Model", "Stage", "Indistinguishable", "Is_Flat", "Winner_Gap", "Gap_SE"],
        )
        best_row = pd.Series({"Model": "Ridge", "Stage": "default"})
        assert pipeline._prefer_reduced_on_tie(result, robust, pd.DataFrame(), best_row) is best_row

    @pytest.mark.parametrize(("is_flat", "stage"), [(True, "reduced"), (False, "optimized")])
    def test_step4_rejects_a_flat_tuned_model(self, is_flat, stage):
        pipeline = _pipeline([_ridge()])
        result = SimpleNamespace(
            step4_agg_df=pd.DataFrame(
                {"RMSE_Test_Pooled": [0.5], "RMSE_Test_Mean": [0.5], "RMSE_Test_Std": [0.1], "Is_Flat": [is_flat]}
            ),
            step4_cv_result=SimpleNamespace(failed_folds=[]),
            selection_scoring="pooled",
            best_model_name="Ridge",
            best_model_value=1.0,
            best_model_std=0.1,
            best_stage="reduced",
            best_params=None,
            cv_warnings=[],
        )
        pipeline._select_step4(result, {"alpha": 2.0}, {})
        assert result.best_stage == stage
        assert any("constant-prediction" in w for w in result.cv_warnings) is is_flat


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        with _mock_selector(FEATURE_NAMES[:2]):
            result = _pipeline([_ridge()], selection_scoring="fold_mean").run_step1_only(_store())
        save_result(result, tmp_path / "r")
        assert load_result(tmp_path / "r").selection_scoring == "fold_mean"

    def test_results_saved_before_the_option_stay_on_the_fold_mean(self, tmp_path):
        with _mock_selector(FEATURE_NAMES[:2]):
            result = _pipeline([_ridge()]).run_step1_only(_store())
        save_result(result, tmp_path / "r")
        path = tmp_path / "r" / "metadata.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        del meta["selection_scoring"]
        path.write_text(json.dumps(meta), encoding="utf-8")
        assert load_result(tmp_path / "r").selection_scoring == "fold_mean"


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


class _FixedFoldsRegressor(RegressorMixin, BaseEstimator):
    """Predicts y plus an error that depends on the test size."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, 0] + (1.0 if len(X) < 20 else 0.0)


def _entry() -> ModelEntry:
    return ModelEntry(model_cls=_FixedFoldsRegressor, param_fn=lambda trial: {})


class TestObjective:
    def _objective(self, score_fn, pooled_score_fn):
        y = np.arange(50.0)
        X = y[:, None]
        splits = [(np.arange(10, 50), np.arange(10)), (np.arange(0, 10), np.arange(10, 50))]
        splitter = SimpleNamespace(split=lambda X, y: iter(splits))
        objective = _build_objective(
            _entry(), X, y, "regression", 2, 0, score_fn, pooled_score_fn=pooled_score_fn, _splitter=splitter
        )
        return objective(optuna.trial.FixedTrial({})), y

    def test_pooled_scores_all_rows_once(self):
        value, y = self._objective(_score_rmse, _neg_rmse)
        pred = y + (y < 10)
        assert value == pytest.approx(-np.sqrt(mean_squared_error(y, pred)))

    def test_without_pooled_it_averages_folds(self):
        value, _ = self._objective(_score_rmse, None)
        assert value == pytest.approx(-0.5)  # folds of 10 and 40 rows count equally

    def test_pooled_r2(self):
        value, y = self._objective(_score_r2, _r2)
        assert value == pytest.approx(r2_score(y, y + (y < 10)))
