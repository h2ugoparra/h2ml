"""Tests for h2ml.evaluation.ranking — paired-fold and cross-metric selection robustness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from h2ml.evaluation.metrics import METRIC_COL, aggregate_metrics, select_best
from h2ml.evaluation.ranking import ranking_robustness, ranking_warnings


def _fold_df(scores: dict[str, dict[str, list[float]]], stage: str = "default") -> pd.DataFrame:
    """scores: {model: {metric: per-fold values}} → long fold DataFrame."""
    rows = []
    for model, by_metric in scores.items():
        n = len(next(iter(by_metric.values())))
        for fold in range(n):
            row = {"Model": model, "Fold": fold, "Stage": stage}
            row.update({f"{m}_Test": v[fold] for m, v in by_metric.items()})
            rows.append(row)
    return pd.DataFrame(rows)


# Fold difficulty varies a lot (0.6 → 0.9); A beats B by exactly 0.01 on every fold.
HARD_EASY = [0.60, 0.70, 0.80, 0.90, 0.65]


class TestRankingRobustness:
    def test_winner_matches_select_best(self):
        df = _fold_df({"A": {"AUC": HARD_EASY}, "B": {"AUC": [s - 0.01 for s in HARD_EASY]}})
        out = ranking_robustness(df, "AUC")
        best = select_best(aggregate_metrics(df), METRIC_COL["AUC"], n_folds=5, fold_df=df)
        assert out.loc[0, "Model"] == best["model_name"] == "A"
        assert out.loc[0, "Is_Winner"] and out["Is_Winner"].sum() == 1

    def test_ranks_use_fold_centered_spread(self):
        """A hedger with uniform but worse fold scores must not outrank the model with the
        better mean whose one bad fold inflates its raw std — the order prefer_reduced_on_tie
        picks tied candidates from."""
        df = _fold_df(
            {
                "Tracker": {"Brier": [0.05, 0.40, 0.05, 0.10, 0.02]},
                "Hedger": {"Brier": [0.15, 0.25, 0.13, 0.14, 0.12]},
            }
        )
        out = ranking_robustness(df, "Brier")
        assert list(out["Model"]) == ["Tracker", "Hedger"]
        assert out.loc[0, "Is_Winner"]

    def test_pairing_cancels_fold_difficulty(self):
        """Marginal std (~0.12) dwarfs the 0.01 gap, but the paired gap is consistent."""
        df = _fold_df({"A": {"AUC": HARD_EASY}, "B": {"AUC": [s - 0.01 for s in HARD_EASY]}})
        b = ranking_robustness(df, "AUC").set_index("Model").loc["B"]
        assert b["Winner_Gap"] == pytest.approx(0.01)
        assert b["Folds_Won"] == 5 and b["Paired_Folds"] == 5
        assert not b["Indistinguishable"]

    def test_noisy_pair_is_indistinguishable(self):
        df = _fold_df(
            {
                "A": {"AUC": [0.80, 0.70, 0.82, 0.71, 0.80]},
                "B": {"AUC": [0.70, 0.80, 0.71, 0.82, 0.79]},
            }
        )
        out = ranking_robustness(df, "AUC").set_index("Model")
        loser = out.loc[~out["Is_Winner"]].iloc[0]
        assert loser["Indistinguishable"]

    def test_corrected_se(self):
        """SE = sqrt(var(d) * (1/k + 1/(k-1))) — Nadeau–Bengio for k-fold CV."""
        d = np.array([0.02, 0.01, 0.03, 0.00, 0.04])
        df = _fold_df({"A": {"AUC": HARD_EASY}, "B": {"AUC": list(np.array(HARD_EASY) - d)}})
        b = ranking_robustness(df, "AUC").set_index("Model").loc["B"]
        assert b["Gap_SE"] == pytest.approx(np.sqrt(d.var(ddof=1) * (1 / 5 + 1 / 4)))

    def test_minimize_metric_gap_positive_when_winner_better(self):
        df = _fold_df({"A": {"RMSE": [1.0, 2.0, 3.0, 4.0, 5.0]}, "B": {"RMSE": [1.5, 2.5, 3.5, 4.5, 5.5]}})
        b = ranking_robustness(df, "RMSE").set_index("Model").loc["B"]
        assert b["Winner_Gap"] == pytest.approx(0.5)
        assert b["Folds_Won"] == 5

    def test_nan_fold_dropped_pairwise(self):
        """AUC is NaN on single-class folds — that fold leaves only the affected pair."""
        b_scores = [s - 0.01 for s in HARD_EASY]
        b_scores[2] = np.nan
        df = _fold_df({"A": {"AUC": HARD_EASY}, "B": {"AUC": b_scores}})
        b = ranking_robustness(df, "AUC").set_index("Model").loc["B"]
        assert b["Paired_Folds"] == 4

    def test_ranks_per_metric(self):
        df = _fold_df(
            {
                "A": {"AUC": [0.90] * 5, "Brier": [0.20] * 5},
                "B": {"AUC": [0.80] * 5, "Brier": [0.10] * 5},
            }
        )
        out = ranking_robustness(df, "AUC").set_index("Model")
        assert out.loc["A", "Rank_AUC"] == 1 and out.loc["A", "Rank_Brier"] == 2
        assert out.loc["B", "Rank_Brier"] == 1

    def test_stage_is_part_of_candidate_key(self):
        df = pd.concat(
            [
                _fold_df({"A": {"AUC": HARD_EASY}}, stage="default"),
                _fold_df({"A": {"AUC": [s + 0.01 for s in HARD_EASY]}}, stage="reduced"),
            ],
            ignore_index=True,
        )
        out = ranking_robustness(df, "AUC")
        assert len(out) == 2
        assert out.loc[0, "Stage"] == "reduced"


class TestRankingWarnings:
    def test_single_candidate_no_warnings(self):
        df = _fold_df({"A": {"AUC": HARD_EASY}})
        assert ranking_warnings(ranking_robustness(df, "AUC"), "AUC") == []

    def test_clear_winner_no_warnings(self):
        df = _fold_df(
            {
                "A": {"AUC": HARD_EASY, "Brier": [0.10] * 5},
                "B": {"AUC": [s - 0.05 for s in HARD_EASY], "Brier": [0.20] * 5},
            }
        )
        assert ranking_warnings(ranking_robustness(df, "AUC"), "AUC") == []

    def test_tie_warning_names_candidates(self):
        df = _fold_df({"A": {"AUC": HARD_EASY}, "B": {"AUC": HARD_EASY}})
        msgs = ranking_warnings(ranking_robustness(df, "AUC"), "AUC")
        assert len(msgs) == 1
        assert "within one corrected SE" in msgs[0] and "B" in msgs[0]

    def test_counterpart_warning(self):
        """Best ranker but worse calibrated → ranking-vs-calibration trade-off."""
        df = _fold_df(
            {
                "A": {"AUC": HARD_EASY, "Brier": [0.20] * 5},
                "B": {"AUC": [s - 0.05 for s in HARD_EASY], "Brier": [0.10] * 5},
            }
        )
        msgs = ranking_warnings(ranking_robustness(df, "AUC"), "AUC")
        assert len(msgs) == 1
        assert "ranks 2/2 on Brier" in msgs[0] and "ranking vs calibration" in msgs[0]

    def test_stage_in_label_only_when_it_varies(self):
        df = _fold_df({"A": {"AUC": HARD_EASY}, "B": {"AUC": HARD_EASY}})
        msgs = ranking_warnings(ranking_robustness(df, "AUC"), "AUC")
        assert "default" not in msgs[0]
