"""
h2ml/evaluation/ranking.py

Robustness of a CV model selection, computed from per-fold metrics after select_best():
    - Paired fold comparison: every candidate is scored on the same folds, so the
      per-fold difference to the winner cancels fold difficulty (large under spatial
      CV) that the reported ± std includes. Its standard error uses the Nadeau–Bengio
      correction for overlapping training sets.
    - Cross-metric ranks: where each candidate ranks under every other metric.

Advisory only — nothing here changes which model is selected.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from h2ml.evaluation.metrics import METRIC_COL, METRIC_MINIMIZE, aggregate_metrics, select_best, selection_score

# Columns identifying one selection candidate in a fold DataFrame.
KEY_COLS = ("Model", "Y_transform", "Stage")

# Metric from a different family than the selection metric, and what the pair contrasts.
# Disagreement within a family (AUC vs AUCPR, Brier vs LogLoss) is mostly noise, and R2
# and RMSE rank identically on shared folds, so only one counterpart per metric is checked.
COUNTERPART_METRIC: dict[str, tuple[str, str]] = {
    "AUC": ("Brier", "ranking vs calibration"),
    "AUCPR": ("Brier", "ranking vs calibration"),
    "LogLoss": ("AUC", "calibration vs ranking"),
    "Brier": ("AUC", "calibration vs ranking"),
    "F1": ("AUC", "thresholded decisions vs ranking"),
    "R2": ("MAE", "conditional mean vs median"),
    "RMSE": ("MAE", "conditional mean vs median"),
    "MAE": ("RMSE", "conditional median vs mean"),
}


def ranking_robustness(fold_df: pd.DataFrame, metric: str, n_folds: Optional[int] = None) -> pd.DataFrame:
    """
    Paired-fold and cross-metric robustness of the selection made on fold_df.

    The winner is re-derived with select_best() exactly as the pipeline selects it
    (confidence-bound score). Each other candidate is compared with it on the folds
    both completed (a NaN score, e.g. AUC on a single-class fold, drops that fold
    from that pair only).

    Args:
        fold_df: Per-fold metrics (step1_fold_df, or step 1 + step 3 concatenated).
        metric:  Short selection metric name, e.g. "AUC" or "RMSE".
        n_folds: CV folds behind fold_df; inferred from the "Fold" column when None.

    Returns:
        One row per candidate (key columns Model/Y_transform/Stage as present), sorted
        by rank under metric, with columns:
            <metric>_Test_Mean — mean held-out score
            Winner_Gap         — mean paired difference, oriented so positive = winner better
            Gap_SE             — Nadeau–Bengio corrected SE of Winner_Gap
            Folds_Won          — folds where the winner scored strictly better
            Paired_Folds       — folds both completed
            Indistinguishable  — Winner_Gap ≤ Gap_SE (a heuristic tie, not a formal test)
            Is_Winner
            Rank_<m>           — rank (1 = best) under each metric present, by the same
                                 confidence-bound score select_best uses
    """
    n_folds = n_folds or int(fold_df["Fold"].nunique())
    keys = [c for c in KEY_COLS if c in fold_df.columns]
    agg = aggregate_metrics(fold_df)
    best = select_best(agg, METRIC_COL[metric], METRIC_MINIMIZE[metric], n_folds)
    winner = tuple(best["row"][k] for k in keys)

    test_col = f"{metric}_Test"
    sign = -1.0 if METRIC_MINIMIZE[metric] else 1.0
    scores = {key: g.set_index("Fold")[test_col] for key, g in fold_df.groupby(keys, dropna=False)}
    winner_scores = scores[winner]

    rows = []
    for key, s in scores.items():
        is_winner = key == winner
        d = (sign * (winner_scores - s)).dropna()  # aligned on Fold
        k = len(d)
        gap = float(d.mean()) if k else np.nan
        se = float(np.sqrt(d.var(ddof=1) * (1 / k + 1 / (k - 1)))) if k >= 2 and not is_winner else np.nan
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "Winner_Gap": gap,
                "Gap_SE": se,
                "Folds_Won": int((d > 0).sum()),
                "Paired_Folds": k,
                "Indistinguishable": bool(gap <= se) if not np.isnan(se) else False,
                "Is_Winner": is_winner,
            }
        )
    paired = pd.DataFrame(rows)

    rank_cols = []
    for m, col in METRIC_COL.items():
        if col in agg.columns:
            score = selection_score(agg, col, METRIC_MINIMIZE[m], n_folds)
            agg[f"Rank_{m}"] = score.rank(ascending=METRIC_MINIMIZE[m], method="min")
            rank_cols.append(f"Rank_{m}")

    out = agg[keys + [METRIC_COL[metric]] + rank_cols].merge(paired, on=keys)
    out = out[keys + [METRIC_COL[metric]] + list(paired.columns[len(keys) :]) + rank_cols]
    return out.sort_values(f"Rank_{metric}").reset_index(drop=True)


def ranking_warnings(robust_df: pd.DataFrame, metric: str) -> list[str]:
    """
    Turn a ranking_robustness() table into at most two warnings:
        - candidates indistinguishable from the winner on paired folds
        - the winner ranking below 1st on the counterpart-family metric
    """
    if len(robust_df) < 2:
        return []
    # Label by the key columns that actually vary (Stage is constant within step 1).
    keys = [c for c in KEY_COLS if c in robust_df.columns and (c == "Model" or robust_df[c].nunique() > 1)]

    def label(row: pd.Series) -> str:
        return " | ".join(str(row[k]) for k in keys if pd.notna(row[k]))

    winner = robust_df.loc[robust_df["Is_Winner"]].iloc[0]
    msgs = []

    ties = robust_df.loc[robust_df["Indistinguishable"]]
    if not ties.empty:
        names = ", ".join(label(r) for _, r in ties.iterrows())
        msgs.append(
            f"{metric}: {len(ties)} candidate(s) within one corrected SE of the winner {label(winner)} "
            f"on paired folds ({names}). The CV data does not separate them."
        )

    cp, contrast = COUNTERPART_METRIC[metric]
    rank_col = f"Rank_{cp}"
    if rank_col in robust_df.columns and pd.notna(winner[rank_col]) and winner[rank_col] > 1:
        top = robust_df.loc[robust_df[rank_col] == 1].iloc[0]
        msgs.append(
            f"{metric} winner {label(winner)} ranks {int(winner[rank_col])}/{len(robust_df)} on {cp} "
            f"(best on {cp}: {label(top)}). The two metrics contrast {contrast} — "
            "a trade-off to weigh, not a reason to switch metric."
        )
    return msgs
