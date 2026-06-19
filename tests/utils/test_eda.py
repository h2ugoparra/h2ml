"""Tests for the EDA helpers in h2ml/utils/eda.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from h2ml.utils.eda import (
    check_pipeline_readiness,
    correlated_feature_pairs,
    flag_outliers,
    profile_dataset,
    target_correlations,
)


@pytest.fixture
def df() -> pd.DataFrame:
    """A small mixed-type frame with known structure."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y_reg": 3.0 * x + rng.normal(0, 0.1, n),  # strongly linear with x
            "x": x,
            "x_copy": 2.0 * x,  # perfectly correlated with x
            "noise": rng.normal(size=n),
            "label": rng.choice(["a", "b", "c"], n),  # non-numeric
        }
    )


# ---------------------------------------------------------------------------
# profile_dataset
# ---------------------------------------------------------------------------


class TestProfileDataset:
    def test_prints_profile(self, df, capsys):
        profile_dataset(df)
        out = capsys.readouterr().out
        assert "DATASET PROFILE" in out
        assert "Rows: 200" in out

    def test_empty_dataframe_does_not_raise(self, capsys):
        profile_dataset(pd.DataFrame({"a": [], "b": []}))
        assert "Rows: 0" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# flag_outliers
# ---------------------------------------------------------------------------


class TestFlagOutliers:
    def test_detects_known_outlier(self):
        data = pd.DataFrame({"v": [1, 2, 3, 4, 5, 1000]})
        res = flag_outliers(data)
        assert res.loc[res["column"] == "v", "outlier_count"].iloc[0] == 1

    def test_higher_multiplier_flags_fewer(self):
        data = pd.DataFrame({"v": list(range(100)) + [500]})
        strict = flag_outliers(data, multiplier=3.0)["outlier_count"].iloc[0]
        mild = flag_outliers(data, multiplier=1.5)["outlier_count"].iloc[0]
        assert strict <= mild

    def test_empty_dataframe_returns_empty_with_columns(self):
        res = flag_outliers(pd.DataFrame({"v": []}))
        assert res.empty
        assert list(res.columns) == ["column", "lower_bound", "upper_bound", "outlier_count", "outlier_%"]

    def test_ignores_non_numeric(self, df):
        res = flag_outliers(df)
        assert "label" not in set(res["column"])


# ---------------------------------------------------------------------------
# target_correlations
# ---------------------------------------------------------------------------


class TestTargetCorrelations:
    def test_ranks_strongest_first(self, df):
        res = target_correlations(df, "y_reg")
        # x and x_copy are the strongest; noise should be last.
        assert set(res["feature"].iloc[:2]) == {"x", "x_copy"}
        assert res["feature"].iloc[-1] == "noise"

    def test_excludes_target_and_non_numeric(self, df):
        res = target_correlations(df, "y_reg")
        assert "y_reg" not in set(res["feature"])
        assert "label" not in set(res["feature"])

    def test_n_features_limits_rows(self, df):
        assert len(target_correlations(df, "y_reg", n_features=2)) == 2

    def test_sort_by_changes_order_key(self, df):
        res = target_correlations(df, "y_reg", sort_by="kendall")
        ranked = res["kendall"].abs().tolist()
        assert ranked == sorted(ranked, reverse=True)

    def test_invalid_sort_by_raises(self, df):
        with pytest.raises(ValueError, match="sort_by"):
            target_correlations(df, "y_reg", sort_by="cosine")

    def test_missing_target_raises(self, df):
        with pytest.raises(ValueError, match="not found"):
            target_correlations(df, "nope")

    def test_non_numeric_target_raises(self, df):
        with pytest.raises(ValueError, match="numeric"):
            target_correlations(df, "label")


# ---------------------------------------------------------------------------
# check_pipeline_readiness
# ---------------------------------------------------------------------------


class TestCheckPipelineReadiness:
    def test_clean_frame_is_ready(self):
        rng = np.random.default_rng(1)
        clean = pd.DataFrame({"t": rng.normal(size=50), "a": rng.normal(size=50), "b": rng.normal(size=50)})
        assert check_pipeline_readiness(clean, target="t", task="regression").empty

    def test_flags_constant_column(self):
        data = pd.DataFrame({"a": np.arange(50.0), "const": np.ones(50)})
        checks = set(check_pipeline_readiness(data)["check"])
        assert "constant_features" in checks

    def test_flags_nonfinite_features(self):
        data = pd.DataFrame({"a": [1.0, 2.0, np.nan, np.inf] + [0.0] * 46})
        checks = set(check_pipeline_readiness(data)["check"])
        assert "nonfinite_features" in checks

    def test_flags_insufficient_rows(self):
        data = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        report = check_pipeline_readiness(data, n_splits=5)
        assert "insufficient_rows" in set(report["check"])

    def test_flags_duplicate_columns(self):
        data = pd.DataFrame(np.ones((50, 2)), columns=["a", "a"])
        assert "duplicate_columns" in set(check_pipeline_readiness(data)["check"])

    def test_flags_single_class_target(self):
        data = pd.DataFrame({"t": np.ones(50), "a": np.arange(50.0)})
        report = check_pipeline_readiness(data, target="t", task="classification")
        assert "single_class_target" in set(report["check"])

    def test_flags_class_imbalance_as_warning(self):
        y = np.array([0] * 48 + [1] * 2)
        data = pd.DataFrame({"t": y, "a": np.arange(50.0)})
        report = check_pipeline_readiness(data, target="t", task="classification")
        row = report[report["check"] == "class_imbalance"]
        assert not row.empty
        assert row["severity"].iloc[0] == "warning"

    def test_regression_skips_class_checks(self):
        data = pd.DataFrame({"t": np.ones(50), "a": np.arange(50.0)})
        # constant target is fine for regression — only the constant *feature* logic
        # applies, and 't' is the target so it is excluded from feature checks.
        report = check_pipeline_readiness(data, target="t", task="regression")
        assert "single_class_target" not in set(report["check"])

    def test_missing_target_flagged(self):
        data = pd.DataFrame({"a": np.arange(50.0)})
        assert "missing_target" in set(check_pipeline_readiness(data, target="t")["check"])


# ---------------------------------------------------------------------------
# correlated_feature_pairs
# ---------------------------------------------------------------------------


class TestCorrelatedFeaturePairs:
    def test_detects_correlated_pair(self, df):
        res = correlated_feature_pairs(df, threshold=0.9, target="y_reg")
        pair = {res["feature_1"].iloc[0], res["feature_2"].iloc[0]}
        assert pair == {"x", "x_copy"}
        assert res["corr"].iloc[0] == pytest.approx(1.0)

    def test_excludes_target(self, df):
        res = correlated_feature_pairs(df, threshold=0.5, target="y_reg")
        appearing = set(res["feature_1"]) | set(res["feature_2"])
        assert "y_reg" not in appearing

    def test_threshold_filters(self, df):
        # x and x_copy correlate at 1.0; a 0.99 threshold keeps them, 1.01 is impossible
        assert len(correlated_feature_pairs(df, threshold=0.99, target="y_reg")) >= 1

    def test_sorted_by_abs_descending(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=200)
        data = pd.DataFrame({"a": a, "b": 0.95 * a + rng.normal(0, 0.1, 200), "c": -a})
        res = correlated_feature_pairs(data, threshold=0.0)
        assert res["corr"].abs().tolist() == sorted(res["corr"].abs().tolist(), reverse=True)

    def test_fewer_than_two_numeric_returns_empty(self):
        data = pd.DataFrame({"a": np.arange(10.0), "label": list("abcdefghij")})
        res = correlated_feature_pairs(data)
        assert res.empty
        assert list(res.columns) == ["feature_1", "feature_2", "corr"]

    def test_invalid_method_raises(self, df):
        with pytest.raises(ValueError, match="method"):
            correlated_feature_pairs(df, method="cosine")  # type: ignore[arg-type]

    def test_invalid_threshold_raises(self, df):
        with pytest.raises(ValueError, match="threshold"):
            correlated_feature_pairs(df, threshold=1.5)
