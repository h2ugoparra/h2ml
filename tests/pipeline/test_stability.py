"""
tests/pipeline/test_stability.py

Tests for seed_stability / SeedStabilityResult. Step 2 (SHAP) is mocked; LR has
opt_enabled=False, so step 4 skips Optuna.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from h2ml import SeedStabilityResult, seed_stability
from h2ml.core.feature_store import PipelineData
from h2ml.features.selector import FeatureSelector
from h2ml.pipeline.pipeline import H2MLPipeline, PipelineConfig
from h2ml.pipeline.step import make_classifier

FEATURE_NAMES = [f"feat_{i}" for i in range(6)]
SELECTED = FEATURE_NAMES[:3]
N = 120


def _store(seed: int = 0) -> PipelineData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, len(FEATURE_NAMES)))
    y = (X[:, 0] + 0.8 * rng.standard_normal(N) > 0).astype(int)
    return PipelineData(X=X, feature_names=FEATURE_NAMES, y=y)


def _mock_selector():
    sel = MagicMock(spec=FeatureSelector)
    sel.selected_features_ = SELECTED
    sel.fit_transform.side_effect = lambda store, y=None: store.select(SELECTED)
    sel.transform.side_effect = lambda store: store.select(SELECTED)
    return patch("h2ml.pipeline.pipeline.FeatureSelector", return_value=sel)


def _pipeline(**kwargs) -> H2MLPipeline:
    config = PipelineConfig(**{"task_type": "classification", "metric": "AUC", "n_splits": 3, "n_trials": 1, **kwargs})
    models = [
        make_classifier(LogisticRegression(max_iter=500), name="LR"),
        make_classifier(RandomForestClassifier(n_estimators=10, random_state=0), name="RF"),
    ]
    return H2MLPipeline(config=config, models=models)


@pytest.fixture(scope="module")
def stability() -> SeedStabilityResult:
    with _mock_selector():
        return seed_stability(_pipeline(), _store(), seeds=[0, 1, 2])


class TestSeedStability:
    def test_one_run_per_seed_with_matching_results(self, stability):
        assert stability.n_runs == 3
        assert list(stability.runs_df["Seed"]) == [0, 1, 2]
        assert stability.failed_seeds == []
        for (_, row), result in zip(stability.runs_df.iterrows(), stability.results, strict=True):
            assert row["Model"] == result.best_model_name
            assert row["AUC"] == pytest.approx(result.best_model_value)
            assert row["Stage"] == result.best_stage

    def test_each_seed_reproduces_a_plain_run(self, stability):
        with _mock_selector():
            plain = _pipeline(random_state=1).run_step1_to_step3(_store())
        assert stability.results[1].best_model_value == pytest.approx(plain.best_model_value)
        assert stability.results[1].best_model_name == plain.best_model_name

    def test_seeds_change_the_folds(self, stability):
        folds = [r.step1_cv_result[0].folds[0].test_idx for r in stability.results]
        assert not np.array_equal(folds[0], folds[1])

    def test_summary(self, stability):
        s = stability.summary().iloc[0]
        scores = stability.runs_df["AUC"]
        assert s["AUC_Mean"] == pytest.approx(scores.mean())
        assert s["AUC_Std"] == pytest.approx(scores.std())
        assert s["AUC_Min"] <= s["AUC_Mean"] <= s["AUC_Max"]
        assert s["N_runs"] == 3
        assert 0 < s["Top_Selection_Fraction"] <= 1
        assert "Holdout_AUC_Mean" not in s

    def test_selection_counts_and_feature_frequency(self, stability):
        counts = stability.selection_counts()
        assert counts["Count"].sum() == 3
        assert counts["Fraction"].sum() == pytest.approx(1.0)
        freq = stability.feature_frequency()
        assert freq.max() <= 1.0 and set(freq.index) <= set(FEATURE_NAMES)

    def test_holdout_scores_reported(self):
        with _mock_selector():
            result = seed_stability(_pipeline(holdout_frac=0.25), _store(), seeds=[0, 1], through_step=4)
        assert "Holdout_AUC" in result.runs_df
        for (_, row), run in zip(result.runs_df.iterrows(), result.results, strict=True):
            assert row["Holdout_AUC"] == pytest.approx(run.holdout_df.iloc[0]["AUC_Test_Mean"])
        assert "Holdout_AUC_Std" in result.summary()

    def test_step1_only(self):
        result = seed_stability(_pipeline(), _store(), seeds=[0, 1], through_step=1)
        assert set(result.runs_df["Stage"]) == {"default"}
        assert set(result.runs_df["N_features"]) == {len(FEATURE_NAMES)}

    def test_failed_seed_is_skipped(self):
        pipeline = _pipeline(n_jobs=1)
        real = H2MLPipeline.run_step1_only

        def flaky(self, store, transforms=None):
            if self.config.random_state == 1:
                raise ValueError("boom")
            return real(self, store, transforms=transforms)

        with patch.object(H2MLPipeline, "run_step1_only", flaky):
            result = seed_stability(pipeline, _store(), seeds=[0, 1, 2], through_step=1)
        assert result.failed_seeds == [1]
        assert list(result.runs_df["Seed"]) == [0, 2]

    @pytest.mark.parametrize("seeds", [[3], [3, 3]])
    def test_needs_two_distinct_seeds(self, seeds):
        with pytest.raises(ValueError, match="two distinct"):
            seed_stability(_pipeline(), _store(), seeds=seeds)

    def test_invalid_through_step_raises(self):
        with pytest.raises(ValueError, match="through_step"):
            seed_stability(_pipeline(), _store(), seeds=[0, 1], through_step=2)
