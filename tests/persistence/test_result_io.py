"""
tests/persistence/test_result_io.py

Tests for save_result() / load_result() in persistence/result_io.py.

Covers:
    - Scalar metadata round-trip
    - DataFrame round-trip (step1/3/4 fold and agg)
    - PipelineData (features, features_reduced) round-trip
    - Partial results (missing optional fields load without error)
    - Overwrite behaviour (save_result replaces existing directory)
    - FileNotFoundError on load from missing path
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from h2ml.features.feature_store import PipelineData
from h2ml.persistence.result_io import save_result, load_result
from h2ml.pipeline.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(n: int = 50, f: int = 4, seed: int = 0) -> PipelineData:
    rng = np.random.default_rng(seed)
    return PipelineData(
        X=rng.standard_normal((n, f)).astype(np.float32),
        feature_names=[f"feat_{i}" for i in range(f)],
        y=rng.integers(0, 2, n).astype(np.float32),
    )


def _make_fold_df(stage: str = "default") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": ["LR", "LR", "LR"],
            "Stage": [stage] * 3,
            "AUC_Test_Mean": [0.8, 0.82, 0.79],
        }
    )


def _make_agg_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": ["LR"],
            "AUC_Test_Mean": [0.80],
            "AUC_Test_Std": [0.01],
        }
    )


def _full_result() -> PipelineResult:
    return PipelineResult(
        features=_make_store(),
        features_reduced=_make_store(f=2),
        best_model_name="LR",
        best_model_value=0.82,
        best_stage="reduced",
        best_feature_stage="reduced",
        best_params={"C": 1.0},
        y_transform=None,
        step1_fold_df=_make_fold_df("default"),
        step1_agg_df=_make_agg_df(),
        step3_fold_df=_make_fold_df("reduced"),
        step3_agg_df=_make_agg_df(),
        step4_fold_df=_make_fold_df("optimized"),
        step4_agg_df=_make_agg_df(),
    )


# ---------------------------------------------------------------------------
# Scalar metadata
# ---------------------------------------------------------------------------


class TestScalarRoundTrip:
    def test_best_model_name_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.best_model_name == result.best_model_name

    def test_best_model_value_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.best_model_value == pytest.approx(result.best_model_value)

    def test_best_model_std_preserved(self, tmp_path):
        result = _full_result()
        result.best_model_std = 0.034
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.best_model_std == pytest.approx(0.034)

    def test_best_model_std_none_preserved(self, tmp_path):
        result = _full_result()
        result.best_model_std = None
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.best_model_std is None

    def test_best_stage_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.best_stage == result.best_stage

    def test_best_feature_stage_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.best_feature_stage == result.best_feature_stage

    def test_best_params_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.best_params == result.best_params

    def test_y_transform_none_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.y_transform is None

    def test_y_transform_string_preserved(self, tmp_path):
        result = _full_result()
        result.y_transform = "log"
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.y_transform == "log"

    def test_cv_type_spatial_preserved(self, tmp_path):
        result = _full_result()
        result.cv_type = "spatial"
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.cv_type == "spatial"

    def test_cv_type_random_preserved(self, tmp_path):
        result = _full_result()
        result.cv_type = "random"
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.cv_type == "random"

    def test_cv_type_defaults_to_random_when_absent(self, tmp_path):
        """Results saved before cv_type was persisted should load as 'random'."""
        result = _full_result()
        save_result(result, tmp_path / "run")
        # Simulate old save by removing cv_type from the JSON
        import json

        meta_path = tmp_path / "run" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta.pop("cv_type", None)
        meta_path.write_text(json.dumps(meta))
        loaded = load_result(tmp_path / "run")
        assert loaded.cv_type == "random"

    def test_cv_warnings_list_preserved(self, tmp_path):
        result = _full_result()
        result.cv_warnings = ["RF: 2/5 fold(s) failed (fold IDs: [0, 1])"]
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.cv_warnings == ["RF: 2/5 fold(s) failed (fold IDs: [0, 1])"]

    def test_cv_warnings_empty_list_preserved(self, tmp_path):
        result = _full_result()
        result.cv_warnings = []
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.cv_warnings == []

    def test_metric_preserved(self, tmp_path):
        result = _full_result()
        result.metric = "AUC"
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.metric == "AUC"

    def test_metric_none_when_absent_in_old_save(self, tmp_path):
        """Results saved before metric was persisted should load with metric=None."""
        import json

        result = _full_result()
        save_result(result, tmp_path / "run")
        p = tmp_path / "run" / "metadata.json"
        meta = json.loads(p.read_text())
        meta.pop("metric", None)
        p.write_text(json.dumps(meta))
        assert load_result(tmp_path / "run").metric is None


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------


class TestDataFrameRoundTrip:
    @pytest.mark.parametrize(
        "attr",
        [
            "step1_fold_df",
            "step1_agg_df",
            "step3_fold_df",
            "step3_agg_df",
            "step4_fold_df",
            "step4_agg_df",
        ],
    )
    def test_dataframe_not_none(self, tmp_path, attr):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert getattr(loaded, attr) is not None

    def test_step1_fold_df_shape_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.step1_fold_df.shape == result.step1_fold_df.shape

    def test_step1_agg_df_values_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        pd.testing.assert_frame_equal(loaded.step1_agg_df, result.step1_agg_df)


# ---------------------------------------------------------------------------
# PipelineData
# ---------------------------------------------------------------------------


class TestPipelineDataRoundTrip:
    def test_features_X_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        np.testing.assert_array_almost_equal(loaded.features.X, result.features.X)

    def test_features_y_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        np.testing.assert_array_equal(loaded.features.y, result.features.y)

    def test_features_feature_names_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.features.feature_names == result.features.feature_names

    def test_features_reduced_n_features_preserved(self, tmp_path):
        result = _full_result()
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.features_reduced.n_features == result.features_reduced.n_features

    def test_y_true_preserved_when_present(self, tmp_path):
        result = _full_result()
        rng = np.random.default_rng(99)
        result.features.y_true = rng.standard_normal(50).astype(np.float32)
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        np.testing.assert_array_almost_equal(loaded.features.y_true, result.features.y_true)

    def test_y_true_none_when_absent(self, tmp_path):
        result = _full_result()
        assert result.features.y_true is None
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.features.y_true is None


# ---------------------------------------------------------------------------
# Partial results (optional fields absent)
# ---------------------------------------------------------------------------


class TestPartialResult:
    def test_partial_result_saves_without_error(self, tmp_path):
        """A result with only step 1 completed should save cleanly."""
        result = PipelineResult(
            features=_make_store(),
            best_model_name="LR",
            best_model_value=0.8,
            best_stage="default",
            step1_fold_df=_make_fold_df(),
            step1_agg_df=_make_agg_df(),
        )
        save_result(result, tmp_path / "partial")

    def test_partial_result_loads_without_error(self, tmp_path):
        result = PipelineResult(
            features=_make_store(),
            best_model_name="LR",
            step1_fold_df=_make_fold_df(),
            step1_agg_df=_make_agg_df(),
        )
        save_result(result, tmp_path / "partial")
        loaded = load_result(tmp_path / "partial")
        assert loaded.step3_fold_df is None
        assert loaded.step4_fold_df is None
        assert loaded.features_reduced is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_load_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_result(tmp_path / "does_not_exist")

    def test_save_overwrites_existing_directory(self, tmp_path):
        result = _full_result()
        dest = tmp_path / "run"
        save_result(result, dest)

        # Save again with different best_model_name
        result.best_model_name = "RF"
        save_result(result, dest)

        loaded = load_result(dest)
        assert loaded.best_model_name == "RF"

    def test_numpy_scalar_params_serialise(self, tmp_path):
        """Optuna may produce numpy int/float params — these must survive JSON round-trip."""
        result = _full_result()
        result.best_params = {
            "n_estimators": np.int64(100),
            "learning_rate": np.float32(0.1),
        }
        save_result(result, tmp_path / "run")
        loaded = load_result(tmp_path / "run")
        assert loaded.best_params["n_estimators"] == 100
        assert abs(loaded.best_params["learning_rate"] - 0.1) < 1e-5
