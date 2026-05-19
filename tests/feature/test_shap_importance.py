"""
tests/feature/test_shap_importance.py

Tests for shap_importance.py:
    - _extract_shap_array: normalises 3-D and 2-D SHAP output
    - _select_explainer: routes to TreeExplainer vs generic Explainer
    - get_shap_values: end-to-end with a small tree model
    - get_oof_shap_values: out-of-fold SHAP with a small tree model
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR

from h2ml.pipeline.base import TaskType
from h2ml.pipeline.step import make_classifier, make_regressor
from h2ml.features.feature_store import PipelineData
from h2ml.features.shap_importance import (
    _extract_shap_array,
    _select_explainer,
    get_shap_values,
    get_oof_shap_values,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["f0", "f1", "f2", "f3", "f4"]
N = 60


@pytest.fixture
def clf_store() -> PipelineData:
    rng = np.random.default_rng(42)
    return PipelineData(
        X=rng.standard_normal((N, len(FEATURE_NAMES))),
        feature_names=FEATURE_NAMES,
        y=rng.integers(0, 2, N),
    )


@pytest.fixture
def reg_store() -> PipelineData:
    rng = np.random.default_rng(42)
    return PipelineData(
        X=rng.standard_normal((N, len(FEATURE_NAMES))),
        feature_names=FEATURE_NAMES,
        y=rng.standard_normal(N),
    )


@pytest.fixture
def multiclass_store() -> PipelineData:
    rng = np.random.default_rng(7)
    return PipelineData(
        X=rng.standard_normal((N, len(FEATURE_NAMES))),
        feature_names=FEATURE_NAMES,
        y=rng.integers(0, 3, N),  # 3 classes: 0, 1, 2
    )


@pytest.fixture
def fitted_rf_clf(clf_store) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(clf_store.X, clf_store.y)
    return clf


@pytest.fixture
def fitted_rf_multiclass(multiclass_store) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(multiclass_store.X, multiclass_store.y)
    return clf


@pytest.fixture
def fitted_rf_reg(reg_store) -> RandomForestRegressor:
    reg = RandomForestRegressor(n_estimators=10, random_state=42)
    reg.fit(reg_store.X, reg_store.y)
    return reg


# ---------------------------------------------------------------------------
# _extract_shap_array
# ---------------------------------------------------------------------------


class TestExtractShapArray:
    def test_2d_array_returned_as_is(self):
        arr = np.ones((10, 5))
        result = _extract_shap_array(arr)
        assert result.shape == (10, 5)

    # -- explicit class_index --------------------------------------------------

    def test_3d_array_extracts_class_index_1(self):
        arr = np.zeros((10, 5, 3))
        arr[..., 1] = 2.0
        result = _extract_shap_array(arr, class_index=1)
        assert result.shape == (10, 5)
        assert (result == 2.0).all()

    def test_3d_array_extracts_class_index_0(self):
        arr = np.zeros((10, 5, 2))
        arr[..., 0] = 5.0
        result = _extract_shap_array(arr, class_index=0)
        assert (result == 5.0).all()

    def test_class_index_out_of_range_raises(self):
        arr = np.zeros((10, 5, 2))
        with pytest.raises(ValueError, match="class_index"):
            _extract_shap_array(arr, class_index=5)

    def test_negative_class_index_raises(self):
        arr = np.zeros((10, 5, 3))
        with pytest.raises(ValueError, match="class_index"):
            _extract_shap_array(arr, class_index=-1)

    # -- auto (class_index=None) -----------------------------------------------

    def test_auto_binary_extracts_class_1(self):
        arr = np.zeros((10, 5, 2))
        arr[..., 1] = 3.0
        result = _extract_shap_array(arr, class_index=None)
        assert result.shape == (10, 5)
        assert (result == 3.0).all()

    def test_auto_multiclass_returns_mean_abs(self):
        # 3-class: class 0 = 1.0, class 1 = -2.0, class 2 = 3.0
        # mean(|1|, |-2|, |3|) = mean(1, 2, 3) = 2.0
        arr = np.zeros((10, 5, 3))
        arr[..., 0] = 1.0
        arr[..., 1] = -2.0
        arr[..., 2] = 3.0
        result = _extract_shap_array(arr, class_index=None)
        assert result.shape == (10, 5)
        np.testing.assert_allclose(result, 2.0)

    def test_auto_multiclass_shape_correct(self):
        arr = np.random.default_rng(0).standard_normal((20, 8, 4))
        result = _extract_shap_array(arr, class_index=None)
        assert result.shape == (20, 8)

    def test_auto_multiclass_values_nonnegative(self):
        arr = np.random.default_rng(1).standard_normal((15, 6, 3))
        result = _extract_shap_array(arr, class_index=None)
        assert (result >= 0).all()

    def test_object_with_values_attribute(self):
        """SHAP Explanation objects expose `.values`."""

        class FakeExplanation:
            values = np.ones((10, 5))

        result = _extract_shap_array(FakeExplanation())
        assert result.shape == (10, 5)


# ---------------------------------------------------------------------------
# _select_explainer
# ---------------------------------------------------------------------------


class TestSelectExplainer:
    def test_rf_uses_tree_explainer(self, fitted_rf_clf, clf_store):
        import shap

        explainer = _select_explainer(fitted_rf_clf, TaskType.CLASSIFICATION, clf_store.to_frame())
        assert isinstance(explainer, shap.TreeExplainer)

    def test_rf_regressor_uses_tree_explainer(self, fitted_rf_reg, reg_store):
        import shap

        explainer = _select_explainer(fitted_rf_reg, TaskType.REGRESSION, reg_store.to_frame())
        assert isinstance(explainer, shap.TreeExplainer)

    def test_logistic_regression_uses_generic_explainer(self, clf_store):
        import shap

        lr = LogisticRegression(max_iter=200, random_state=42)
        lr.fit(clf_store.X, clf_store.y)
        explainer = _select_explainer(lr, TaskType.CLASSIFICATION, clf_store.to_frame())
        assert isinstance(explainer, shap.Explainer)

    def test_svc_linear_no_probability_uses_linear_explainer(self, clf_store):
        # kernel='linear', probability=False → exact LinearExplainer
        import shap

        svc = SVC(kernel="linear", probability=False)
        svc.fit(clf_store.X, clf_store.y)
        explainer = _select_explainer(svc, TaskType.CLASSIFICATION, clf_store.to_frame())
        assert isinstance(explainer, shap.LinearExplainer)

    def test_svc_linear_with_probability_uses_generic_explainer(self, clf_store):
        # kernel='linear', probability=True → Platt sigmoid wraps the decision
        # function, so LinearExplainer would explain the wrong output space.
        # Must fall through to KernelSHAP.
        import shap

        svc = SVC(kernel="linear", probability=True, random_state=42)
        svc.fit(clf_store.X, clf_store.y)
        explainer = _select_explainer(svc, TaskType.CLASSIFICATION, clf_store.to_frame())
        assert isinstance(explainer, shap.Explainer)
        assert not isinstance(explainer, shap.LinearExplainer)

    def test_svc_rbf_uses_generic_explainer(self, clf_store):
        import shap

        svc = SVC(kernel="rbf", probability=True, random_state=42)
        svc.fit(clf_store.X, clf_store.y)
        explainer = _select_explainer(svc, TaskType.CLASSIFICATION, clf_store.to_frame())
        assert isinstance(explainer, shap.Explainer)

    def test_svr_linear_no_probability_uses_linear_explainer(self, reg_store):
        import shap

        svr = SVR(kernel="linear")
        svr.fit(reg_store.X, reg_store.y)
        explainer = _select_explainer(svr, TaskType.REGRESSION, reg_store.to_frame())
        assert isinstance(explainer, shap.LinearExplainer)


# ---------------------------------------------------------------------------
# get_shap_values — end-to-end
# ---------------------------------------------------------------------------


class TestGetShapValues:
    def test_returns_tuple_of_array_and_series(self, fitted_rf_clf, clf_store):
        import pandas as pd

        shap_array, importance = get_shap_values(fitted_rf_clf, clf_store, TaskType.CLASSIFICATION)
        assert isinstance(shap_array, np.ndarray)
        assert isinstance(importance, pd.Series)

    def test_shap_array_shape(self, fitted_rf_clf, clf_store):
        shap_array, _ = get_shap_values(fitted_rf_clf, clf_store, TaskType.CLASSIFICATION)
        assert shap_array.shape == (N, len(FEATURE_NAMES))

    def test_importance_indexed_by_feature_names(self, fitted_rf_clf, clf_store):
        _, importance = get_shap_values(fitted_rf_clf, clf_store, TaskType.CLASSIFICATION)
        assert set(importance.index) == set(FEATURE_NAMES)

    def test_importance_sorted_descending(self, fitted_rf_clf, clf_store):
        _, importance = get_shap_values(fitted_rf_clf, clf_store, TaskType.CLASSIFICATION)
        assert list(importance.values) == sorted(importance.values, reverse=True)

    def test_regression_shap_array_shape(self, fitted_rf_reg, reg_store):
        shap_array, _ = get_shap_values(fitted_rf_reg, reg_store, TaskType.REGRESSION)
        assert shap_array.shape == (N, len(FEATURE_NAMES))

    def test_save_path_creates_file(self, fitted_rf_clf, clf_store, tmp_path):
        out = tmp_path / "shap.npy"
        get_shap_values(fitted_rf_clf, clf_store, TaskType.CLASSIFICATION, save_path=out)
        assert out.exists()
        loaded = np.load(out)
        assert loaded.shape == (N, len(FEATURE_NAMES))

    # -- multiclass default (mean-abs aggregation) ----------------------------

    def test_multiclass_default_shape(self, fitted_rf_multiclass, multiclass_store):
        shap_array, _ = get_shap_values(fitted_rf_multiclass, multiclass_store, TaskType.CLASSIFICATION)
        assert shap_array.shape == (N, len(FEATURE_NAMES))

    def test_multiclass_default_values_nonnegative(self, fitted_rf_multiclass, multiclass_store):
        shap_array, _ = get_shap_values(fitted_rf_multiclass, multiclass_store, TaskType.CLASSIFICATION)
        assert (shap_array >= 0).all()

    def test_multiclass_importance_sorted_descending(self, fitted_rf_multiclass, multiclass_store):
        _, importance = get_shap_values(fitted_rf_multiclass, multiclass_store, TaskType.CLASSIFICATION)
        assert list(importance.values) == sorted(importance.values, reverse=True)

    def test_multiclass_explicit_class_index_differs_from_default(self, fitted_rf_multiclass, multiclass_store):
        default_arr, _ = get_shap_values(fitted_rf_multiclass, multiclass_store, TaskType.CLASSIFICATION)
        class0_arr, _ = get_shap_values(
            fitted_rf_multiclass,
            multiclass_store,
            TaskType.CLASSIFICATION,
            class_index=0,
        )
        # class_index=0 extracts signed values for one class; default is mean-abs across all
        assert not np.array_equal(default_arr, class0_arr)


# ---------------------------------------------------------------------------
# get_oof_shap_values
# ---------------------------------------------------------------------------


class TestGetOofShapValues:
    @pytest.fixture
    def clf_step(self, clf_store):
        return make_classifier(RandomForestClassifier(n_estimators=10, random_state=42))

    @pytest.fixture
    def reg_step(self, reg_store):
        return make_regressor(RandomForestRegressor(n_estimators=10, random_state=42))

    def test_returns_tuple(self, clf_step, clf_store):
        import pandas as pd

        result = get_oof_shap_values(clf_step, clf_store, TaskType.CLASSIFICATION, n_splits=3)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], pd.Series)

    def test_oof_array_shape_matches_input(self, clf_step, clf_store):
        shap_arr, _ = get_oof_shap_values(clf_step, clf_store, TaskType.CLASSIFICATION, n_splits=3)
        assert shap_arr.shape == (N, len(FEATURE_NAMES))

    def test_all_samples_covered(self, clf_step, clf_store):
        """OOF: every row should be non-zero (each sample was explained once)."""
        shap_arr, _ = get_oof_shap_values(clf_step, clf_store, TaskType.CLASSIFICATION, n_splits=3)
        row_norms = np.abs(shap_arr).sum(axis=1)
        assert (row_norms > 0).all(), "Some samples were never assigned OOF SHAP values"

    def test_importance_indexed_by_feature_names(self, clf_step, clf_store):
        _, importance = get_oof_shap_values(clf_step, clf_store, TaskType.CLASSIFICATION, n_splits=3)
        assert set(importance.index) == set(FEATURE_NAMES)

    def test_importance_sorted_descending(self, clf_step, clf_store):
        _, importance = get_oof_shap_values(clf_step, clf_store, TaskType.CLASSIFICATION, n_splits=3)
        assert list(importance.values) == sorted(importance.values, reverse=True)

    def test_regression_shape(self, reg_step, reg_store):
        shap_arr, _ = get_oof_shap_values(reg_step, reg_store, TaskType.REGRESSION, n_splits=3)
        assert shap_arr.shape == (N, len(FEATURE_NAMES))

    def test_save_path_creates_file(self, clf_step, clf_store, tmp_path):
        out = tmp_path / "oof_shap.npy"
        get_oof_shap_values(clf_step, clf_store, TaskType.CLASSIFICATION, n_splits=3, save_path=out)
        assert out.exists()
        loaded = np.load(out)
        assert loaded.shape == (N, len(FEATURE_NAMES))

    def test_reproducible(self, clf_step, clf_store):
        a, _ = get_oof_shap_values(clf_step, clf_store, TaskType.CLASSIFICATION, n_splits=3, random_state=0)
        b, _ = get_oof_shap_values(clf_step, clf_store, TaskType.CLASSIFICATION, n_splits=3, random_state=0)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self, clf_step, clf_store):
        a, _ = get_oof_shap_values(clf_step, clf_store, TaskType.CLASSIFICATION, n_splits=3, random_state=0)
        b, _ = get_oof_shap_values(clf_step, clf_store, TaskType.CLASSIFICATION, n_splits=3, random_state=99)
        assert not np.array_equal(a, b)

    # -- multiclass default (mean-abs aggregation) ----------------------------

    @pytest.fixture
    def multiclass_store(self):
        rng = np.random.default_rng(7)
        return PipelineData(
            X=rng.standard_normal((N, len(FEATURE_NAMES))),
            feature_names=FEATURE_NAMES,
            y=rng.integers(0, 3, N),
        )

    @pytest.fixture
    def multiclass_clf_step(self):
        return make_classifier(RandomForestClassifier(n_estimators=10, random_state=42))

    def test_multiclass_oof_shape(self, multiclass_clf_step, multiclass_store):
        shap_arr, _ = get_oof_shap_values(multiclass_clf_step, multiclass_store, TaskType.CLASSIFICATION, n_splits=3)
        assert shap_arr.shape == (N, len(FEATURE_NAMES))

    def test_multiclass_oof_values_nonnegative(self, multiclass_clf_step, multiclass_store):
        shap_arr, _ = get_oof_shap_values(multiclass_clf_step, multiclass_store, TaskType.CLASSIFICATION, n_splits=3)
        assert (shap_arr >= 0).all()

    def test_multiclass_oof_all_samples_covered(self, multiclass_clf_step, multiclass_store):
        shap_arr, _ = get_oof_shap_values(multiclass_clf_step, multiclass_store, TaskType.CLASSIFICATION, n_splits=3)
        assert (np.abs(shap_arr).sum(axis=1) > 0).all()


# ---------------------------------------------------------------------------
# SVC / SVR SHAP — end-to-end
# ---------------------------------------------------------------------------


class TestSvcShapValues:
    """
    End-to-end tests for SHAP paths that involve SVC/SVR.
    These guard against regressions in explainer routing and
    the probability=True / LinearExplainer mismatch.
    """

    def test_svc_probability_true_oof_shape(self, clf_store):
        # Registry default: SVC(probability=True, random_state=42)
        # Must use KernelSHAP (not LinearExplainer) and complete without error.
        svc_step = make_classifier(
            SVC(kernel="linear", probability=True, random_state=42),
            requires_scaling=True,
        )
        shap_arr, importance = get_oof_shap_values(
            svc_step,
            clf_store,
            TaskType.CLASSIFICATION,
            n_splits=3,
            max_background=20,
        )
        assert shap_arr.shape == (N, len(FEATURE_NAMES))
        assert set(importance.index) == set(FEATURE_NAMES)

    def test_svc_probability_true_all_samples_covered(self, clf_store):
        svc_step = make_classifier(
            SVC(kernel="linear", probability=True, random_state=42),
            requires_scaling=True,
        )
        shap_arr, _ = get_oof_shap_values(
            svc_step,
            clf_store,
            TaskType.CLASSIFICATION,
            n_splits=3,
            max_background=20,
        )
        assert (np.abs(shap_arr).sum(axis=1) > 0).all()

    def test_svc_rbf_oof_shape(self, clf_store):
        svc_step = make_classifier(SVC(kernel="rbf", probability=True, random_state=42), requires_scaling=True)
        shap_arr, _ = get_oof_shap_values(
            svc_step,
            clf_store,
            TaskType.CLASSIFICATION,
            n_splits=3,
            max_background=20,
        )
        assert shap_arr.shape == (N, len(FEATURE_NAMES))

    def test_svr_oof_shape(self, reg_store):
        svr_step = make_regressor(SVR(kernel="rbf"), requires_scaling=True)
        shap_arr, importance = get_oof_shap_values(
            svr_step,
            reg_store,
            TaskType.REGRESSION,
            n_splits=3,
            max_background=20,
        )
        assert shap_arr.shape == (N, len(FEATURE_NAMES))
        assert set(importance.index) == set(FEATURE_NAMES)
