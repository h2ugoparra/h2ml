"""
h2ml/features/shap_importance.py

SHAP-based feature importance computation.

Responsibilities:
    - Select the right SHAP explainer based on model type
    - Handle output shape differences between classifiers and regressors
    - Return raw SHAP array and ranked feature importance Series
    - Optionally save SHAP values to disk

What this module does NOT do:
    - Feature selection / filtering     → features/selector.py
    - Correlation computation           → features/correlation.py
    - Scaling                           → CV engine handles this
"""

from __future__ import annotations

import warnings
from loguru import logger
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from h2ml.pipeline.base import TaskType
from h2ml.features.feature_store import PipelineData


# ---------------------------------------------------------------------------
# Explainer routing
# ---------------------------------------------------------------------------

# Models that require a generic Explainer (predict / predict_proba based)
# rather than TreeExplainer. Extend as needed.
_GENERIC_EXPLAINER_MODELS: dict[TaskType, set[str]] = {
    TaskType.CLASSIFICATION: {
        "SVC",
        "LogisticRegression",
        "CalibratedClassifierCV",
        "AdaBoostClassifier",
        "GaussianNB",
        "BaggingClassifier",
        "KNeighborsClassifier",
    },
    TaskType.REGRESSION: {
        "SVR",
        "Ridge",
        "Lasso",
        "LinearRegression",
        "KNeighborsRegressor",
        "PoissonRegressor",
        "AdaBoostRegressor",
        "BaggingRegressor",
    },
}


_LINEAR_SVM_CLASSES: set[str] = {"SVC", "SVR"}


def _summarize_background(
    background: pd.DataFrame,
    max_samples: int,
) -> pd.DataFrame:
    """
    Reduce a large background dataset for KernelSHAP.

    Uses shap.kmeans when the background exceeds max_samples; this preserves
    the data distribution better than random subsampling and is the canonical
    SHAP recommendation for slow (non-tree) models.

    shap.kmeans returns a DenseData object, which is not a callable masker and
    causes PermutationExplainer to crash. We convert the centroids back to a
    DataFrame so shap.Explainer can use them as a standard background.
    """
    if len(background) <= max_samples:
        return background
    kmeans = shap.kmeans(background, max_samples)
    return pd.DataFrame(kmeans.data, columns=background.columns)


def _select_explainer(
    model: Any,
    task_type: TaskType,
    X: pd.DataFrame,
    X_background: Optional[pd.DataFrame] = None,
    max_background: int = 100,
) -> shap.Explainer:
    """
    Route to the correct SHAP explainer.

    Tree-based models use TreeExplainer (faster, exact).
    Linear SVMs use LinearExplainer (exact, avoids KernelSHAP entirely).
    Other distance/kernel models use generic Explainer (KernelSHAP) with a
    summarized background to cap compute cost.

    Args:
        X_background:   Background dataset for generic/linear explainers.
                        Defaults to X when None. In OOF mode, pass the
                        training-fold DataFrame so the baseline reflects only
                        data the model actually saw.
        max_background: Maximum background rows passed to KernelSHAP.
                        Larger backgrounds are replaced with shap.kmeans
                        centroids of this size (default 100).
    """
    model_name = model.__class__.__name__
    generic_models = _GENERIC_EXPLAINER_MODELS.get(task_type, set())

    if model_name in generic_models:
        background = X_background if X_background is not None else X

        # Linear SVC/SVR: use exact LinearExplainer — avoids KernelSHAP entirely.
        # Skip when probability=True: Platt calibration wraps the decision function
        # in a sigmoid, so coef_/intercept_ no longer represent what predict_proba
        # returns. KernelSHAP (below) explains the actual probability output instead.
        if (
            model_name in _LINEAR_SVM_CLASSES
            and getattr(model, "kernel", None) == "linear"
            and not getattr(model, "probability", False)
        ):
            return shap.LinearExplainer(model, background)

        if task_type == TaskType.CLASSIFICATION:
            # predict_proba may raise AttributeError for SVC(probability=False);
            # fall back to decision_function or predict in that case.
            try:
                base_fn = model.predict_proba
            except AttributeError:
                base_fn = (
                    model.decision_function
                    if hasattr(model, "decision_function")
                    else model.predict
                )
        else:
            base_fn = model.predict

        # Models fitted on numpy arrays warn when called with a DataFrame.
        # Strip column names before passing to keep sklearn happy.
        def predict_fn(X):
            return base_fn(X.values if isinstance(X, pd.DataFrame) else X)

        summarized = _summarize_background(background, max_background)
        return shap.Explainer(predict_fn, summarized)

    return shap.TreeExplainer(model)


def _extract_shap_array(
    shap_values: Any, class_index: Optional[int] = None
) -> np.ndarray:
    """
    Normalize SHAP output to a 2D array (n_samples, n_features).

    Classification returns (n_samples, n_features, n_classes) — collapsed to 2D:
      - class_index=None, binary (n_classes=2): extract class 1 (positive class).
      - class_index=None, multiclass (n_classes>2): mean absolute SHAP across all
        classes. This avoids an arbitrary single-class choice and captures overall
        feature contribution regardless of direction.
      - class_index=k (explicit): extract that specific class; raises if out of range.
    Regression returns (n_samples, n_features) — used as-is.

    Args:
        shap_values:  Raw output from a SHAP explainer.
        class_index:  Which class to extract (None = auto, see above).
                      Ignored for regressors. Raises if negative or out of range.
    """
    arr = shap_values.values if hasattr(shap_values, "values") else shap_values

    if arr.ndim == 3:
        n_classes = arr.shape[-1]
        if class_index is None:
            if n_classes == 2:
                return arr[..., 1]
            logger.debug(
                f"Multiclass SHAP ({n_classes} classes): aggregating mean absolute "
                "values across all classes. Pass class_index=k to inspect a specific class."
            )
            return np.abs(arr).mean(axis=-1)
        if class_index < 0 or class_index >= n_classes:
            raise ValueError(
                f"class_index={class_index} is out of range for a model with "
                f"{n_classes} classes. Valid range: 0–{n_classes - 1}."
            )
        return arr[..., class_index]

    return arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_shap_values(
    model: Any,
    store: PipelineData,
    task_type: TaskType,
    save_path: Optional[Path] = None,
    class_index: Optional[int] = None,
    max_background: int = 100,
) -> tuple[np.ndarray, pd.Series]:
    """
    Compute SHAP values and ranked feature importance for a fitted model.

    Args:
        model:          A fitted sklearn-compatible model (inner estimator, not PipelineStep).
        store:          PipelineData — feature names are read from here, X converted
                        to DataFrame locally for SHAP.
        task_type:      TaskType.CLASSIFICATION or TaskType.REGRESSION.
        save_path:      Optional path to save the SHAP array as .npy.
        class_index:    For classifiers, which class's SHAP values to use.
                        None (default): binary uses class 1 (positive class);
                        multiclass aggregates mean absolute SHAP across all classes.
                        Pass an integer to inspect a specific class.
                        Ignored for regressors.
        max_background: Max background samples for KernelSHAP (non-tree, non-linear models).
                        Larger datasets are summarised with shap.kmeans. Default 100.

    Returns:
        shap_array:         np.ndarray of shape (n_samples, n_features).
        feature_importance: pd.Series indexed by feature name, sorted descending.
    """
    logger.info(f"Computing SHAP values for {model.__class__.__name__}")

    # SHAP needs a DataFrame for column names — convert locally
    X_frame = store.to_frame()

    explainer = _select_explainer(
        model, task_type, X_frame, max_background=max_background
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*does not have valid feature names.*"
        )
        shap_output = explainer(X_frame)
    shap_array = _extract_shap_array(shap_output, class_index=class_index)

    if save_path is not None:
        np.save(save_path, shap_array)
        logger.info(f"SHAP values saved at {save_path}")

    # Mean absolute SHAP per feature → overall importance
    importance = np.abs(shap_array).mean(axis=0)
    feature_importance = pd.Series(importance, index=store.feature_names).sort_values(
        ascending=False
    )

    return shap_array, feature_importance


def get_oof_shap_values(
    step: Any,
    store: PipelineData,
    task_type: TaskType,
    n_splits: int = 5,
    random_state: int = 42,
    save_path: Optional[Path] = None,
    class_index: Optional[int] = None,
    max_background: int = 100,
    _splitter: Any = None,
) -> tuple[np.ndarray, pd.Series]:
    """
    Out-of-fold SHAP: each sample's importance is computed when it is held out.

    For each fold, a fresh model is trained on the training split and SHAP values
    are computed on the test split. The results are assembled into a single OOF
    matrix where no sample's SHAP values leak from its own fold's training data.

    Args:
        step:           ModelWrapper (or any object with .estimator and .requires_scaling).
                        Used as a prototype — its class and params seed each fold's model.
        store:          PipelineData containing X, y, and feature_names.
        task_type:      TaskType.CLASSIFICATION or TaskType.REGRESSION.
        n_splits:       Number of CV folds (default 5).
        random_state:   Seed for KFold shuffle (default 42).
        save_path:      Optional path to save the OOF SHAP array as .npy.
        class_index:    For classifiers, which class's SHAP values to use.
                        None (default): binary uses class 1 (positive class);
                        multiclass aggregates mean absolute SHAP across all classes.
                        Pass an integer to inspect a specific class.
        max_background: Max background samples passed to KernelSHAP per fold.
                        Training folds larger than this are summarised with
                        shap.kmeans, giving a significant speedup for SVC/SVR
                        with non-linear kernels. Default 100.

    Returns:
        oof_shap:           np.ndarray of shape (n_samples, n_features).
        feature_importance: pd.Series indexed by feature name, sorted descending.
    """
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler

    X = store.X
    y = store.y
    feature_names = store.feature_names
    n_samples = X.shape[0]
    n_features = len(feature_names)

    if _splitter is not None:
        splitter = _splitter
    else:
        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            if task_type == TaskType.CLASSIFICATION
            else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        )

    estimator_cls = step.estimator.__class__
    estimator_params = step.estimator.get_params()
    requires_scaling = getattr(step, "requires_scaling", False)

    logger.info(
        f"Computing OOF SHAP values for {estimator_cls.__name__} "
        f"({n_splits}-fold, n_samples={n_samples})"
    )

    oof_shap = np.zeros((n_samples, n_features), dtype=np.float64)

    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        if requires_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = estimator_cls(**estimator_params)
        model.fit(X_train, y_train)

        X_train_frame = pd.DataFrame(X_train, columns=feature_names)
        X_test_frame = pd.DataFrame(X_test, columns=feature_names)

        explainer = _select_explainer(
            model,
            task_type,
            X_test_frame,
            X_background=X_train_frame,
            max_background=max_background,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*does not have valid feature names.*"
            )
            shap_output = explainer(X_test_frame)

        oof_shap[test_idx] = _extract_shap_array(shap_output, class_index=class_index)

    if save_path is not None:
        np.save(save_path, oof_shap)
        logger.info(f"OOF SHAP values saved at {save_path}")

    importance = np.abs(oof_shap).mean(axis=0)
    feature_importance = pd.Series(importance, index=feature_names).sort_values(
        ascending=False
    )

    return oof_shap, feature_importance
