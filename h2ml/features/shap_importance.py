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
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
from loguru import logger

from h2ml.core.base import TaskType
from h2ml.core.feature_store import PipelineData

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

# TabPFN needs shapiq's TabPFNExplainer, which removes features by re-contextualising the
# in-context training set instead of refitting per coalition like KernelSHAP.
_TABPFN_MODELS: set[str] = {"TabPFNClassifier", "TabPFNRegressor"}

# Rows of labelled context handed to TabPFN when explaining. Distinct from max_background:
# shap's background can be kmeans centroids, but TabPFN's context needs real labelled rows.
_TABPFN_MAX_CONTEXT: int = 1_000


class _TabPFNExplainerAdapter:
    """
    Present shapiq's TabPFNExplainer through the shap explainer call protocol.

    Calling the instance with a DataFrame returns an object exposing `.values` shaped
    (n_samples, n_features) or (n_samples, n_features, n_classes), so _extract_shap_array
    and every downstream caller treat TabPFN exactly like any other model.

    Multiclass note: shapiq explains one class per explainer, so with class_index=None and
    more than two classes we build one explainer per class and stack them on the last axis.
    That reproduces h2ml's mean-absolute-across-classes convention at k times the cost.
    """

    def __init__(
        self,
        model: Any,
        task_type: TaskType,
        X_background: pd.DataFrame,
        y_background: np.ndarray,
        class_index: Optional[int] = None,
        max_context: int = _TABPFN_MAX_CONTEXT,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.task_type = task_type
        self.class_index = class_index
        self.random_state = random_state

        ctx_X = np.asarray(X_background)
        ctx_y = np.asarray(y_background)

        # The context is TabPFN's training set for every coalition it evaluates, so its size
        # drives the cost of the whole explanation. Subsample rather than summarise: kmeans
        # centroids have no labels and cannot serve as context.
        if len(ctx_X) > max_context:
            rng = np.random.default_rng(random_state)
            keep = rng.choice(len(ctx_X), size=max_context, replace=False)
            ctx_X, ctx_y = ctx_X[keep], ctx_y[keep]

        self._ctx_X = ctx_X
        self._ctx_y = ctx_y

        if task_type != TaskType.CLASSIFICATION:
            self._class_indices: list[Optional[int]] = [None]
        elif class_index is not None:
            self._class_indices = [class_index]
        else:
            n_classes = len(np.unique(ctx_y))
            # Binary: class 1, matching _extract_shap_array's positive-class convention.
            self._class_indices = [1] if n_classes <= 2 else list(range(n_classes))

    def _explain_one_class(self, X: np.ndarray, class_index: Optional[int]) -> np.ndarray:
        import shapiq

        explainer = shapiq.TabPFNExplainer(
            model=self.model,
            data=self._ctx_X,
            labels=self._ctx_y,
            index="SV",
            max_order=1,
            x_test=X,
            class_index=class_index,
        )
        # n_jobs=1: this may already run inside an outer joblib pool, and TabPFN is memory-heavy.
        values = explainer.explain_X(X, n_jobs=1, random_state=self.random_state)
        return np.vstack([iv.get_n_order_values(1) for iv in values])

    def __call__(self, X: Any) -> SimpleNamespace:
        X_arr = np.asarray(X)
        per_class = [self._explain_one_class(X_arr, ci) for ci in self._class_indices]
        if len(per_class) == 1:
            return SimpleNamespace(values=per_class[0])
        return SimpleNamespace(values=np.stack(per_class, axis=-1))


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
    y_background: Optional[np.ndarray] = None,
    class_index: Optional[int] = None,
    random_state: int = 42,
) -> Any:
    """
    Route to the correct SHAP explainer.

    Returns a shap.Explainer, or for TabPFN a _TabPFNExplainerAdapter that satisfies the
    same call protocol (callable with a DataFrame → object exposing .values).

    Tree-based models use TreeExplainer (faster, exact).
    Linear SVMs use LinearExplainer (exact, avoids KernelSHAP entirely).
    TabPFN uses shapiq's TabPFNExplainer, wrapped to look like a shap explainer.
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
        y_background:   Labels aligned to X_background. Required for TabPFN —
                        shapiq needs them as in-context training labels.
        class_index:    TabPFN only. shapiq explains one class per explainer, so the
                        choice has to be made here rather than left to
                        _extract_shap_array. Other explainers return every class and
                        are sliced afterwards. Same semantics either way.
        random_state:   Seed for TabPFN context subsampling.
    """
    model_name = model.__class__.__name__
    generic_models = _GENERIC_EXPLAINER_MODELS.get(task_type, set())

    if model_name in _TABPFN_MODELS:
        if y_background is None:
            raise ValueError(
                f"{model_name} SHAP requires y_background — shapiq's TabPFNExplainer needs "
                "in-context training labels alongside X_background."
            )
        return _TabPFNExplainerAdapter(
            model,
            task_type,
            X_background if X_background is not None else X,
            y_background,
            class_index=class_index,
            random_state=random_state,
        )

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
                base_fn = model.decision_function if hasattr(model, "decision_function") else model.predict
        else:
            base_fn = model.predict

        # Models fitted on numpy arrays warn when called with a DataFrame.
        # Strip column names before passing to keep sklearn happy.
        def predict_fn(X):
            return base_fn(X.values if isinstance(X, pd.DataFrame) else X)

        summarized = _summarize_background(background, max_background)
        return shap.Explainer(predict_fn, summarized)

    return shap.TreeExplainer(model)


def _extract_shap_array(shap_values: Any, class_index: Optional[int] = None) -> np.ndarray:
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
        model,
        task_type,
        X_frame,
        max_background=max_background,
        y_background=store.y,
        class_index=class_index,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
        shap_output = explainer(X_frame)
    shap_array = _extract_shap_array(shap_output, class_index=class_index)

    if save_path is not None:
        np.save(save_path, shap_array)
        logger.info(f"SHAP values saved at {save_path}")

    # Mean absolute SHAP per feature → overall importance
    importance = np.abs(shap_array).mean(axis=0)
    feature_importance = pd.Series(importance, index=store.feature_names).sort_values(ascending=False)

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
    from sklearn.model_selection import KFold, StratifiedKFold
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

    logger.info(f"Computing OOF SHAP values for {estimator_cls.__name__} ({n_splits}-fold, n_samples={n_samples})")

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
            y_background=y_train,
            class_index=class_index,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
            shap_output = explainer(X_test_frame)

        oof_shap[test_idx] = _extract_shap_array(shap_output, class_index=class_index)

    if save_path is not None:
        np.save(save_path, oof_shap)
        logger.info(f"OOF SHAP values saved at {save_path}")

    importance = np.abs(oof_shap).mean(axis=0)
    feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)

    return oof_shap, feature_importance
