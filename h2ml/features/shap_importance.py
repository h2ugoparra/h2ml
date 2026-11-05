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

import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
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

# Rows per parallel task on the model-agnostic path. Fixed (not derived from n_jobs) so the
# per-chunk seeds — and therefore Permutation-explainer values — do not depend on worker count.
_SHAP_CHUNK_ROWS = 16


def _is_linear_svm(model: Any) -> bool:
    """
    True for linear SVC/SVR without Platt scaling — explained exactly by LinearExplainer.

    probability=True is excluded: Platt calibration wraps the decision function in a sigmoid,
    so coef_/intercept_ no longer represent what predict_proba returns.
    """
    return (
        model.__class__.__name__ in _LINEAR_SVM_CLASSES
        and getattr(model, "kernel", None) == "linear"
        and not getattr(model, "probability", False)
    )


def _uses_model_agnostic_explainer(model: Any, task_type: TaskType) -> bool:
    """True when _select_explainer routes the model to the slow model-agnostic explainer."""
    generic_models = _GENERIC_EXPLAINER_MODELS.get(task_type, set())
    return model.__class__.__name__ in generic_models and not _is_linear_svm(model)


def _summarize_background(
    background: pd.DataFrame,
    max_samples: int,
) -> pd.DataFrame:
    """
    Reduce a large background dataset for the model-agnostic explainer.

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
    seed: Optional[int] = None,
) -> shap.Explainer:
    """
    Route to the correct SHAP explainer.

    Tree-based models use TreeExplainer (faster, exact).
    Linear SVMs use LinearExplainer (exact, avoids the model-agnostic path entirely).
    Other distance/kernel models use the model-agnostic shap.Explainer — which
    resolves to ExactExplainer for <= 10 features and PermutationExplainer above
    that — with a summarized background to cap compute cost.

    Args:
        X_background:   Background dataset for generic/linear explainers.
                        Defaults to X when None. In OOF mode, pass the
                        training-fold DataFrame so the baseline reflects only
                        data the model actually saw.
        max_background: Maximum background rows passed to the model-agnostic
                        explainer. Larger backgrounds are replaced with shap.kmeans
                        centroids of this size (default 100).
        seed:           Seed for the PermutationExplainer's random permutations
                        (ignored by ExactExplainer). Note shap applies it via
                        np.random.seed, i.e. to the global RNG.
    """
    model_name = model.__class__.__name__
    generic_models = _GENERIC_EXPLAINER_MODELS.get(task_type, set())

    if model_name in generic_models:
        background = X_background if X_background is not None else X

        # Linear SVC/SVR: use exact LinearExplainer — avoids the model-agnostic path.
        # probability=True stays on the model-agnostic path (see _is_linear_svm), which
        # explains the actual probability output instead.
        if _is_linear_svm(model):
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
        return shap.Explainer(predict_fn, summarized, seed=seed)

    return shap.TreeExplainer(model)


def _explain_chunk(
    model: Any,
    task_type: TaskType,
    X_background: pd.DataFrame,
    X_chunk: pd.DataFrame,
    class_index: Optional[int],
    max_background: int,
    seed: int,
) -> np.ndarray:
    """
    Explain one chunk of rows with the model-agnostic explainer.

    Module-level so joblib can pickle it. Each row's SHAP values depend only on the
    model, the background and the row itself, so chunks are independent. The global
    numpy RNG (reseeded by PermutationExplainer) is restored afterwards.
    """
    rng_state = np.random.get_state()
    try:
        explainer = _select_explainer(
            model,
            task_type,
            X_chunk,
            X_background=X_background,
            max_background=max_background,
            seed=seed,
        )
        # Filters don't cross process boundaries — install inside the worker.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
            shap_output = explainer(X_chunk, silent=True)
    finally:
        np.random.set_state(rng_state)
    return _extract_shap_array(shap_output, class_index=class_index)


def _explain_chunks(
    chunks: list[tuple[Any, pd.DataFrame, pd.DataFrame]],
    task_type: TaskType,
    class_index: Optional[int],
    max_background: int,
    random_state: int,
    n_jobs: int,
) -> list[np.ndarray]:
    """
    Explain (model, background, X_chunk) chunks in parallel; results in input order.

    Chunk i is seeded with random_state + i, so values are reproducible and
    independent of n_jobs. A tqdm bar shows chunk progress, elapsed time and ETA;
    total elapsed time is logged on completion.
    """
    from tqdm import tqdm

    logger.info(f"Explaining {len(chunks)} chunks of <= {_SHAP_CHUNK_ROWS} rows (n_jobs={n_jobs})")
    start = time.perf_counter()
    # Ordered generator: results arrive in input order as they complete, so the bar
    # can advance without collecting everything first.
    results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
        delayed(_explain_chunk)(model, task_type, background, X_chunk, class_index, max_background, random_state + i)
        for i, (model, background, X_chunk) in enumerate(chunks)
    )
    values = cast(list[np.ndarray], list(tqdm(results, total=len(chunks), desc="SHAP", unit="chunk", leave=True)))
    logger.info(f"SHAP explanation finished in {time.perf_counter() - start:.1f}s")
    return values


def _split_rows(X_frame: pd.DataFrame) -> list[slice]:
    """Fixed-size row slices of X_frame (see _SHAP_CHUNK_ROWS)."""
    return [slice(start, start + _SHAP_CHUNK_ROWS) for start in range(0, len(X_frame), _SHAP_CHUNK_ROWS)]


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
    n_jobs: int = 1,
    random_state: int = 42,
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
        max_background: Max background samples for the model-agnostic explainer (non-tree,
                        non-linear models). Larger datasets are summarised with shap.kmeans.
                        Default 100.
        n_jobs:         Worker processes for the model-agnostic explainer, which explains
                        row chunks in parallel. Tree/linear explainers ignore it. Default 1.
        random_state:   Base seed for the PermutationExplainer (> 10 features). Default 42.

    Returns:
        shap_array:         np.ndarray of shape (n_samples, n_features).
        feature_importance: pd.Series indexed by feature name, sorted descending.
    """
    logger.info(f"Computing SHAP values for {model.__class__.__name__}")

    # SHAP needs a DataFrame for column names — convert locally
    X_frame = store.to_frame()

    if _uses_model_agnostic_explainer(model, task_type):
        background = _summarize_background(X_frame, max_background)
        chunks = [(model, background, X_frame.iloc[rows]) for rows in _split_rows(X_frame)]
        values = _explain_chunks(chunks, task_type, class_index, max_background, random_state, n_jobs)
        shap_array = np.vstack(values)
    else:
        explainer = _select_explainer(model, task_type, X_frame, max_background=max_background)
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
    train_target_fn: Optional[Callable] = None,
    n_jobs: int = 1,
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
        random_state:   Seed for KFold shuffle and base seed for the PermutationExplainer
                        (default 42).
        save_path:      Optional path to save the OOF SHAP array as .npy.
        class_index:    For classifiers, which class's SHAP values to use.
                        None (default): binary uses class 1 (positive class);
                        multiclass aggregates mean absolute SHAP across all classes.
                        Pass an integer to inspect a specific class.
        max_background: Max background samples passed to the model-agnostic explainer
                        per fold. Training folds larger than this are summarised with
                        shap.kmeans, giving a significant speedup for SVC/SVR
                        with non-linear kernels. Default 100.
        train_target_fn: Fitted on and applied to each training fold's targets
                        (e.g. winsorize clipping), so its statistics never see
                        held-out rows.
        n_jobs:         Worker processes for the model-agnostic explainer (SVC/SVR with
                        non-linear kernels, KNN, ...). Folds are fitted serially, then
                        all folds' test rows are explained in parallel chunks — results
                        do not depend on n_jobs. Tree/linear explainers ignore it. Default 1.

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
    explained = np.zeros(n_samples, dtype=bool)

    def fit_fold(train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        if train_target_fn is not None:
            y_train = train_target_fn(y_train)

        if requires_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = estimator_cls(**estimator_params)
        model.fit(X_train, y_train)
        return model, pd.DataFrame(X_train, columns=feature_names), pd.DataFrame(X_test, columns=feature_names)

    if _uses_model_agnostic_explainer(step.estimator, task_type):
        # Explanation dominates the cost (seconds per row vs. sub-second fits), so fit
        # folds serially here, then explain every fold's test rows in parallel chunks.
        chunk_rows: list[np.ndarray] = []
        chunks: list[tuple[Any, pd.DataFrame, pd.DataFrame]] = []
        for train_idx, test_idx in splitter.split(X, y):
            try:
                model, X_train_frame, X_test_frame = fit_fold(train_idx, test_idx)
            except Exception as e:
                logger.warning(f"OOF SHAP: fold skipped — {e}")
                continue
            background = _summarize_background(X_train_frame, max_background)
            for rows in _split_rows(X_test_frame):
                chunk_rows.append(test_idx[rows])
                chunks.append((model, background, X_test_frame.iloc[rows]))
        values = _explain_chunks(chunks, task_type, class_index, max_background, random_state, n_jobs) if chunks else []
        for idx, chunk_values in zip(chunk_rows, values, strict=True):
            oof_shap[idx] = chunk_values
            explained[idx] = True
    else:
        for train_idx, test_idx in splitter.split(X, y):
            try:
                model, X_train_frame, X_test_frame = fit_fold(train_idx, test_idx)

                explainer = _select_explainer(
                    model,
                    task_type,
                    X_test_frame,
                    X_background=X_train_frame,
                    max_background=max_background,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
                    shap_output = explainer(X_test_frame)

                oof_shap[test_idx] = _extract_shap_array(shap_output, class_index=class_index)
                explained[test_idx] = True
            except Exception as e:
                logger.warning(f"OOF SHAP: fold skipped — {e}")

    if not explained.any():
        raise RuntimeError(f"OOF SHAP: every fold failed for {estimator_cls.__name__}.")

    if save_path is not None:
        np.save(save_path, oof_shap)
        logger.info(f"OOF SHAP values saved at {save_path}")

    # Rows of skipped folds hold zeros, which would dilute the importance of every feature
    importance = np.abs(oof_shap[explained]).mean(axis=0)
    feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)

    return oof_shap, feature_importance
