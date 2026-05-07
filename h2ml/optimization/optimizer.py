"""
h2ml/optimization/optimizer.py

Optuna-based hyperparameter optimization engine.

Responsibilities:
    - Build and run an Optuna study for a registered model
    - Compute cross-validated score as the objective function
    - Return the best params via study.best_params
    - Support both classification and regression tasks

What this module does NOT do:
    - Define hyperparameter search spaces  → optimization/params/
    - Register models                      → optimization/opt_params.py
    - Run the final CV with best params    → pipeline/pipeline.py (_run_step4)

Interface contract (expected by pipeline/pipeline.py):
    study = run_study(name, X, y, task, n_trials)
    result.best_params = study.best_params
"""

from __future__ import annotations

from loguru import logger
from typing import Any, Callable, Optional

import numpy as np
import optuna
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from h2ml.optimization.opt_params import ModelEntry, get_entry


# Silence Optuna's default verbose logging — pipeline controls verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Metrics whose objective returns a negated value for internal maximisation.
# Used to flip display values back to their natural (positive) direction.
_NEGATED_METRICS: frozenset[str] = frozenset({"LogLoss", "Brier", "MAE", "RMSE"})


def _to_display(value: float, metric: str) -> float:
    """Convert an internal objective value to its natural display value."""
    return -value if metric in _NEGATED_METRICS else value


def _make_trial_callback(metric: str, n_trials: int) -> "tuple[Callable, Any]":
    """
    Return (callback, pbar) for study.optimize.

    The tqdm progress bar shows trial count and the best value with the correct
    sign (negated metrics displayed as positive). Call pbar.close() after
    study.optimize completes.
    """
    from tqdm import tqdm

    pbar = tqdm(total=n_trials, desc=f"HPO ({metric})", unit="trial", leave=True)

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        pbar.update(1)
        if study.best_trial is not None:
            pbar.set_postfix(
                {f"best_{metric}": f"{_to_display(study.best_value, metric):.4f}"}
            )

    return _cb, pbar


# ---------------------------------------------------------------------------
# Objective metric functions
# All functions return higher-is-better values (negation applied where needed).
# ---------------------------------------------------------------------------


def _score_auc(model, X_train, X_test, y_train, y_test) -> float:
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    if proba.shape[1] == 2:
        return float(roc_auc_score(y_test, proba[:, 1]))
    return float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))


def _score_auc_pr(model, X_train, X_test, y_train, y_test) -> float:
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    if proba.shape[1] == 2:
        return float(average_precision_score(y_test, proba[:, 1]))
    lb = LabelBinarizer().fit(y_train)
    return float(
        average_precision_score(lb.transform(y_test), proba, average="weighted")
    )


def _score_logloss(model, X_train, X_test, y_train, y_test) -> float:
    """Negated log-loss so higher is better."""
    model.fit(X_train, y_train)
    return float(-log_loss(y_test, model.predict_proba(X_test)))


def _score_brier(model, X_train, X_test, y_train, y_test) -> float:
    """Negated Brier score so higher is better."""
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    if proba.shape[1] == 2:
        return float(-brier_score_loss(y_test, proba[:, 1]))
    lb = LabelBinarizer().fit(y_train)
    y_bin = lb.transform(y_test)
    return float(-np.mean(np.sum((y_bin - proba) ** 2, axis=1)))


def _score_f1(model, X_train, X_test, y_train, y_test) -> float:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    average = "binary" if len(np.unique(y_test)) == 2 else "weighted"
    return float(f1_score(y_test, y_pred, average=average, zero_division=0))


def _score_r2(
    model, X_train, X_test, y_train, y_test, *, y_true_test=None, inverse_fn=None
) -> float:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if inverse_fn is not None:
        y_pred = inverse_fn(y_pred)
        y_test = y_true_test
    return float(r2_score(y_test, y_pred))


def _score_mae(
    model, X_train, X_test, y_train, y_test, *, y_true_test=None, inverse_fn=None
) -> float:
    """Negated MAE so higher is better."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if inverse_fn is not None:
        y_pred = inverse_fn(y_pred)
        y_test = y_true_test
    return float(-mean_absolute_error(y_test, y_pred))


def _score_rmse(
    model, X_train, X_test, y_train, y_test, *, y_true_test=None, inverse_fn=None
) -> float:
    """Negated RMSE so higher is better."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if inverse_fn is not None:
        y_pred = inverse_fn(y_pred)
        y_test = y_true_test
    return float(-np.sqrt(mean_squared_error(y_test, y_pred)))


# Registry of available metrics per task.
# All entries return higher-is-better values — study direction stays 'maximize'.
CLF_METRICS: dict[str, Callable] = {
    "AUC": _score_auc,
    "AUC_PR": _score_auc_pr,
    "LogLoss": _score_logloss,
    "F1": _score_f1,
    "Brier": _score_brier,
}

REG_METRICS: dict[str, Callable] = {
    "R2": _score_r2,
    "MAE": _score_mae,
    "RMSE": _score_rmse,
}

_TASK_METRICS: dict[str, dict[str, Callable]] = {
    "classification": CLF_METRICS,
    "regression": REG_METRICS,
}

_DEFAULT_METRIC: dict[str, str] = {
    "classification": "AUC",
    "regression": "R2",
}

_SPLITTER = {
    "classification": StratifiedKFold,
    "regression": KFold,
}


# ---------------------------------------------------------------------------
# Objective builder
# ---------------------------------------------------------------------------


def _build_objective(
    entry: ModelEntry,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    n_splits: int,
    random_state: int,
    score_fn: Callable,
    coords: Optional[np.ndarray] = None,
    n_blocks_per_fold: int = 5,
    spatial_cv_method: str = "block",
    ahc_threshold: Optional[float] = None,
    spatial_cv_metric: str = "euclidean",
    pca_components: float = 0.95,
    exact_max_samples: int = 5_000,
    knn_neighbors: int = 15,
    fixed_params: Optional[dict] = None,
    y_true: Optional[np.ndarray] = None,
    inverse_fn: Optional[Callable] = None,
    _splitter: Any = None,
):
    """
    Returns a closure that Optuna calls on each trial.

    The objective:
        1. Samples hyperparameters via entry.param_fn(trial)
        2. Instantiates the model
        3. Applies scaling inside each fold if requires_scaling
        4. Computes mean CV score across n_splits folds
        5. Reports intermediate (per-fold) scores for pruning

    Fold indices and scaled arrays are pre-computed once at construction time —
    they are identical across all trials since X and y are fixed.

    _splitter: pre-built spatial splitter (already cloned with correct n_splits and
               random_state for this repeat). When provided, splitter construction
               is skipped entirely — avoiding redundant AHC in SPCVSplitter.
    """
    if _splitter is not None:
        splitter = _splitter
    elif coords is not None:
        if spatial_cv_method == "spcv":
            from h2ml.features.spatial_cv import SPCVSplitter

            splitter = SPCVSplitter(
                coords=coords,
                X=X,
                y=y,
                n_splits=n_splits,
                threshold=ahc_threshold,
                random_state=random_state,
                metric=spatial_cv_metric,
                pca_components=pca_components,
                exact_max_samples=exact_max_samples,
                knn_neighbors=knn_neighbors,
            )
        else:
            from h2ml.features.spatial_cv import SpatialBlockSplitter

            splitter = SpatialBlockSplitter(
                coords=coords,
                n_splits=n_splits,
                n_blocks_per_fold=n_blocks_per_fold,
                random_state=random_state,
                metric=spatial_cv_metric,
            )
    else:
        splitter = _SPLITTER[task](
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    fixed_params = fixed_params or {}

    # Pre-compute fold indices once — avoids repeated split() calls per trial
    fold_indices = list(splitter.split(X, y))

    # Pre-compute scaled fold arrays once for models that require scaling.
    # The scaler fit is deterministic given the same fold split, so recomputing
    # it every trial is pure waste.
    scaled_folds: list[tuple[np.ndarray, np.ndarray]] | None = None
    if getattr(entry, "requires_scaling", False):
        scaled_folds = []
        for train_idx, test_idx in fold_indices:
            scaler = StandardScaler()
            scaled_folds.append(
                (
                    scaler.fit_transform(X[train_idx]),
                    scaler.transform(X[test_idx]),
                )
            )

    # Pre-compute original-scale test slices for regression transforms.
    # When y_true is provided, the score functions receive original-scale y so
    # HPO optimises the same metric that the pipeline reports.
    y_true_folds: list[np.ndarray] | None = (
        [y_true[test_idx] for _, test_idx in fold_indices]
        if y_true is not None
        else None
    )

    def objective(trial: optuna.Trial) -> float:
        if entry.param_fn is None:
            raise ValueError(
                f"Model '{entry.model_cls.__class__.__name__}' has no param_fn defined. "
                f"Add its search space to optimization/params/"
                f"{'classifiers' if task == 'classification' else 'regressors'}.py"
            )
        params = {**entry.param_fn(trial), **fixed_params}
        model = entry.model_cls(**params)

        fold_scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
            if scaled_folds is not None:
                X_train, X_test = scaled_folds[fold_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if task == "regression" and inverse_fn is not None:
                score = score_fn(
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    y_true_test=y_true_folds[fold_idx],
                    inverse_fn=inverse_fn,
                )
            else:
                score = score_fn(model, X_train, X_test, y_train, y_test)
            fold_scores.append(score)

            # Report running mean so the pruner can kill unpromising trials early
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_study(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    task: str = "classification",
    metric: Optional[str] = None,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    study_name: Optional[str] = None,
    verbose: bool = True,
    coords: Optional[np.ndarray] = None,
    n_blocks_per_fold: int = 5,
    spatial_cv_method: str = "block",
    ahc_threshold: Optional[float] = None,
    spatial_cv_metric: str = "euclidean",
    pca_components: float = 0.95,
    exact_max_samples: int = 5_000,
    knn_neighbors: int = 15,
    n_hpo_repeats: int = 1,
    fixed_params: Optional[dict] = None,
    y_true: Optional[np.ndarray] = None,
    inverse_fn: Optional[Callable] = None,
    _splitter: Any = None,
) -> optuna.Study:
    """
    Run an Optuna hyperparameter optimization study for a registered model.

    Args:
        name:           Model class name (e.g. 'RandomForestClassifier').
                        Must be registered in CLASSIFIER_OPT_PARAMS or
                        REGRESSOR_OPT_PARAMS.
        X:              Feature matrix as numpy array (n_samples, n_features).
        y:              Target array (n_samples,).
        task:           'classification' or 'regression'.
        metric:         Scoring metric for the objective. Must be a key in
                        CLF_METRICS (classification) or REG_METRICS (regression).
                        Defaults to 'AUC' for classification, 'R2' for regression.
                        All metrics are maximised internally (error metrics are negated).
        n_trials:       Total number of Optuna trials across all repeats (default 50).
                        Divided evenly: trials_per_repeat = max(1, n_trials // n_hpo_repeats).
        n_splits:       CV folds inside each trial objective (default 5).
        random_state:   Base seed for reproducibility. Each repeat uses random_state + repeat_idx
                        for both the TPE sampler and fold splitter (default 42).
        study_name:     Optional Optuna study name for logging/storage.
        verbose:        If True, log trial progress.
        n_hpo_repeats:  Number of independent HPO runs with different fold seeds (default 1).
                        Each repeat draws fold splits from a different random seed, amortising
                        the risk of an unlucky single split. The study with the highest
                        best_value is returned.

    Returns:
        optuna.Study with best_params, best_value, and full trial history.
        When n_hpo_repeats > 1, this is the repeat whose best_value was highest.

    Raises:
        KeyError:   If the model is not registered.
        ValueError: If the model is registered but disabled, or metric is invalid.

    Example:
        >>> study = run_study("RandomForestClassifier", X, y, task="classification")
        >>> study.best_params
        {'n_estimators': 300, 'max_depth': 8, ...}
        >>> study.best_value
        0.923
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")

    available = _TASK_METRICS[task]
    resolved_metric = metric or _DEFAULT_METRIC[task]
    if resolved_metric not in available:
        raise ValueError(
            f"metric '{resolved_metric}' is not valid for task '{task}'. "
            f"Available: {list(available)}"
        )
    score_fn = available[resolved_metric]

    entry = get_entry(name, task)

    trials_per_repeat = max(1, n_trials // n_hpo_repeats)
    base_name = study_name or f"{name}_{task}"

    if verbose:
        repeat_info = f" × {n_hpo_repeats} repeats" if n_hpo_repeats > 1 else ""
        logger.info(
            f"Starting Optuna study for {name} | task={task} | "
            f"metric={resolved_metric} | {trials_per_repeat} trials{repeat_info}"
        )

    best_study: Optional[optuna.Study] = None
    best_repeat: int = 0

    for repeat in range(n_hpo_repeats):
        seed = random_state + repeat

        if verbose and n_hpo_repeats > 1:
            logger.info(f"  Repeat {repeat + 1}/{n_hpo_repeats} (seed={seed})")

        # Clone the pre-built splitter for this repeat's seed, skipping AHC.
        # Falls back to None (build from scratch) when no splitter was provided.
        repeat_splitter = (
            _splitter.clone_with_n_splits(n_splits, random_state=seed)
            if _splitter is not None
            else None
        )

        objective = _build_objective(
            entry=entry,
            X=X,
            y=y,
            task=task,
            n_splits=n_splits,
            random_state=seed,
            score_fn=score_fn,
            coords=coords,
            n_blocks_per_fold=n_blocks_per_fold,
            spatial_cv_method=spatial_cv_method,
            ahc_threshold=ahc_threshold,
            spatial_cv_metric=spatial_cv_metric,
            pca_components=pca_components,
            exact_max_samples=exact_max_samples,
            knn_neighbors=knn_neighbors,
            fixed_params=fixed_params,
            y_true=y_true,
            inverse_fn=inverse_fn,
            _splitter=repeat_splitter,
        )

        sampler = optuna.samplers.TPESampler(seed=seed)
        # Prune trials whose running fold-mean falls below the median of completed
        # trials at the same fold step. n_startup_trials=10 lets TPE build a prior
        # before pruning kicks in; n_warmup_steps=1 waits for at least 2 folds.
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
        repeat_name = base_name if n_hpo_repeats == 1 else f"{base_name}_r{repeat}"
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=repeat_name,
        )

        # Use n_jobs=1 to avoid joblib loky memmapping temp-folder leaks on Windows.
        # Parallel Optuna trials (n_jobs=-1) spawn loky worker processes that clean up
        # their own temp dirs before the resource tracker can deregister them, causing
        # benign but noisy "leaked folder" warnings at shutdown.
        if verbose:
            cb, pbar = _make_trial_callback(resolved_metric, trials_per_repeat)
            callbacks = [cb]
        else:
            callbacks, pbar = [], None

        try:
            study.optimize(
                objective,
                n_trials=trials_per_repeat,
                show_progress_bar=False,
                n_jobs=1,
                callbacks=callbacks,
            )
        finally:
            if pbar is not None:
                pbar.close()

        if best_study is None or study.best_value > best_study.best_value:
            best_study = study
            best_repeat = repeat

    assert best_study is not None  # loop always runs at least once (n_hpo_repeats >= 1)

    if verbose:
        best_display = _to_display(best_study.best_value, resolved_metric)
        if n_hpo_repeats > 1:
            logger.info(
                f"Best repeat: {best_repeat + 1}/{n_hpo_repeats} | "
                f"{resolved_metric}={best_display:.4f} | "
                f"params={best_study.best_params}"
            )
        else:
            logger.info(
                f"Best trial: {resolved_metric}={best_display:.4f} | "
                f"params={best_study.best_params}"
            )

    return best_study


def optimize_all(
    names: list[str],
    X: np.ndarray,
    y: np.ndarray,
    task: str = "classification",
    metric: Optional[str] = None,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = False,
) -> dict[str, optuna.Study]:
    """
    Run optimization studies for multiple models.

    Returns a dict keyed by model name.
    Models not found in the registry are skipped with a warning.

    Args:
        names:  List of model class names to optimize.
        Others: Same as run_study().

    Returns:
        Dict[model_name, optuna.Study]

    Example:
        >>> studies = optimize_all(
        ...     ["RandomForestClassifier", "LGBMClassifier"],
        ...     X, y, task="classification", n_trials=30
        ... )
        >>> for name, study in studies.items():
        ...     print(name, study.best_params)
    """
    studies: dict[str, optuna.Study] = {}

    for name in names:
        try:
            study = run_study(
                name=name,
                X=X,
                y=y,
                task=task,
                metric=metric,
                n_trials=n_trials,
                n_splits=n_splits,
                random_state=random_state,
                verbose=verbose,
            )
            studies[name] = study
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping '{name}': {e}")

    return studies
