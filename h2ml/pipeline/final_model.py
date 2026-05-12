"""
h2ml/pipeline/final_model.py

FinalModel — deployment artifact produced after a full pipeline run.

Responsibilities
----------------
- Fit the best model (with best_params) on the full training dataset
- Handle per-sample scaling when the model requires it
- Expose predict() / predict_proba() for inference on new data
- Save / load via joblib

Separation of concerns
-----------------------
PipelineResult  — evaluation artifact (CV metrics, SHAP, stage selection)
FinalModel      — deployment artifact (fitted estimator ready for prediction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from h2ml.pipeline.pipeline import PipelineResult

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from h2ml.pipeline.base import TaskType
from h2ml.preprocessing.transforms import INVERSE_TRANSFORMS


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


@dataclass
class ConformalCalibration:
    """
    Nonconformity scores from out-of-fold CV predictions, used to construct
    finite-sample coverage-guaranteed prediction intervals (regression) or
    prediction sets (classification).

    Scores are pre-sorted ascending. The threshold at level 1-alpha is the
    ceil((1-alpha)(n+1))/n quantile, which guarantees marginal coverage ≥ 1-alpha.

    Attributes
    ----------
    scores:    Sorted nonconformity scores from calibration folds.
    n:         Number of calibration samples (len(scores)).
    task_type: TaskType of the calibrated model.
    """

    scores: np.ndarray
    n: int
    task_type: TaskType

    def threshold(self, alpha: float) -> float:
        """
        Return the nonconformity threshold that gives ≥ 1-alpha coverage.

        Args:
            alpha: Miscoverage level (e.g. 0.10 for 90% coverage).
        """
        level = min(np.ceil((1 - alpha) * (self.n + 1)) / self.n, 1.0)
        return float(np.quantile(self.scores, level))


@dataclass
class FinalModel:
    """
    Fitted model ready for inference on new data.

    Attributes
    ----------
    estimator:        Sklearn-compatible estimator fitted on the full training set.
    feature_names:    Ordered list of features the model was trained on.
    task_type:        TaskType.CLASSIFICATION or TaskType.REGRESSION.
    requires_scaling: Whether StandardScaler was applied before fitting.
    scaler:           Fitted StandardScaler (None when requires_scaling is False).
    best_model_name:  Name of the model as registered in the h2ml registry.
    best_params:      Hyperparameters used for the final fit (None = defaults).

    Example
    -------
    >>> final = result.build_final_model()
    >>> final.predict(X_new_df)
    >>> final.save("models/final_model.pkl")
    >>> final = FinalModel.load("models/final_model.pkl")
    """

    estimator: Any
    feature_names: list[str]
    task_type: TaskType
    requires_scaling: bool = False
    scaler: Optional[Any] = None
    best_model_name: Optional[str] = None
    best_params: Optional[dict] = field(default=None)
    conformal: Optional[ConformalCalibration] = field(default=None)
    y_transform: Optional[str] = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict on new data.

        Args:
            X: DataFrame (columns are aligned by name) or ndarray
                (columns must match feature_names order).

        Returns:
            1-D array of predictions.
        """
        return self.estimator.predict(self._prepare(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Returns:
            Binary:     1-D array of positive-class probabilities.
            Multiclass: 2-D array of shape (n_samples, n_classes).
        """
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks.")
        proba = self.estimator.predict_proba(self._prepare(X))
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def predict_interval(
        self,
        X: pd.DataFrame | np.ndarray,
        alpha: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Conformal prediction interval for each sample (regression only).

        The interval is centred on the point estimate and has width 2q, where q
        is the conformal threshold calibrated from out-of-fold residuals.
        Coverage is guaranteed to be ≥ 1-alpha in expectation.

        Note: if the pipeline used a y-transform, the interval is in the
        transformed space. Apply the inverse transform to the bounds if needed.

        Args:
            X:     Input features (DataFrame or ndarray).
            alpha: Miscoverage level. Default 0.10 → 90% prediction intervals.

        Returns:
            (lower, upper) as a pair of 1-D arrays.
        """
        if self.task_type != TaskType.REGRESSION:
            raise ValueError("predict_interval is only available for regression tasks.")
        if self.conformal is None:
            raise ValueError(
                "No conformal calibration available. "
                "Rebuild FinalModel via result.build_final_model() after a full pipeline run."
            )
        y_hat = self.predict(X)
        q = self.conformal.threshold(alpha)
        return y_hat - q, y_hat + q

    def predict_set(
        self,
        X: pd.DataFrame | np.ndarray,
        alpha: float = 0.10,
    ) -> list[np.ndarray]:
        """
        Conformal prediction set for each sample (classification only).

        Each prediction set contains the classes that are plausible given the
        coverage guarantee. A singleton set means the model is confident; a
        larger set means it is uncertain.

        Coverage guarantee: the true label is in the prediction set with
        probability ≥ 1-alpha.

        Nonconformity score per class k: 1 - p_k. Class k is included in the
        set when its score does not exceed the calibrated threshold q.

        Args:
            X:     Input features (DataFrame or ndarray).
            alpha: Miscoverage level. Default 0.10 → 90% coverage sets.

        Returns:
            List of arrays of class labels, one per sample.
        """
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_set is only available for classification tasks.")
        if self.conformal is None:
            raise ValueError(
                "No conformal calibration available. "
                "Rebuild FinalModel via result.build_final_model() after a full pipeline run."
            )
        p = self.predict_proba(X)  # 1-D for binary, 2-D for multiclass
        q = self.conformal.threshold(alpha)
        classes = self.estimator.classes_
        sets = []
        if p.ndim == 1:
            # Binary: p is positive-class probability; nonconformity for class 0 = p, for class 1 = 1-p
            for pi in p:
                labels = []
                if pi <= q:
                    labels.append(classes[0])
                if 1 - pi <= q:
                    labels.append(classes[1])
                sets.append(np.array(labels))
        else:
            # Multiclass: nonconformity for class k = 1 - p_k
            for row in p:
                labels = [classes[k] for k in range(len(classes)) if 1 - row[k] <= q]
                sets.append(np.array(labels))
        return sets

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist to *path* (single .pkl file via joblib). Parent directory is created if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FinalModel":
        """
        Reload a FinalModel saved with :meth:`save`.

        WARNING: only load files from trusted sources — joblib uses pickle,
        which executes arbitrary code on deserialisation.
        """
        return joblib.load(Path(path))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Align columns and apply scaling if required."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].to_numpy()
        X = np.asarray(X)
        if self.requires_scaling and self.scaler is not None:
            X = self.scaler.transform(X)
        return X  # type: ignore

    def __repr__(self) -> str:
        return (
            f"FinalModel("
            f"model={self.best_model_name!r}, "
            f"n_features={len(self.feature_names)}, "
            f"task={self.task_type.value!r})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_conformal_calibration(result, classes=None) -> Optional[ConformalCalibration]:
    """
    Build a ConformalCalibration from the out-of-fold predictions in best_cv_result.

    Regression:          nonconformity score = |y_test - y_pred_test|
    Binary classification:   nonconformity score = 1 - p(true_class)
    Multiclass classification: nonconformity score = 1 - p(true_class),
        looked up via the classes array from estimator.classes_.

    Returns None when no CV result is available or when the class mapping
    cannot be resolved for multiclass folds.
    """
    from loguru import logger

    cv = result.best_cv_result
    if cv is None or not cv.folds:
        return None

    task_type = cv.task_type

    if task_type == TaskType.REGRESSION:
        scores = np.concatenate([np.abs(f.y_test - f.y_pred_test) for f in cv.folds])
    else:
        sample_scores = []
        for f in cv.folds:
            if f.y_prob_test.ndim == 1:
                # Binary: p is positive-class probability
                # nonconformity = 1 - p(true_class)
                sample_scores.append(np.where(f.y_test == 1, 1.0 - f.y_prob_test, f.y_prob_test))
            else:
                # Multiclass: map each true label to its column index via classes_
                if classes is None:
                    logger.warning(
                        "Conformal calibration skipped: multiclass folds require "
                        "estimator.classes_ to map labels to probability columns."
                    )
                    return None
                label_to_idx = {c: i for i, c in enumerate(classes)}
                try:
                    col_idx = np.array([label_to_idx[label] for label in f.y_test])
                except KeyError as e:
                    logger.warning(
                        f"Conformal calibration skipped: y_test contains label {e} not found in estimator.classes_."
                    )
                    return None
                p_true = f.y_prob_test[np.arange(len(col_idx)), col_idx]
                sample_scores.append(1.0 - p_true)
        scores = np.concatenate(sample_scores)

    return ConformalCalibration(
        scores=np.sort(scores),
        n=len(scores),
        task_type=task_type,
    )


def build_final_model(result: "PipelineResult") -> FinalModel:
    """
    Fit the overall best model on the full training dataset and return a
    FinalModel ready for inference.

    Picks the correct feature store (reduced or full) via best_feature_stage,
    applies StandardScaler when the model requires it, and fits with
    best_params when step 4 ran.

    Args:
        result: PipelineResult from H2MLPipeline.run().

    Returns:
        FinalModel instance.
    """
    from h2ml.utils.registry import CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY

    task_type = result.step1_cv_result[0].task_type

    feature_stage = result.best_feature_stage or result.best_stage
    store = result.features_reduced if feature_stage == "reduced" else result.features

    registry = CLASSIFIER_REGISTRY if task_type == TaskType.CLASSIFICATION else REGRESSOR_REGISTRY
    entry = registry.get(result.best_model_name)
    if entry is None:
        raise ValueError(
            f"Model '{result.best_model_name}' not found in registry. Was it removed after the pipeline ran?"
        )

    # Instantiate with best_params when available, otherwise default
    if result.best_params:
        estimator = entry.model_cls(**result.best_params)
    else:
        estimator = entry.model_cls(**entry.default_kwargs)

    # Scale if required — fit scaler on the full training set
    X = store.X
    scaler = None
    if entry.requires_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    estimator.fit(X, store.y)

    classes = getattr(estimator, "classes_", None)

    return FinalModel(
        estimator=estimator,
        feature_names=store.feature_names,
        task_type=task_type,
        requires_scaling=entry.requires_scaling,
        scaler=scaler,
        best_model_name=result.best_model_name,
        best_params=result.best_params,
        conformal=_build_conformal_calibration(result, classes=classes),
        y_transform=result.y_transform,
    )


# ---------------------------------------------------------------------------
# Delta model — two-component presence × abundance
# ---------------------------------------------------------------------------


def _build_delta_conformal(
    clf_result: "PipelineResult",
    reg_result: "PipelineResult",
    reg_final: FinalModel,
    X_full: "pd.DataFrame | np.ndarray",
    y_full: np.ndarray,
    positive_indices: np.ndarray,
) -> Optional[ConformalCalibration]:
    """
    Build a ConformalCalibration for the combined delta output using OOF predictions.

    P_oof (all N samples) comes from clf_result.best_cv_result.
    count_oof_pos (positive samples, already inverse-transformed by CV) comes from
    reg_result.best_cv_result. For zero-count samples the regressor is called as a
    regular predictor (not OOF) and the inverse transform is applied manually.

    Pass X_full as a DataFrame (with all feature column names) when the clf and reg
    were trained on different feature sets — reg_final.predict selects its own columns.

    Nonconformity score: |y_full - (P_oof × count_hat)|
    """
    from loguru import logger

    clf_cv = clf_result.best_cv_result
    reg_cv = reg_result.best_cv_result

    if clf_cv is None or not clf_cv.folds:
        logger.warning("Delta conformal calibration skipped: no classifier CV folds available.")
        return None
    if reg_cv is None or not reg_cv.folds:
        logger.warning("Delta conformal calibration skipped: no regressor CV folds available.")
        return None

    n_full = len(y_full)

    P_oof = clf_cv.oof_predictions  # shape (n_full,) — 1D probabilities for binary
    if P_oof is None:
        logger.warning("Delta conformal calibration skipped: classifier OOF predictions unavailable.")
        return None
    if len(P_oof) != n_full:
        logger.warning(
            f"Delta conformal calibration skipped: classifier OOF length {len(P_oof)} "
            f"does not match y_full length {n_full}. Pass the same X/y the classifier was trained on."
        )
        return None

    # count_oof_pos: shape (n_pos,) — regressor OOF predictions, already inverse-transformed
    # (CV applies inverse_fn to y_pred_test before storing in FoldResult)
    count_oof_pos = reg_cv.oof_predictions
    if count_oof_pos is None:
        logger.warning("Delta conformal calibration skipped: regressor OOF predictions unavailable.")
        return None

    count_hat_full = np.full(n_full, np.nan)
    count_hat_full[positive_indices] = count_oof_pos

    # For zero-count samples, call the final regressor (not OOF) and inverse-transform manually
    zero_mask = np.ones(n_full, dtype=bool)
    zero_mask[positive_indices] = False
    zero_indices = np.where(zero_mask)[0]
    if zero_indices.size > 0:
        # Slice preserving type: DataFrame slicing keeps column names so _prepare
        # can select the reg's feature subset when clf and reg have different features.
        X_zero = X_full.iloc[zero_indices] if isinstance(X_full, pd.DataFrame) else X_full[zero_indices]
        zero_preds = reg_final.predict(X_zero)  # transform space
        if reg_final.y_transform is not None:
            inverse_fn = INVERSE_TRANSFORMS.get(reg_final.y_transform)
            if inverse_fn is not None:
                zero_preds = inverse_fn(zero_preds)
        count_hat_full[zero_indices] = zero_preds

    delta_oof = P_oof * count_hat_full
    valid = np.isfinite(delta_oof)
    n_excluded = int((~valid).sum())
    if n_excluded:
        logger.warning(
            f"Delta conformal calibration: {n_excluded} samples excluded due to NaN (likely from failed CV folds)."
        )

    scores = np.abs(y_full[valid] - delta_oof[valid])
    return ConformalCalibration(
        scores=np.sort(scores),
        n=int(valid.sum()),
        task_type=TaskType.REGRESSION,
    )


@dataclass
class DeltaFinalModel:
    """
    Two-component delta model combining a presence/absence classifier and a
    count/abundance regressor.

    Prediction: P(present) × E(count | present)
    Conformal interval: [max(0, delta − q), delta + q]
    where q is calibrated from combined out-of-fold residuals on the full delta output.

    The regressor's y-transform (if any) is inverted inside predict() so the output is
    always in the original count scale. This differs from FinalModel, where the caller
    handles inversion — the multiplication P × count requires the original scale.

    Example
    -------
    >>> positive_idx = np.where(y_all > 0)[0]
    >>> delta = build_delta_final_model(clf_result, reg_result, X_all, y_all, positive_idx)
    >>> delta.save("models/sparrow_delta-model")
    >>> delta = DeltaFinalModel.load("models/sparrow_delta-model")
    >>> lower, upper = delta.predict_interval(X_new, alpha=0.10)
    """

    clf: FinalModel
    reg: FinalModel
    conformal: Optional[ConformalCalibration] = field(default=None)

    def predict(self, X: "pd.DataFrame | np.ndarray") -> np.ndarray:
        """
        Delta prediction: P(present) × E(count | present).

        The regressor's y-transform is inverted here so the result is in the
        original count scale. Pass a DataFrame to let each sub-model select its own
        features by name; pass an ndarray only when both models share the same
        feature set and column order.
        """
        p = self.clf.predict_proba(X)
        count = self.reg.predict(X)
        if self.reg.y_transform is not None:
            inverse_fn = INVERSE_TRANSFORMS.get(self.reg.y_transform)
            if inverse_fn is not None:
                count = inverse_fn(count)
        return p * count

    def predict_interval(
        self,
        X: "pd.DataFrame | np.ndarray",
        alpha: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Conformal prediction interval for each sample (count scale).

        Coverage is guaranteed for the full delta output, not each component
        separately. The lower bound is clipped at zero (counts are non-negative).

        Args:
            X:     Input features.
            alpha: Miscoverage level. Default 0.10 → 90% coverage.

        Returns:
            (lower, upper) pair of 1-D arrays.
        """
        if self.conformal is None:
            raise ValueError(
                "No conformal calibration available. "
                "Rebuild via build_delta_final_model() with positive_indices provided."
            )
        delta = self.predict(X)
        q = self.conformal.threshold(alpha)
        return np.maximum(0.0, delta - q), delta + q

    def save(self, path: "str | Path") -> None:
        """
        Persist to *path* (directory). Creates the directory if needed.
        Saves clf.pkl, reg.pkl, and (when calibrated) conformal.pkl.
        Reload with DeltaFinalModel.load(path).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path / "clf.pkl")
        joblib.dump(self.reg, path / "reg.pkl")
        if self.conformal is not None:
            joblib.dump(self.conformal, path / "conformal.pkl")

    @classmethod
    def load(cls, path: "str | Path") -> "DeltaFinalModel":
        """
        Reload a DeltaFinalModel saved with :meth:`save`.

        WARNING: only load files from trusted sources — joblib uses pickle,
        which executes arbitrary code on deserialisation.
        """
        path = Path(path)
        clf = joblib.load(path / "clf.pkl")
        reg = joblib.load(path / "reg.pkl")
        conformal_path = path / "conformal.pkl"
        conformal = joblib.load(conformal_path) if conformal_path.exists() else None
        return cls(clf=clf, reg=reg, conformal=conformal)

    def __repr__(self) -> str:
        return (
            f"DeltaFinalModel("
            f"clf={self.clf.best_model_name!r}, "
            f"reg={self.reg.best_model_name!r}, "
            f"calibrated={self.conformal is not None})"
        )


def build_delta_final_model(
    clf_result: "PipelineResult",
    reg_result: "PipelineResult",
    X_full: "pd.DataFrame | np.ndarray",
    y_full: np.ndarray,
    positive_indices: np.ndarray,
) -> DeltaFinalModel:
    """
    Fit a DeltaFinalModel from a classifier and a regressor PipelineResult.

    Builds a clf FinalModel (all N samples) and a reg FinalModel (positive samples),
    then constructs a ConformalCalibration using combined OOF delta predictions
    following Option B — combined residual conformalization on the full output.

    Args:
        clf_result:       PipelineResult from the presence/absence classifier,
                          trained on all N samples.
        reg_result:       PipelineResult from the count regressor, trained on
                          positive-count samples (X_full[positive_indices]).
        X_full:           Feature matrix for ALL N samples. Pass a pandas DataFrame
                          (with all feature column names) when clf and reg have different
                          feature sets — the regressor will select its own columns by name.
        y_full:           Raw count targets for ALL N samples (0 for absence,
                          positive integer for presence).
        positive_indices: Integer indices into X_full / y_full identifying the
                          positive-count rows (i.e. those the regressor was trained on).

    Returns:
        DeltaFinalModel with clf, reg, and a combined ConformalCalibration.

    Example
    -------
    >>> positive_idx = np.where(y_all > 0)[0]
    >>> X_df = pd.DataFrame(X_all, columns=feature_names)
    >>> delta = build_delta_final_model(clf_result, reg_result, X_df, y_all, positive_idx)
    >>> delta.save("models/sparrow_delta-model")
    """
    clf_final = build_final_model(clf_result)
    reg_final = build_final_model(reg_result)
    conformal = _build_delta_conformal(clf_result, reg_result, reg_final, X_full, y_full, positive_indices)
    return DeltaFinalModel(clf=clf_final, reg=reg_final, conformal=conformal)
