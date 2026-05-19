"""
h2ml/utils/registry.py

ModelEntry — unified descriptor for every model in the h2ml ecosystem.

Each entry encodes everything needed to build, scale, and optimize a model:
    - model_cls:        The sklearn estimator class.
    - default_kwargs:   Constructor arguments used for the default (non-optimized) step.
    - requires_scaling: Whether StandardScaler must be applied before fitting.
    - param_fn:         Optuna trial function → hyperparameter dict (None = disabled).
    - opt_enabled:      Set False to exclude from optimization even if param_fn exists.
    - single_njob:      Force n_jobs=1 in Optuna to avoid SQLite concurrency issues.

CLASSIFIER_REGISTRY and REGRESSOR_REGISTRY are the single source of truth consumed
by both the pipeline (build_steps) and the optimizer (get_entry via opt_params.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from h2ml.pipeline.step import ModelWrapper

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
)
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

from h2ml.optimization.params import classifiers as cp
from h2ml.optimization.params import regressors as rp


# ---------------------------------------------------------------------------
# ModelEntry
# ---------------------------------------------------------------------------


@dataclass
class ModelEntry:
    """
    Unified descriptor for a single model in the h2ml registry.

    Args:
        model_cls:        Sklearn estimator class.
        default_kwargs:   Constructor args for the default (non-optimized) instance.
        requires_scaling: Whether StandardScaler must be applied before fitting.
        param_fn:         Optuna trial function returning a hyperparameter dict.
                          None means optimization is disabled for this model.
        opt_enabled:      Set False to exclude from optimization even if param_fn exists.
        single_njob:      Force n_jobs=1 in Optuna to avoid SQLite concurrency issues.
    """

    model_cls: type
    default_kwargs: dict = field(default_factory=dict)
    requires_scaling: bool = False
    param_fn: Optional[Callable] = None
    opt_enabled: bool = True
    single_njob: bool = False
    supports_class_weight: bool = False

    def __post_init__(self) -> None:
        if self.opt_enabled and self.param_fn is None:
            raise ValueError(
                f"ModelEntry for {self.model_cls.__name__} has opt_enabled=True "
                "but param_fn=None. Provide a param_fn or set opt_enabled=False."
            )

    def build_model(self) -> "ModelWrapper":
        """Instantiate a ModelWrapper with default kwargs and scaling flag."""
        from h2ml.pipeline.step import ModelWrapper

        return ModelWrapper(
            estimator=self.model_cls(**self.default_kwargs),
            requires_scaling=self.requires_scaling,
        )


# ---------------------------------------------------------------------------
# Classifier registry
# ---------------------------------------------------------------------------

CLASSIFIER_REGISTRY: dict[str, ModelEntry] = {
    "LogisticRegression": ModelEntry(
        LogisticRegression,
        default_kwargs={"random_state": 42},
        requires_scaling=True,
        param_fn=None,
        opt_enabled=False,
        supports_class_weight=True,
    ),
    "GaussianNB": ModelEntry(
        GaussianNB,
        param_fn=None,
        opt_enabled=False,
    ),
    "KNeighborsClassifier": ModelEntry(
        KNeighborsClassifier,
        default_kwargs={"weights": "distance"},
        requires_scaling=True,
        param_fn=None,
        opt_enabled=False,
    ),
    "RandomForestClassifier": ModelEntry(
        RandomForestClassifier,
        default_kwargs={"random_state": 42},
        param_fn=cp.randomforest_c_params,
        supports_class_weight=True,
    ),
    "GradientBoostingClassifier": ModelEntry(
        GradientBoostingClassifier,
        default_kwargs={"random_state": 42},
        param_fn=cp.gradientboosting_c_params,
    ),
    "HistGradientBoostingClassifier": ModelEntry(
        HistGradientBoostingClassifier,
        default_kwargs={"random_state": 42},
        param_fn=cp.histgradientboosting_c_params,
        supports_class_weight=True,
    ),
    "SVC": ModelEntry(
        SVC,
        default_kwargs={"random_state": 42, "probability": True},
        requires_scaling=True,
        param_fn=cp.svc_c_params,
        supports_class_weight=True,
    ),
    "ExtraTreesClassifier": ModelEntry(
        ExtraTreesClassifier,
        default_kwargs={"random_state": 42},
        param_fn=cp.extratrees_c_params,
        supports_class_weight=True,
    ),
    "BaggingClassifier": ModelEntry(
        BaggingClassifier,
        default_kwargs={"random_state": 42},
        param_fn=None,
        opt_enabled=False,
    ),
    "AdaBoostClassifier": ModelEntry(
        AdaBoostClassifier,
        default_kwargs={"random_state": 42},
        param_fn=None,
        opt_enabled=False,
    ),
}

# Optional heavy dependencies
try:
    from lightgbm import LGBMClassifier  # type: ignore[import-not-found]

    CLASSIFIER_REGISTRY["LGBMClassifier"] = ModelEntry(
        LGBMClassifier,
        default_kwargs={"random_state": 42, "verbose": -1},
        param_fn=cp.lightgbm_c_params,
        supports_class_weight=True,
    )
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier  # type: ignore[import-not-found]

    CLASSIFIER_REGISTRY["CatBoostClassifier"] = ModelEntry(
        CatBoostClassifier,
        default_kwargs={"random_seed": 42, "silent": True, "thread_count": 1},
        param_fn=cp.catboost_c_params,
    )
except ImportError:
    pass

try:
    from xgboost import XGBClassifier  # type: ignore[import-not-found]

    CLASSIFIER_REGISTRY["XGBClassifier"] = ModelEntry(
        XGBClassifier,
        default_kwargs={"random_state": 42},
        param_fn=cp.xgboost_c_params,
    )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Regressor registry
# ---------------------------------------------------------------------------

REGRESSOR_REGISTRY: dict[str, ModelEntry] = {
    "PoissonRegressor": ModelEntry(
        PoissonRegressor,
        requires_scaling=True,
        param_fn=None,
        opt_enabled=False,
    ),
    "KNeighborsRegressor": ModelEntry(
        KNeighborsRegressor,
        default_kwargs={"weights": "distance"},
        requires_scaling=True,
        param_fn=None,
        opt_enabled=False,
    ),
    "RandomForestRegressor": ModelEntry(
        RandomForestRegressor,
        default_kwargs={"random_state": 42},
        param_fn=rp.randomforest_r_params,
        single_njob=True,
    ),
    "GradientBoostingRegressor": ModelEntry(
        GradientBoostingRegressor,
        default_kwargs={"random_state": 42},
        param_fn=rp.gradientboosting_r_params,
    ),
    "HistGradientBoostingRegressor": ModelEntry(
        HistGradientBoostingRegressor,
        default_kwargs={"random_state": 42},
        param_fn=rp.histgradientboosting_r_params,
        single_njob=True,
    ),
    "SVR": ModelEntry(
        SVR,
        requires_scaling=True,
        param_fn=rp.svr_r_params,
    ),
    "ExtraTreesRegressor": ModelEntry(
        ExtraTreesRegressor,
        default_kwargs={"random_state": 42},
        param_fn=rp.extratrees_r_params,
        single_njob=True,
    ),
    "BaggingRegressor": ModelEntry(
        BaggingRegressor,
        default_kwargs={"random_state": 42},
        param_fn=None,
        opt_enabled=False,
    ),
    "AdaBoostRegressor": ModelEntry(
        AdaBoostRegressor,
        default_kwargs={"random_state": 42},
        param_fn=None,
        opt_enabled=False,
    ),
}

# Optional heavy dependencies
try:
    from lightgbm import LGBMRegressor # type: ignore[import-not-found]

    REGRESSOR_REGISTRY["LGBMRegressor"] = ModelEntry(
        LGBMRegressor,
        default_kwargs={"random_state": 42, "verbose": -1},
        param_fn=rp.lightgbm_r_params,
    )
except ImportError:
    pass

try:
    from catboost import CatBoostRegressor # type: ignore[import-not-found]

    REGRESSOR_REGISTRY["CatBoostRegressor"] = ModelEntry(
        CatBoostRegressor,
        default_kwargs={"random_seed": 42, "silent": True, "thread_count": 1},
        param_fn=rp.catboost_r_params,
    )
except ImportError:
    pass

try:
    from xgboost import XGBRegressor # type: ignore[import-not-found]

    REGRESSOR_REGISTRY["XGBRegressor"] = ModelEntry(
        XGBRegressor,
        default_kwargs={"random_state": 42},
        param_fn=rp.xgboost_r_params,
    )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_models(task: str) -> list["ModelWrapper"]:
    """
    Build a ModelWrapper list from the registry for a given task.

    Args:
        task: 'classification' or 'regression'.

    Returns:
        List of ModelWrapper instances with correct scaling flags and default kwargs.
    """
    registry = CLASSIFIER_REGISTRY if task == "classification" else REGRESSOR_REGISTRY
    return [entry.build_model() for entry in registry.values()]
