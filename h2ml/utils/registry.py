"""
h2ml/utils/registry.py

ModelEntry — unified descriptor for every model in the h2ml ecosystem.

Each entry encodes everything needed to build, scale, and optimize a model:
    - model_cls:        The sklearn estimator class.
    - default_kwargs:   Constructor arguments used for the default (non-optimized) step.
    - requires_scaling: Whether StandardScaler must be applied before fitting.
    - param_fn:         Optuna trial function → hyperparameter dict (None = disabled).
    - opt_enabled:      Set False to exclude from optimization even if param_fn exists.

CLASSIFIER_REGISTRY and REGRESSOR_REGISTRY are the single source of truth consumed
by both the pipeline (build_steps) and the optimizer (get_entry via opt_params.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from loguru import logger
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from h2ml.core.param_spaces import classifiers as cp
from h2ml.core.param_spaces import regressors as rp
from h2ml.core.step import ModelWrapper

# ---------------------------------------------------------------------------
# ModelEntry
# ---------------------------------------------------------------------------


@dataclass
class ModelEntry:
    """
    Unified descriptor for a single model in the h2ml registry.

    Attributes:
        model_cls:        Sklearn estimator class.
        default_kwargs:   Constructor args for the default (non-optimized) instance.
        requires_scaling: Whether StandardScaler must be applied before fitting.
        param_fn:         Optuna trial function returning a hyperparameter dict.
                          None means optimization is disabled for this model.
        opt_enabled:      Set False to exclude from optimization even if param_fn exists.
        supports_class_weight: Whether the estimator accepts class_weight="balanced".
                          When True, PipelineConfig.handle_imbalance can inject it.
    """

    model_cls: type
    default_kwargs: dict = field(default_factory=dict)
    requires_scaling: bool = False
    param_fn: Optional[Callable] = None
    opt_enabled: bool = True
    supports_class_weight: bool = False

    def __post_init__(self) -> None:
        if self.opt_enabled and self.param_fn is None:
            raise ValueError(
                f"ModelEntry for {self.model_cls.__name__} has opt_enabled=True "
                "but param_fn=None. Provide a param_fn or set opt_enabled=False."
            )

    def build_model(self) -> ModelWrapper:
        """Instantiate a ModelWrapper with default kwargs and scaling flag."""
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
    from lightgbm import LGBMRegressor  # type: ignore[import-not-found]

    REGRESSOR_REGISTRY["LGBMRegressor"] = ModelEntry(
        LGBMRegressor,
        default_kwargs={"random_state": 42, "verbose": -1},
        param_fn=rp.lightgbm_r_params,
    )
except ImportError:
    pass

try:
    from catboost import CatBoostRegressor  # type: ignore[import-not-found]

    REGRESSOR_REGISTRY["CatBoostRegressor"] = ModelEntry(
        CatBoostRegressor,
        default_kwargs={"random_seed": 42, "silent": True, "thread_count": 1},
        param_fn=rp.catboost_r_params,
    )
except ImportError:
    pass

try:
    from xgboost import XGBRegressor  # type: ignore[import-not-found]

    REGRESSOR_REGISTRY["XGBRegressor"] = ModelEntry(
        XGBRegressor,
        default_kwargs={"random_state": 42},
        param_fn=rp.xgboost_r_params,
    )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

# Maps model name → a human-readable provenance string, for models whose behaviour
# is not pinned by this repo's lockfile alone. Every sklearn/boosting model is fully
# determined by its pinned package version, so they are absent here. TabPFN is not:
# it runs pretrained weights downloaded from a repo the vendor controls, so a run is
# only reproducible if you also know which weights it used.
MODEL_PROVENANCE: dict[str, str] = {}


def collect_model_provenance(model_names) -> dict[str, str]:
    """Return provenance entries for the given model names (empty when none apply)."""
    names = set(model_names)
    return {name: info for name, info in MODEL_PROVENANCE.items() if name in names}


# ---------------------------------------------------------------------------
# TabPFN (opt-in)
# ---------------------------------------------------------------------------


def _record_tabpfn_provenance() -> None:
    """
    Record which TabPFN weights a run will use, into MODEL_PROVENANCE.

    Unlike every other registry model, TabPFN's predictions are not pinned by
    uv.lock: `model_path="auto"` resolves to a checkpoint downloaded from a repo
    PriorLabs controls. The package version fixes which filename "auto" selects
    (the lists are hardcoded in tabpfn.model_loading), and model_version fixes the
    generation, so the pair is what makes a run auditable after the fact.

    Best-effort: this reads a vendor setting, so any change there degrades to a
    partial record rather than breaking registration.
    """
    import tabpfn  # type: ignore[import-not-found]

    info = f"tabpfn=={getattr(tabpfn, '__version__', 'unknown')}"
    try:
        from tabpfn.settings import settings  # type: ignore[import-not-found]

        version = settings.tabpfn.model_version
        info += f", model_version={getattr(version, 'value', version)}"
    except Exception:  # pragma: no cover - vendor internals may move
        info += ", model_version=unknown"

    MODEL_PROVENANCE["TabPFNClassifier"] = info
    MODEL_PROVENANCE["TabPFNRegressor"] = info


def _warn_if_tabpfn_cpu() -> None:
    """
    Warn when TabPFN will fall back to CPU inference.

    TabPFN's device="auto" resolves in the order: CUDA GPUs → mps → cpu. We mirror
    that priority so the warning only fires when it will genuinely run on CPU, where
    inference is slow enough that ~5 000 samples is a practical ceiling.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - tabpfn depends on torch
        return

    mps = getattr(torch.backends, "mps", None)
    if torch.cuda.is_available() or (mps is not None and mps.is_available()):
        return

    build = getattr(torch, "__version__", "")
    hint = ""
    if "+cpu" in build or torch.version.cuda is None:
        # The default PyPI wheel is CPU-only on Windows even with an NVIDIA card present;
        # CUDA builds come from download.pytorch.org. Say so, or users hunt a driver bug.
        hint = (
            f" torch is a CPU-only build ({build}) — an installed NVIDIA GPU will not be used "
            "unless you install a CUDA wheel from https://download.pytorch.org/whl."
        )

    logger.warning(
        "TabPFN registered but no CUDA/mps device is visible — inference will run on CPU. "
        "Expect it to be slow; ~5 000 samples is a practical ceiling. Larger inputs raise "
        "TabPFN's own pretraining-limit error, which the CV engine reports as a skipped model." + hint
    )


# Opt-in via H2ML_ENABLE_TABPFN=1: the checkpoint is large, CPU inference is slow, and the
# first fit needs PriorLabs authentication (browser login, or TABPFN_TOKEN when headless).
# Gating on an env var keeps installing the extra from silently changing every pipeline run.
if os.environ.get("H2ML_ENABLE_TABPFN") == "1":
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore[import-not-found]

        # No device kwarg: TabPFN's default device="auto" already picks CUDA → mps → cpu.
        # opt_enabled=False — TabPFN is pretrained; its knobs are inference-time settings,
        # not a search space, so it must never reach the step-4 Optuna study.
        CLASSIFIER_REGISTRY["TabPFNClassifier"] = ModelEntry(
            TabPFNClassifier,
            default_kwargs={"random_state": 42},
            param_fn=None,
            opt_enabled=False,
        )
        REGRESSOR_REGISTRY["TabPFNRegressor"] = ModelEntry(
            TabPFNRegressor,
            default_kwargs={"random_state": 42},
            param_fn=None,
            opt_enabled=False,
        )
        _record_tabpfn_provenance()
        _warn_if_tabpfn_cpu()
    except ImportError:
        logger.warning("H2ML_ENABLE_TABPFN=1 but tabpfn is not installed — run: uv sync --extra tabpfn")


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
