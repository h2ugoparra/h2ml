"""
h2ml/optimization/opt_params.py

Thin lookup layer over the unified model registry.
Provides get_entry() for the optimizer to retrieve a ModelEntry by name and task.
"""

from __future__ import annotations

from h2ml.utils.registry import CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY, ModelEntry

# Re-export ModelEntry so optimizer.py can import it from here without change.
__all__ = ["ModelEntry", "get_entry"]


def get_entry(name: str, task: str) -> ModelEntry:
    """
    Retrieve a ModelEntry by model name and task type.

    Args:
        name: Model class name (e.g. 'RandomForestClassifier').
        task: 'classification' or 'regression'.

    Raises:
        KeyError:   If the model is not registered for the given task.
        ValueError: If the entry exists but has optimization disabled.
    """
    registry = CLASSIFIER_REGISTRY if task == "classification" else REGRESSOR_REGISTRY

    if name not in registry:
        available = [k for k, v in registry.items() if v.opt_enabled and v.param_fn is not None]
        raise KeyError(f"Model '{name}' not found in {task} registry. Available: {available}")

    entry = registry[name]
    if not entry.opt_enabled or entry.param_fn is None:
        raise ValueError(
            f"Model '{name}' has optimization disabled. "
            f"Set opt_enabled=True and provide a param_fn in utils/registry.py to activate it."
        )

    return entry
