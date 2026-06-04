"""Backward-compatible re-export shim.

The base step abstractions now live in :mod:`h2ml.core.base` (a dependency-free
foundational layer). This module re-exports them so existing
``from h2ml.pipeline.base import ...`` imports keep working.
"""

from h2ml.core.base import (  # noqa: F401
    BaseClassifier,
    BasePreprocessor,
    BaseRegressor,
    BaseStep,
    PredictorMixin,
    PredictorStep,
    TaskType,
)
