"""Backward-compatible re-export shim.

The estimator-wrapping step types now live in :mod:`h2ml.core.step` (a
dependency-free foundational layer). This module re-exports them so existing
``from h2ml.pipeline.step import ...`` imports keep working.
"""

from h2ml.core.step import (  # noqa: F401
    ModelWrapper,
    make_classifier,
    make_preprocessor,
    make_regressor,
)
