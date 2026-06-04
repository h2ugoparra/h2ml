"""Backward-compatible re-export shim.

``PipelineData`` now lives in :mod:`h2ml.core.feature_store` (a dependency-free
foundational layer). This module re-exports it so existing
``from h2ml.features.feature_store import PipelineData`` imports keep working.
"""

from h2ml.core.feature_store import PipelineData  # noqa: F401
