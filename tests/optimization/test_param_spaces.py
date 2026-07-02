"""
tests/optimization/test_param_spaces.py

Consistency guards for the Optuna param-space functions in core/param_spaces.

Each param_fn returns a dict passed to model_cls(**params). A typo'd dict key
(e.g. "subssample") silently detaches the sampled value from the estimator for
permissive wrappers like LGBM, so HPO explores a dead dimension while the final
refit uses the real key from study.best_params.
"""

from __future__ import annotations

import pytest

from h2ml.utils.registry import CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY


class _StubTrial:
    """Minimal optuna.Trial stand-in: returns each range's low / first choice
    and records every suggested parameter name."""

    def __init__(self):
        self.suggested: list[str] = []

    def suggest_int(self, name, low, high, step=1, log=False):
        self.suggested.append(name)
        return low

    def suggest_float(self, name, low, high, step=None, log=False):
        self.suggested.append(name)
        return low

    def suggest_categorical(self, name, choices):
        self.suggested.append(name)
        return choices[0]


def _optimizable_entries(registry):
    return [(name, entry) for name, entry in registry.items() if entry.param_fn is not None]


_ALL_ENTRIES = _optimizable_entries(CLASSIFIER_REGISTRY) + _optimizable_entries(REGRESSOR_REGISTRY)


@pytest.mark.parametrize("name,entry", _ALL_ENTRIES, ids=[n for n, _ in _ALL_ENTRIES])
class TestParamSpaceConsistency:
    def test_every_suggested_name_is_a_returned_key(self, name, entry):
        """A suggest_* name missing from the returned dict means the sampled
        value never reaches the estimator (key/name drift, e.g. 'subssample')."""
        trial = _StubTrial()
        params = entry.param_fn(trial)
        missing = [s for s in trial.suggested if s not in params]
        assert not missing, f"{name}: suggested params {missing} are not keys of the returned dict"

    def test_estimator_accepts_returned_params(self, name, entry):
        """Sklearn estimators reject unknown constructor kwargs — instantiating
        with the emitted params catches misspelled keys for strict estimators."""
        trial = _StubTrial()
        params = entry.param_fn(trial)
        entry.model_cls(**params)
