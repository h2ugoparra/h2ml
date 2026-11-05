"""
tests/geo/test_h2mare_dependency.py

h2mare keeps its plotting stack behind a `viz` extra. predict_map imports it at
call time, so a dependency on plain `h2mare` only fails once a user draws a map.
"""

from __future__ import annotations


def test_base_install_provides_predict_map_imports():
    from h2mare.storage import aggregate_by_space_time
    from h2mare.utils.plot import plot_maps

    assert callable(aggregate_by_space_time)
    assert callable(plot_maps)
