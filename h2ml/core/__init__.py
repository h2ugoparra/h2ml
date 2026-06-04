"""Foundational, dependency-free types shared across h2ml.

This package sits at the bottom of the import graph: nothing in core imports
from other h2ml packages, so importing core can never create a cycle.
"""
