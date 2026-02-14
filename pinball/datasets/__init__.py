"""Bundled datasets for examples and testing."""

from pinball.datasets._barro import load_barro
from pinball.datasets._engel import load_engel

__all__ = ["load_engel", "load_barro"]
