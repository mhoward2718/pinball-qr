"""Pinball — fast quantile regression for Python.

Provides an sklearn-compatible :class:`QuantileRegressor` backed by
high-performance Fortran solvers (Barrodale-Roberts simplex, Frisch-Newton
interior point) ported from R's ``quantreg`` package.

Quick start::

    from pinball import QuantileRegressor
    model = QuantileRegressor(tau=0.5, method="fn")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
"""

__author__ = """Michael Howard"""
__email__ = "mah38900@gmail.com"
__version__ = "0.1.0"

from pinball._estimator import QuantileRegressor
from pinball._inference import summary, InferenceResult
from pinball._bootstrap import bootstrap, BootstrapResult
from pinball.datasets import load_engel
from pinball.solvers import get_solver, list_solvers, register_solver
from pinball.solvers.base import BaseSolver, SolverResult

__all__ = [
    "QuantileRegressor",
    "BaseSolver",
    "SolverResult",
    "InferenceResult",
    "BootstrapResult",
    "summary",
    "bootstrap",
    "get_solver",
    "list_solvers",
    "register_solver",
    "load_engel",
]
