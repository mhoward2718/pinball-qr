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

from pinball.linear._estimator import QuantileRegressor
from pinball.linear._inference import summary, InferenceResult
from pinball.linear._bootstrap import bootstrap, BootstrapResult
from pinball.datasets import load_engel
from pinball.linear.solvers import get_solver, list_solvers, register_solver
from pinball.linear.solvers.base import BaseSolver, SolverResult
from pinball.estimators._base import BaseQuantileEstimator

__all__ = [
    "BaseQuantileEstimator",
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
