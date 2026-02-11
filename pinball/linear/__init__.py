"""Linear (parametric) quantile regression subpackage.

Contains the :class:`QuantileRegressor` estimator, inference/bootstrap
modules, and the LP solver registry — all ported from R's ``quantreg``.
"""

from pinball.linear._estimator import QuantileRegressor
from pinball.linear._inference import summary, InferenceResult
from pinball.linear._bootstrap import bootstrap, BootstrapResult
from pinball.linear.solvers import get_solver, list_solvers, register_solver
from pinball.linear.solvers.base import BaseSolver, SolverResult

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
]
