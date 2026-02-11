"""Backward-compatibility shim — re-exports from ``pinball.linear.solvers``.

All solver classes have moved to :mod:`pinball.linear.solvers`.
This module re-exports the public API so that existing code using
``from pinball.solvers import get_solver`` continues to work.
"""

from pinball.linear.solvers import (  # noqa: F401
    BaseSolver,
    SolverResult,
    get_solver,
    list_solvers,
    register_solver,
)

__all__ = [
    "BaseSolver",
    "SolverResult",
    "get_solver",
    "list_solvers",
    "register_solver",
]
