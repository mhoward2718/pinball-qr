"""Solver registry — maps method names to solver classes.

The registry implements the **Open / Closed Principle**: adding a new solver
means calling :func:`register_solver`; no existing code needs to change.

Usage
-----
>>> from pinball.linear.solvers import get_solver
>>> solver = get_solver("fn")      # returns an FNBSolver instance
>>> result = solver.solve(X, y, tau=0.5)
"""

from __future__ import annotations

from pinball.linear.solvers.base import BaseSolver, SolverResult

__all__ = [
    "BaseSolver",
    "SolverResult",
    "get_solver",
    "register_solver",
    "list_solvers",
]

# ──────────────────────────────────────────────────────────────────────
# Private registry
# ──────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[BaseSolver]] = {}


# ──────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────

def register_solver(name: str, cls: type[BaseSolver]) -> None:
    """Register a solver class under *name*.

    Parameters
    ----------
    name : str
        Short method name (e.g. ``"br"``, ``"fn"``).
    cls : type
        A concrete subclass of :class:`BaseSolver`.

    Raises
    ------
    TypeError
        If *cls* is not a subclass of ``BaseSolver``.
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseSolver)):
        raise TypeError(f"{cls!r} is not a BaseSolver subclass.")
    _REGISTRY[name] = cls


def get_solver(name: str, **kwargs) -> BaseSolver:
    """Return an **instance** of the solver registered under *name*.

    Parameters
    ----------
    name : str
        A key previously passed to :func:`register_solver`.
    **kwargs
        Forwarded to the solver constructor.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown solver {name!r}. Available solvers: {available}"
        ) from None
    return cls(**kwargs)


def list_solvers() -> list[str]:
    """Return the names of all registered solvers."""
    return sorted(_REGISTRY)


# ──────────────────────────────────────────────────────────────────────
# Eagerly register built-in solvers (imported at bottom to avoid cycles)
# ──────────────────────────────────────────────────────────────────────

def _register_builtins() -> None:
    from pinball.linear.solvers.br import BRSolver
    from pinball.linear.solvers.fnb import FNBSolver

    register_solver("br", BRSolver)
    register_solver("fn", FNBSolver)
    register_solver("fnb", FNBSolver)

    # Lazy imports for optional solvers — they depend on FNB internally
    try:
        from pinball.linear.solvers.lasso import LassoSolver
        register_solver("lasso", LassoSolver)
    except ImportError:
        pass

    try:
        from pinball.linear.solvers.pfn import PreprocessingSolver
        register_solver("pfn", PreprocessingSolver)
    except ImportError:
        pass

    # POGS ADMM solver — uses vendored native C++ library
    from pinball.linear.solvers.pogs import POGSSolver
    register_solver("pogs", POGSSolver)


_register_builtins()
