"""POGS ADMM solver for quantile regression.

Uses the Proximal Operator Graph Solver (POGS) to solve quantile regression
via a vendored native C++ library loaded through ctypes.  No external ``pogs``
package is required — the solver is always available after installing pinball.

Mathematical formulation
------------------------
Quantile regression minimises the pinball loss:

.. math::

    \\min_\\beta \\sum_{i=1}^{m} \\rho_\\tau(y_i - \\mathbf{x}_i^\\top \\beta)

Using the identity
:math:`\\rho_\\tau(u) = \\tfrac{1}{2}|u| + (\\tau - \\tfrac{1}{2})\\,u`,
this maps to the POGS graph-form ``minimise  sum f_i(y_i) + sum g_j(x_j)``
subject to ``y = A x`` with:

* :math:`f_i`:  ``FunctionObj(kAbs, a=1, b=y_i, c=0.5, d=0.5−tau)``
* :math:`g_j`:  ``FunctionObj(kZero)``

This keeps the original :math:`m \\times n` design matrix (no doubling).

References
----------
.. [1] Fougner, C. and Boyd, S. (2018). "Parameter selection and
       preconditioning for a graph form solver." *Foundations and
       Trends in Optimization* 4(1).

Design (SOLID)
--------------
* **Single Responsibility** — this module contains only the POGS-specific
  solver logic.  The graph-form mapping (:func:`_build_graph_form`) and
  the native C call (:func:`_call_pogs`) are separate, testable functions.
* **Open / Closed** — ``POGSSolver`` extends ``BaseSolver`` without
  modifying it.
* **Liskov Substitution** — ``POGSSolver`` is a drop-in replacement for
  any ``BaseSolver`` subclass.
* **Interface Segregation** — callers only depend on ``BaseSolver``; no
  POGS-specific interface leaks out.
* **Dependency Inversion** — the native library is loaded lazily;
  the module is always importable.
"""

from __future__ import annotations

import ctypes
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from pinball.linear.solvers.base import BaseSolver, SolverResult

# ──────────────────────────────────────────────────────────────────────
# Lightweight stand-in for POGS FunctionObj / Function enums.
# ──────────────────────────────────────────────────────────────────────

# POGS Function enum values (from interface_defs.h)
_kAbs = 0
_kZero = 15


@dataclass
class _FunctionObj:
    """Minimal mirror of the POGS ``FunctionObj``.

    Represents: ``c * h(a * x - b) + d * x + e * x^2``
    """

    h: int = _kZero
    a: float = 1.0
    b: float = 0.0
    c: float = 1.0
    d: float = 0.0
    e: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# Graph-form construction (pure function, no native dependency)
# ──────────────────────────────────────────────────────────────────────

def _build_graph_form(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
) -> tuple[list[_FunctionObj], list[_FunctionObj]]:
    """Map (X, y, tau) to POGS graph-form function arrays.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Design matrix.
    y : ndarray, shape (m,)
        Response vector.
    tau : float
        Quantile level in (0, 1).

    Returns
    -------
    f : list of _FunctionObj, length m
        Row objectives.  Each encodes
        ``0.5 * |y_i - b_i| + (0.5 - tau) * y_i``.
    g : list of _FunctionObj, length n
        Column objectives.  Each is the zero function.
    """
    m, n = X.shape
    d_val = 0.5 - tau

    f = [
        _FunctionObj(h=_kAbs, a=1.0, b=float(y[i]), c=0.5, d=d_val, e=0.0)
        for i in range(m)
    ]
    g = [_FunctionObj(h=_kZero) for _ in range(n)]

    return f, g


# ──────────────────────────────────────────────────────────────────────
# Native library loading (lazy singleton)
# ──────────────────────────────────────────────────────────────────────

_lib: ctypes.CDLL | None = None


def _find_native_library() -> pathlib.Path:
    """Locate the ``_pogs_native`` shared library.

    Searches three locations (in order):
    1. The pinball package directory (wheel installs)
    2. The meson build directory (editable installs)
    3. Adjacent to the package root

    Returns
    -------
    pathlib.Path
        Absolute path to the shared library.

    Raises
    ------
    OSError
        If the library cannot be found.
    """
    pkg_dir = pathlib.Path(__file__).resolve().parents[2]  # pinball/
    suffix = {
        "darwin": ".dylib",
        "win32": ".dll",
    }.get(sys.platform, ".so")

    lib_name = f"lib_pogs_native{suffix}"
    alt_name = f"_pogs_native{suffix}"

    candidates = [
        pkg_dir / lib_name,
        pkg_dir / alt_name,
    ]

    # Editable install: check meson build directory
    project_root = pkg_dir.parent  # one level above pinball package
    for build_dir in sorted(project_root.glob("build/*/csrc")):
        candidates.append(build_dir / lib_name)
        candidates.append(build_dir / alt_name)

    for path in candidates:
        if path.exists():
            return path
    raise OSError(
        f"Cannot find _pogs_native shared library. "
        f"Looked in: {[str(p) for p in candidates]}"
    )


def _get_lib() -> ctypes.CDLL:
    """Return the loaded native library (singleton)."""
    global _lib
    if _lib is None:
        lib_path = _find_native_library()
        # winmode=0 restores legacy PATH-based DLL search on Windows
        # so that runtime dependencies (OpenBLAS, MinGW) are found.
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["winmode"] = 0
        _lib = ctypes.CDLL(str(lib_path), **kwargs)

        # Declare the C signature for qr_pogs_solve
        _lib.qr_pogs_solve.restype = ctypes.c_int
        _lib.qr_pogs_solve.argtypes = [
            ctypes.c_size_t,                         # m
            ctypes.c_size_t,                         # n
            ctypes.POINTER(ctypes.c_double),         # A
            ctypes.POINTER(ctypes.c_double),         # y
            ctypes.c_double,                         # tau
            ctypes.c_double,                         # abs_tol
            ctypes.c_double,                         # rel_tol
            ctypes.c_uint,                           # max_iter
            ctypes.c_uint,                           # verbose
            ctypes.c_int,                            # adaptive_rho
            ctypes.c_double,                         # rho
            ctypes.POINTER(ctypes.c_double),         # beta_out
            ctypes.POINTER(ctypes.c_double),         # optval_out
            ctypes.POINTER(ctypes.c_uint),           # final_iter_out
        ]
    return _lib


# ──────────────────────────────────────────────────────────────────────
# POGS call (thin wrapper — isolated for mocking in tests)
# ──────────────────────────────────────────────────────────────────────

def _call_pogs(
    A: np.ndarray,
    f: list[_FunctionObj],
    g: list[_FunctionObj],
    *,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-4,
    max_iter: int = 2500,
    rho: float = 1.0,
    verbose: int = 0,
    adaptive_rho: bool = True,
) -> dict:
    """Call the vendored POGS native library via ctypes.

    Parameters
    ----------
    A : ndarray
        Design matrix, shape (m, n).
    f, g : lists of _FunctionObj
        Graph-form function arrays (from :func:`_build_graph_form`).
    abs_tol, rel_tol, max_iter, rho, verbose, adaptive_rho
        Forwarded to the native solver.

    Returns
    -------
    dict
        ``{"x": beta, "optval": float, "iterations": int, "status": int}``.

    Raises
    ------
    OSError
        If the native library cannot be loaded.
    """
    lib = _get_lib()

    m, n = A.shape

    # Ensure contiguous row-major double arrays for ctypes
    A_c = np.ascontiguousarray(A, dtype=np.float64)
    y_arr = np.array([fi.b for fi in f], dtype=np.float64)

    # Extract tau from the first f object's d field: d = 0.5 - tau
    tau = 0.5 - f[0].d

    # Output buffers
    beta = np.zeros(n, dtype=np.float64)
    optval = ctypes.c_double(0.0)
    final_iter = ctypes.c_uint(0)

    status = lib.qr_pogs_solve(
        ctypes.c_size_t(m),
        ctypes.c_size_t(n),
        A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(tau),
        ctypes.c_double(abs_tol),
        ctypes.c_double(rel_tol),
        ctypes.c_uint(max_iter),
        ctypes.c_uint(verbose),
        ctypes.c_int(int(adaptive_rho)),
        ctypes.c_double(rho),
        beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(optval),
        ctypes.byref(final_iter),
    )

    return {
        "x": beta,
        "optval": optval.value,
        "iterations": final_iter.value,
        "status": status,
    }


# ──────────────────────────────────────────────────────────────────────
# Solver class (BaseSolver subclass)
# ──────────────────────────────────────────────────────────────────────

class POGSSolver(BaseSolver):
    """Quantile regression solver using the POGS ADMM library.

    This solver is well suited for *large* dense problems where the
    Fortran interior-point code becomes slow.  It uses a first-order
    ADMM algorithm with optional adaptive step-size.

    Parameters
    ----------
    abs_tol : float
        Absolute tolerance (default 1e-4).
    rel_tol : float
        Relative tolerance (default 1e-4).
    max_iter : int
        Maximum ADMM iterations (default 2500).
    rho : float
        Initial penalty parameter (default 1.0).
    verbose : int
        Verbosity level (0 = silent).
    adaptive_rho : bool
        Whether to let POGS adapt rho (default True).
    """

    def __init__(
        self,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        max_iter: int = 2500,
        rho: float = 1.0,
        verbose: int = 0,
        adaptive_rho: bool = True,
    ) -> None:
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        self.rho = rho
        self.verbose = verbose
        self.adaptive_rho = adaptive_rho

    # -- BaseSolver hooks ------------------------------------------------

    def _solve_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
        **kwargs: Any,
    ) -> SolverResult:
        """Solve quantile regression via POGS ADMM.

        Parameters
        ----------
        X, y, tau
            Standard solver inputs (already validated by ``BaseSolver``).
        **kwargs
            Overrides for any constructor parameter.

        Returns
        -------
        SolverResult
        """
        # Allow per-call overrides
        abs_tol = kwargs.get("abs_tol", self.abs_tol)
        rel_tol = kwargs.get("rel_tol", self.rel_tol)
        max_iter = kwargs.get("max_iter", self.max_iter)
        rho = kwargs.get("rho", self.rho)
        verbose = kwargs.get("verbose", self.verbose)
        adaptive_rho = kwargs.get("adaptive_rho", self.adaptive_rho)

        # Build graph-form (pure, testable)
        f, g = _build_graph_form(X, y, tau)

        # Call native POGS (isolated, mockable)
        raw = _call_pogs(
            X,
            f,
            g,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            max_iter=max_iter,
            rho=rho,
            verbose=verbose,
            adaptive_rho=adaptive_rho,
        )

        coef = raw["x"]
        residuals = y - X @ coef

        return SolverResult(
            coefficients=coef,
            residuals=residuals,
            dual_solution=None,
            objective_value=raw.get("optval", 0.0),
            status=raw.get("status", 0),
            iterations=raw.get("iterations", 0),
            solver_info={"solver": "pogs"},
        )
