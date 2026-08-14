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
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from pinball.linear.solvers.base import BaseSolver, SolverResult

# POGS native status codes (csrc/pogs/include/pogs.h, enum PogsStatus)
_POGS_SUCCESS = 0
_POGS_STATUS_NAME = {
    0: "POGS_SUCCESS",
    1: "POGS_INFEASIBLE",
    2: "POGS_UNBOUNDED",
    3: "POGS_MAX_ITER",
    4: "POGS_NAN_FOUND",
    5: "POGS_ERROR",
}

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


def _center_design_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Mean-center every column of *X* except a detected constant column.

    POGS's ADMM solver converges very slowly when a column sits far from
    zero (e.g. a ``year`` column around 1985 alongside an intercept
    column of 1s) -- Sinkhorn-Knopp equilibration (run internally by the
    native solver) balances row/column *norms*, but that doesn't fix a
    column being off-center, which is what actually drives ADMM's
    convergence rate here. Centering leaves the fitted slopes unchanged
    (quantile regression, like OLS, is translation-invariant given an
    intercept term to absorb the shift); only that intercept's
    coefficient needs correcting afterward, via
    :func:`_uncenter_intercept`.

    Returns
    -------
    X_centered, means, const_idx
        ``means[j]`` is the subtracted mean for column *j* (0 for the
        constant column, if any). ``const_idx`` is the index of the
        detected constant *intercept* column, or ``None`` if *X* has none
        (e.g. ``fit_intercept=False``) -- callers should skip centering
        in that case, since there would be no column left to absorb the
        compensating shift.

        A constant column whose value is exactly 0 is never selected: a
        real-world design matrix can contain a constant-zero feature
        column too (e.g. a category that never occurs in a given data
        slice, as in one-hot encodings), and that isn't the intercept --
        picking it would divide by zero in :func:`_uncenter_intercept`.
        The true intercept column added by ``QuantileRegressor`` is always
        1s, so the first *nonzero*-constant column found is used.
    """
    means = np.zeros(X.shape[1])
    const_idx = None
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.ptp(col) == 0.0:
            if const_idx is None and col[0] != 0.0:
                const_idx = j
            continue
        means[j] = col.mean()
    return X - means, means, const_idx


def _uncenter_intercept(
    coef: np.ndarray, means: np.ndarray, const_idx: int, const_value: float
) -> np.ndarray:
    """Fold the column-centering shift back into the intercept coefficient.

    If the centered fit is ``y = b0' + sum_j b_j * (X_j - mean_j)``, then in
    the original (uncentered) scale that's
    ``y = (b0' - sum_j b_j * mean_j) + sum_j b_j * X_j`` -- the correction
    is *subtracted*, not added.
    """
    coef = coef.copy()
    coef[const_idx] -= np.dot(coef, means) / const_value
    return coef


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

        # On Windows, register DLL search directories so that runtime
        # dependencies (OpenBLAS, MinGW runtime) bundled by delvewheel
        # can be found.  This supplements the early setup done by
        # pinball._dll_setup and works even if that module has not yet
        # executed (e.g. direct ctypes usage without importing pinball).
        if sys.platform == "win32":
            import contextlib
            import os

            for d in (
                lib_path.parent,                          # package dir
                lib_path.parent / ".libs",                # legacy vendor
                lib_path.parent.parent / "pinball.libs",  # delvewheel
            ):
                if d.is_dir():
                    with contextlib.suppress(OSError):
                        os.add_dll_directory(str(d))

        _lib = ctypes.CDLL(str(lib_path))

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

        # Center non-intercept columns -- see _center_design_matrix for why
        # this matters for ADMM's convergence rate. Only safe to do when
        # there's a constant column to absorb the compensating shift back
        # into (i.e. fit_intercept=True upstream).
        X_centered, means, const_idx = _center_design_matrix(X)
        X_solve = X_centered if const_idx is not None else X

        # Build graph-form (pure, testable)
        f, g = _build_graph_form(X_solve, y, tau)

        # Call native POGS (isolated, mockable)
        raw = _call_pogs(
            X_solve,
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
        if const_idx is not None:
            coef = _uncenter_intercept(coef, means, const_idx, X[0, const_idx])

        status = raw.get("status", 0)
        if status != _POGS_SUCCESS:
            warnings.warn(
                f"POGS did not reach the requested tolerance "
                f"({_POGS_STATUS_NAME.get(status, status)} after "
                f"{raw.get('iterations', 0)} iterations) — coefficients may be "
                f"inaccurate. Consider raising max_iter.",
                stacklevel=2,
            )

        residuals = y - X @ coef

        return SolverResult(
            coefficients=coef,
            residuals=residuals,
            dual_solution=None,
            objective_value=raw.get("optval", 0.0),
            status=status,
            iterations=raw.get("iterations", 0),
            solver_info={"solver": "pogs"},
        )
