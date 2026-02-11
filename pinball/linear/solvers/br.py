"""Barrodale-Roberts simplex solver for quantile regression.

Wraps the Fortran ``rqbr`` subroutine (Koenker & d'Orey, 1987/1994).
Suitable for problems with n ≲ 5 000 observations.  Supports single-τ
fitting, the full quantile-regression process (all τ), and rank-inversion
confidence intervals.

References
----------
.. [1] Barrodale, I. and Roberts, F.D.K. (1974). "Solution of an
       overdetermined system of equations in the :math:`\\ell_1` norm."
.. [2] Koenker, R. and d'Orey, V. (1987, 1994). "Computing regression
       quantiles." *Applied Statistics*.
.. [3] Koenker, R. (1994). "Confidence intervals for regression quantiles."
.. [4] Koenker, R. and Machado, J.A.F. (1999). "Goodness of fit and related
       inference processes for quantile regression." *JASA* 94(448).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import norm, t as student_t

from pinball.linear.solvers.base import BaseSolver, SolverResult
from pinball.util.bandwidth import hall_sheather


# ──────────────────────────────────────────────────────────────────────────────
# Helpers – parameter derivation
# ──────────────────────────────────────────────────────────────────────────────

def _derive_br_params(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
) -> dict:
    """Build the full argument dict expected by the Fortran ``rqbr`` routine.

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
    tau : float or None
        A value in (0, 1) for a single quantile, or *None* for the full
        quantile-regression process.

    Returns
    -------
    dict
        Keyword arguments ready to be unpacked into ``rqbr(**params)``.
    """
    n, p = X.shape
    nsol = 2
    ndsol = 2
    t_val = tau

    if tau is None:  # full-process mode
        nsol = 3 * n
        ndsol = 3 * n
        t_val = -1.0

    return dict(
        m=n,
        nn=np.int32(p),
        m5=np.int32(n + 5),
        n3=np.int32(p + 3),
        n4=np.int32(p + 4),
        a=np.asfortranarray(X, dtype=np.float64),
        b=np.ascontiguousarray(y, dtype=np.float64),
        t=float(t_val),
        toler=np.finfo(np.float64).eps ** (2.0 / 3.0),
        ift=np.int32(1),
        x=np.zeros(p, dtype=np.float64),
        e=np.zeros(n, dtype=np.float64),
        s=np.zeros(n, dtype=np.int32),
        wa=np.zeros((n + 5, p + 4), dtype=np.float64),
        wb=np.zeros(n, dtype=np.float64),
        nsol=np.int32(nsol),
        ndsol=np.int32(ndsol),
        sol=np.zeros((p + 3, nsol), dtype=np.float64),
        dsol=np.zeros((n, ndsol), dtype=np.float64),
        lsol=np.int32(0),
        h=np.zeros((p, nsol), dtype=np.int32),
        qn=np.zeros(p, dtype=np.float64),
        cutoff=np.float64(0),
        ci=np.zeros((4, p), dtype=np.float64),
        tnmat=np.zeros((4, p), dtype=np.float64),
        big=np.finfo(np.float64).max,
        lci1=np.bool_(False),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers – confidence intervals
# ──────────────────────────────────────────────────────────────────────────────

def _get_wls_weights(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    eps: float = np.finfo(np.float64).eps ** (2.0 / 3.0),
    bandwidth_method: Callable = hall_sheather,
) -> np.ndarray:
    """Compute WLS weights for the non-IID sparsity estimate.

    The local density of the response conditional on X is estimated by
    differencing two nearby quantile-regression fits (Hendricks & Koenker 1992).

    Parameters
    ----------
    X, y : ndarray
    tau : float
    eps : float
        Floor for hat-matrix leverages.
    bandwidth_method : callable(tau, n) -> float

    Returns
    -------
    ndarray, shape (n,)
    """
    n = X.shape[0]
    h = bandwidth_method(n, tau)

    # Fit at tau ± h  (import here to avoid circular at module level)
    solver = BRSolver()
    bhi = solver.solve(X, y, min(tau + h, 1 - 1e-6)).coefficients
    blo = solver.solve(X, y, max(tau - h, 1e-6)).coefficients

    dyhat = X @ (bhi - blo)
    if np.any(dyhat <= 0):
        pct = 100.0 * np.sum(dyhat <= 0) / n
        warnings.warn(f"Percent fis <= 0: {pct:.1f}")

    return np.maximum(eps, (2.0 * h) / (dyhat - eps))


def _get_rank_inversion_qn(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    iid: bool = True,
    bandwidth_method: Callable = hall_sheather,
) -> np.ndarray:
    """Compute the *qn* vector used to scale the rank-inversion CI.

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    tau : float
    iid : bool
        If ``True`` use the simpler Koenker (1994) IID formula.  If
        ``False`` use the Koenker-Machado (1999) approach with WLS.
    bandwidth_method : callable

    Returns
    -------
    ndarray, shape (p,)
    """
    n, p = X.shape
    qn = np.zeros(p, dtype=np.float64)

    if iid:
        XtX_inv = np.linalg.inv(X.T @ X)
        qn = 1.0 / np.diag(XtX_inv)
    else:
        weights = _get_wls_weights(X, y, tau, bandwidth_method=bandwidth_method)
        sqrt_w = np.sqrt(weights)
        for j in range(p):
            # WLS regression of X[:, j] on X[:, -j] with weights
            Xj = np.delete(X, j, axis=1)
            # Weight the design and response
            Xj_w = sqrt_w[:, np.newaxis] * Xj
            yj_w = sqrt_w * X[:, j]
            # Solve via least-squares
            coef_j, _, _, _ = np.linalg.lstsq(Xj_w, yj_w, rcond=None)
            resid_j = X[:, j] - Xj @ coef_j
            qn[j] = np.sum(weights * resid_j ** 2)

    return qn


# ──────────────────────────────────────────────────────────────────────────────
# BRSolver
# ──────────────────────────────────────────────────────────────────────────────

class BRSolver(BaseSolver):
    """Barrodale-Roberts simplex solver.

    Parameters
    ----------
    ci : bool
        Whether to compute rank-inversion confidence intervals.
    iid : bool
        Assume IID errors for CIs (Koenker 1994).  When ``False`` the
        Koenker-Machado (1999) heterogeneous approach is used.
    alpha : float
        Significance level for CIs  (default 0.10 → 90 % intervals).
    interp : bool
        Interpolate between the two bounding CI solutions.
    tcrit : bool
        Use Student-t critical values (``True``) or normal (``False``).
    bandwidth_method : callable
        Bandwidth selector for the sparsity / density estimate.
    """

    def __init__(
        self,
        ci: bool = False,
        iid: bool = True,
        alpha: float = 0.10,
        interp: bool = True,
        tcrit: bool = True,
        bandwidth_method: Callable = hall_sheather,
    ) -> None:
        self.ci = ci
        self.iid = iid
        self.alpha = alpha
        self.interp = interp
        self.tcrit = tcrit
        self.bandwidth_method = bandwidth_method

    # -- BaseSolver hooks ----------------------------------------------------

    def validate_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
    ) -> None:
        n, p = X.shape
        if n <= p:
            raise ValueError(
                f"BR solver requires n_samples > n_features; got "
                f"n_samples={n}, n_features={p}."
            )
        cond = np.linalg.cond(X)
        if cond >= 1.0 / np.finfo(X.dtype).eps:
            raise ValueError("Singular design matrix")

    @staticmethod
    def supports_multiple_quantiles() -> bool:
        return True

    def _solve_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
        **kwargs: Any,
    ) -> SolverResult:
        from pinball._native import rqbr  # lazy import — extension may not be built

        params = _derive_br_params(X, y, tau)
        n, p = X.shape

        ci = kwargs.get("ci", self.ci)
        iid = kwargs.get("iid", self.iid)
        alpha = kwargs.get("alpha", self.alpha)
        tcrit = kwargs.get("tcrit", self.tcrit)
        bw = kwargs.get("bandwidth_method", self.bandwidth_method)

        # Force CI when there is a single predictor
        if p == 1:
            ci = True

        if ci:
            if tcrit:
                cutoff = student_t.ppf(1 - alpha / 2, n - p)
            else:
                cutoff = norm.ppf(1 - alpha / 2)
            qn = _get_rank_inversion_qn(X, y, tau, iid=iid, bandwidth_method=bw)
            params.update(lci1=np.bool_(True), qn=qn, cutoff=np.float64(cutoff))

        # Call Fortran
        raw = rqbr(**params)

        # rqbr returns: ift, x, e, sol, dsol, lsol, h, qn, cutoff, ci, tnmat
        flag = raw[0]
        coef = raw[1]
        resid = raw[2]
        sol = raw[3]
        dsol = raw[4]
        lsol = raw[5]
        h_out = raw[6]
        qn_out = raw[7]
        cutoff_out = raw[8]
        ci_out = raw[9]
        tnmat_out = raw[10]

        if flag:
            warnings.warn(
                "Solution may be non-unique — possible conditioning problem in X."
            )

        # Compute objective value (weighted pinball loss)
        pos_resid = np.maximum(resid, 0.0)
        neg_resid = np.maximum(-resid, 0.0)
        obj = tau * np.sum(pos_resid) + (1 - tau) * np.sum(neg_resid)

        return SolverResult(
            coefficients=coef,
            residuals=resid,
            dual_solution=dsol[:, :int(lsol)] if lsol else None,
            objective_value=obj,
            status=int(flag),
            iterations=0,
            solver_info={
                "sol": sol,
                "dsol": dsol,
                "lsol": int(lsol),
                "h": h_out,
                "qn": qn_out,
                "cutoff": float(cutoff_out),
                "ci": ci_out,
                "tnmat": tnmat_out,
            },
        )
