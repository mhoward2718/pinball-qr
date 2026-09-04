"""Frisch-Newton interior-point solver for quantile regression.

Wraps the Fortran ``rqfnb`` subroutine which implements the Frisch-Newton
algorithm with log-barrier (Portnoy & Koenker, 1997).  This is the
recommended solver for medium-to-large problems.

References
----------
.. [1] Portnoy, S. and Koenker, R. (1997). "The Gaussian hare and the
       Laplacian tortoise." *Statistical Science* 12(4): 279–300.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from pinball.linear.solvers.base import BaseSolver, SolverResult

#: Mirrors ``parameter( maxit = 500 )`` in ``lpfnb`` (``fortran/rqfnb.f``).  The
#: Fortran does not export it, so it has to be duplicated here; keep in sync.
_LPFNB_MAXIT = 500

#: ``SolverResult.status`` value meaning "the iteration budget ran out before the
#: duality gap met ``eps``".  Deliberately out of band: LAPACK's ``dposv`` (the
#: only other source of ``status`` here) reports 0, or a leading-minor index in
#: ``1..p``, so no collision is possible.
STATUS_NOT_CONVERGED = -99

#: ``SolverResult.status`` value meaning "the routine returned non-finite
#: coefficients".  Same out-of-band rationale as :data:`STATUS_NOT_CONVERGED`.
STATUS_NOT_FINITE = -98


class FNBSolver(BaseSolver):
    """Frisch-Newton interior-point solver (bounded variables formulation).

    Parameters
    ----------
    beta : float
        Step-size damping parameter, must be in (0, 1).
        Default 0.99995 (as in quantreg).
    eps : float
        Convergence tolerance.
    """

    def __init__(self, beta: float = 0.99995, eps: float = 1e-6) -> None:
        if not (0 < beta < 1):
            raise ValueError(f"beta must be in (0, 1), got {beta}.")
        # A non-positive eps makes lpfnb's `gap > eps` loop test unsatisfiable;
        # the iteration then runs on until the gap goes NaN, at which point the
        # comparison is false and the routine returns NaN coefficients with
        # info = 0.  Reject it here rather than let that happen silently.
        if not eps > 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.beta = beta
        self.eps = eps

    # -- BaseSolver hooks ----------------------------------------------------

    def validate_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
    ) -> None:
        if tau < self.eps or tau > 1 - self.eps:
            raise ValueError(
                f"FNB requires tau in ({self.eps}, {1 - self.eps}), got {tau}."
            )

    def _solve_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
        **kwargs: Any,
    ) -> SolverResult:
        from pinball._native import rqfnb  # lazy import

        n, p = X.shape
        beta = kwargs.get("beta", self.beta)
        eps = kwargs.get("eps", self.eps)

        # Fortran expects a(p, n) — column-major transposed design
        a = np.asfortranarray(X.T, dtype=np.float64)

        # c = -y  (the Fortran routine minimises c^T x)
        c = np.ascontiguousarray(-y, dtype=np.float64)

        # Right-hand side: (1 - tau) * colSums(X)
        rhs = (1.0 - tau) * X.sum(axis=0).astype(np.float64)

        d = np.ones(n, dtype=np.float64)
        u = np.ones(n, dtype=np.float64)

        # Workspace: wn has shape (n, 9) when reshaped; pass as (n*9,)
        # but Fortran declares wn(n, 9), so we pass a 2-D array
        wn = np.zeros((n, 9), dtype=np.float64, order="F")
        wn[:, 0] = 1.0 - tau  # initial dual

        # wp has shape (p, p+3)
        wp = np.zeros((p, p + 3), dtype=np.float64, order="F")

        nit = np.zeros(3, dtype=np.int32)
        info = np.int32(0)

        # Call Fortran: a,y,rhs,d,u,wn,wp,nit,info = rqfnb(a,y,rhs,d,u,beta,eps,wn,wp,nit,info,[n,p])
        # n, p are optional trailing args (inferred from array shapes)
        (a_out, c_out, rhs_out, d_out, u_out,
         wn_out, wp_out, nit_out, info_out) = rqfnb(
            a, c, rhs, d, u, beta, eps, wn, wp, nit, info,
        )

        info_val = int(info_out.item()) if isinstance(info_out, np.ndarray) else int(info_out)

        if info_val != 0:
            warnings.warn(
                f"rqfnb info = {info_val}: possibly singular design.", stacklevel=2
            )

        n_iter = int(nit_out[0]) if hasattr(nit_out, "__len__") else 0

        # `lpfnb` leaves its loop when the duality gap is small OR the iteration
        # budget runs out, and in the latter case it does *not* touch `info` —
        # `info` is only ever set by stepy's dposv.  Without the check below an
        # unconverged (i.e. wrong) fit is returned looking exactly like a good
        # one.  The iteration count is the only evidence available here.
        converged = n_iter < _LPFNB_MAXIT
        if converged:
            status = info_val
        else:
            status = info_val if info_val != 0 else STATUS_NOT_CONVERGED
            warnings.warn(
                f"rqfnb hit its iteration limit ({_LPFNB_MAXIT}) without reaching "
                f"eps={eps:g}; the returned coefficients may not be optimal. "
                "Consider loosening eps or rescaling the design.",
                stacklevel=2,
            )

        coefficients = -wp_out[:, 0]

        # Defence in depth: lpfnb can leave its loop with a NaN gap (the test
        # `gap > eps` is false for NaN) and hand back NaN coefficients while
        # info is still 0.  Never return that quietly.
        if not np.all(np.isfinite(coefficients)):
            status = STATUS_NOT_FINITE
            warnings.warn(
                "rqfnb returned non-finite coefficients; the fit failed. This "
                "usually indicates a badly scaled or rank-deficient design.",
                stacklevel=2,
            )

        residuals = y - X @ coefficients

        # Objective: weighted pinball loss
        pos_resid = np.maximum(residuals, 0.0)
        neg_resid = np.maximum(-residuals, 0.0)
        obj = tau * np.sum(pos_resid) + (1.0 - tau) * np.sum(neg_resid)

        # wn[:, 2] holds the optimal primal weights x in [0, 1] of the bounded
        # dual LP.  Shifting by (1 - tau) gives the quantile-regression dual
        # a in [tau - 1, tau] satisfying X' a = 0 — i.e. a certificate of
        # optimality that callers can check without re-solving.
        dual = wn_out[:, 2] - (1.0 - tau)

        return SolverResult(
            coefficients=coefficients,
            residuals=residuals,
            dual_solution=dual,
            objective_value=obj,
            status=status,
            iterations=n_iter,
            solver_info={"nit": nit_out, "converged": converged},
        )
