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

from pinball.solvers.base import BaseSolver, SolverResult


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

        if isinstance(info_out, np.ndarray):
            info_val = int(info_out.item())
        else:
            info_val = int(info_out)

        if info_val != 0:
            warnings.warn(
                f"rqfnb info = {info_val}: possibly singular design."
            )

        coefficients = -wp_out[:, 0]
        residuals = y - X @ coefficients

        # Objective: weighted pinball loss
        pos_resid = np.maximum(residuals, 0.0)
        neg_resid = np.maximum(-residuals, 0.0)
        obj = tau * np.sum(pos_resid) + (1.0 - tau) * np.sum(neg_resid)

        return SolverResult(
            coefficients=coefficients,
            residuals=residuals,
            dual_solution=None,
            objective_value=obj,
            status=info_val,
            iterations=int(nit_out[0]) if hasattr(nit_out, "__len__") else 0,
            solver_info={"nit": nit_out},
        )
