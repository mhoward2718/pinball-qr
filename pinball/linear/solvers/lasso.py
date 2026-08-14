"""L1-penalised (lasso) quantile regression solver.

Augments the design matrix with a penalty block and delegates to :class:`FNBSolver`.
Follows the approach in R's ``rq.fit.lasso`` (Belloni & Chernozhukov, 2011).

The augmented penalty rows must contribute a *symmetric* ``lambda_j *
|beta_j|`` term regardless of ``beta_j``'s sign. R's Fortran implementation
gets this by fitting the penalty rows at the median (tau=0.5, where the
check function rho_0.5(u) = 0.5*|u| is symmetric in u by construction)
while the data rows are fit at the target tau -- done via a custom mixed
right-hand side passed straight to the Frisch-Newton routine. This module
can't override FNBSolver's internal rhs, so it reaches the same symmetric
result differently: each penalised coefficient gets *two* augmented rows,
``+lambda_j*e_j`` and ``-lambda_j*e_j`` (both targeting y=0), fit at the
*same* single tau as everything else. That relies on the general identity
rho_tau(u) + rho_tau(-u) = |u|, which holds for every tau in (0, 1), not
just 0.5 -- so the two rows' contributions sum to exactly
``lambda_j * |beta_j|`` no matter what tau is. A single row per
coefficient (the earlier approach here) does not have this property away
from tau=0.5: it contributes ``lambda_j*|beta_j|*(1-tau)`` when
``beta_j > 0`` and ``lambda_j*|beta_j|*tau`` when ``beta_j < 0`` --
an asymmetric, sign-dependent penalty that silently over-shrinks one sign
and under-shrinks the other, worst exactly at the extreme tau values this
solver is often used for.

References
----------
.. [1] Belloni, A. and Chernozhukov, V. (2011). "ℓ1-penalized quantile
       regression in high-dimensional sparse models." *Annals of Statistics*.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pinball.linear.solvers.base import BaseSolver, SolverResult
from pinball.linear.solvers.fnb import FNBSolver
from pinball.util.lambda_selection import lambda_hat_bcv


class LassoSolver(BaseSolver):
    """L1-penalised quantile regression via an augmented interior-point solve.

    Parameters
    ----------
    lambda_ : float or None
        Penalty parameter.  If ``None`` (default) the Belloni-Chernozhukov
        default is used.
    penalize_intercept : bool
        Whether the first column (intercept) is penalised.  Default ``False``.
    beta : float
        Interior-point damping (forwarded to :class:`FNBSolver`).
    eps : float
        Convergence tolerance (forwarded to :class:`FNBSolver`).
    """

    def __init__(
        self,
        lambda_: float | None = None,
        penalize_intercept: bool = False,
        beta: float = 0.99995,
        eps: float = 1e-6,
    ) -> None:
        self.lambda_ = lambda_
        self.penalize_intercept = penalize_intercept
        self._fnb = FNBSolver(beta=beta, eps=eps)

    def _solve_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
        **kwargs: Any,
    ) -> SolverResult:
        n, p = X.shape

        # Determine lambda
        lam = self.lambda_
        if lam is None:
            lam = lambda_hat_bcv(X, tau)

        # Build penalty vector (0 for intercept if not penalised)
        pen = np.full(p, lam, dtype=np.float64)
        if not self.penalize_intercept and p > 1:
            pen[0] = 0.0

        # Augment with +/- rows for every penalised coefficient (see module
        # docstring: this is what makes the penalty symmetric at any tau).
        penalized = np.flatnonzero(pen)
        R = np.diag(pen)[penalized]
        X_aug = np.vstack([X, R, -R])
        y_aug = np.concatenate([y, np.zeros(2 * len(penalized))])

        result = self._fnb.solve(X_aug, y_aug, tau, **kwargs)

        # Residuals on original data only
        residuals = y - X @ result.coefficients

        pos_resid = np.maximum(residuals, 0.0)
        neg_resid = np.maximum(-residuals, 0.0)
        obj = tau * np.sum(pos_resid) + (1 - tau) * np.sum(neg_resid)
        obj += lam * np.sum(np.abs(result.coefficients[int(not self.penalize_intercept):]))

        return SolverResult(
            coefficients=result.coefficients,
            residuals=residuals,
            dual_solution=None,
            objective_value=obj,
            status=result.status,
            iterations=result.iterations,
            solver_info={"lambda": lam, **result.solver_info},
        )
