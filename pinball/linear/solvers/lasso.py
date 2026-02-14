"""L1-penalised (lasso) quantile regression solver.

Augments the design matrix with a penalty block and delegates to :class:`FNBSolver`.
Follows the approach in R's ``rq.fit.lasso`` (Belloni & Chernozhukov, 2011).

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

        # Augment:  X_aug = [X; diag(pen)],  y_aug = [y; 0]
        X_aug = np.vstack([X, np.diag(pen)])
        y_aug = np.concatenate([y, np.zeros(p)])

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
