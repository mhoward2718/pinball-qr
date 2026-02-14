"""Preprocessing + Frisch-Newton solver for large-n quantile regression.

Implements the Portnoy-Koenker preprocessing approach: fit on a subsample,
identify observations clearly above / below the quantile hyperplane,
aggregate them, then re-fit on the reduced problem.  Iterates until the
solution stabilises (or a fixup budget is exhausted).

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
from pinball.linear.solvers.fnb import FNBSolver


class PreprocessingSolver(BaseSolver):
    """Preprocessing wrapper that accelerates any inner solver on large data.

    Parameters
    ----------
    inner_solver : BaseSolver or None
        Solver applied to the reduced subproblem.  Defaults to
        :class:`FNBSolver`.
    mm_factor : float
        Controls how many observations are kept in the "middle" band
        (as a fraction of the initial subsample size *m*).
    max_bad_fixups : int
        Maximum number of fixup iterations before doubling *m*.
    eps : float
        Floor for the bandwidth used to detect extreme residuals.
    """

    def __init__(
        self,
        inner_solver: BaseSolver | None = None,
        mm_factor: float = 0.8,
        max_bad_fixups: int = 3,
        eps: float = 1e-6,
    ) -> None:
        self.inner_solver = inner_solver or FNBSolver()
        self.mm_factor = mm_factor
        self.max_bad_fixups = max_bad_fixups
        self.eps = eps

    def _solve_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
        **kwargs: Any,
    ) -> SolverResult:
        n, p = X.shape
        eps = self.eps
        fit = self.inner_solver.solve

        # Initial subsample size (Portnoy-Koenker heuristic)
        m = int(np.rint(np.sqrt(p) * n ** (2.0 / 3.0)))

        while True:
            if m >= n:
                return fit(X, y, tau, **kwargs)

            # Draw a random subsample
            idx = np.random.choice(n, m, replace=False)
            Xs, ys = X[idx], y[idx]
            solution = fit(Xs, ys, tau, **kwargs)

            # Cholesky of X_s^T X_s for leverage / bandwidth
            L = np.linalg.cholesky(Xs.T @ Xs)
            L_inv = np.linalg.inv(L)
            band = np.sqrt(np.sum((X @ L_inv) ** 2, axis=1))

            # Signed residuals on the *full* data
            r = y - X @ solution.coefficients

            M = self.mm_factor * m
            lo_q = max(1.0 / n, tau - M / (2.0 * n))
            hi_q = min(tau + M / (2.0 * n), (n - 1.0) / n)
            kappa = np.quantile(r / np.maximum(eps, band), [lo_q, hi_q])

            s_l = r < band * kappa[0]
            s_u = r > band * kappa[1]

            bad_fixups = 0
            converged = False

            while bad_fixups < self.max_bad_fixups:
                # Assemble reduced problem: middle band + aggregated extremes
                mask_mid = ~s_l & ~s_u
                xx_parts = [X[mask_mid]]
                yy_parts = [y[mask_mid]]

                if np.any(s_l):
                    xx_parts.append(X[s_l].sum(axis=0, keepdims=True))
                    yy_parts.append(np.atleast_1d(y[s_l].sum()))
                if np.any(s_u):
                    xx_parts.append(X[s_u].sum(axis=0, keepdims=True))
                    yy_parts.append(np.atleast_1d(y[s_u].sum()))

                xx = np.vstack(xx_parts)
                yy = np.concatenate(yy_parts)

                solution = fit(xx, yy, tau, **kwargs)

                # Check for misclassified observations
                r = y - X @ solution.coefficients
                su_bad = (r < 0) & s_u
                sl_bad = (r > 0) & s_l

                if not (np.any(su_bad) or np.any(sl_bad)):
                    converged = True
                    break

                n_bad = int(np.sum(su_bad | sl_bad))
                if n_bad > 0.1 * M:
                    warnings.warn("Too many fixups — doubling subsample size m.", stacklevel=2)
                    m = 2 * m
                    break

                # Reclassify the bad observations
                s_u = s_u & ~su_bad
                s_l = s_l & ~sl_bad
                bad_fixups += 1

            if converged:
                break

        # Compute residuals on full data
        residuals = y - X @ solution.coefficients
        return SolverResult(
            coefficients=solution.coefficients,
            residuals=residuals,
            dual_solution=None,
            objective_value=solution.objective_value,
            status=solution.status,
            iterations=solution.iterations,
            solver_info={**solution.solver_info, "preprocessing": True},
        )
