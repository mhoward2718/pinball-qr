"""Preprocessing ("globbing") solver for large-n quantile regression.

Single τ follows Portnoy & Koenker (1997): fit a subsample, use a simultaneous
confidence band to decide which observations are *confidently* above or below
the fitted hyperplane, replace each of those two groups by one summed
pseudo-observation, and re-solve the much smaller problem.  Because the pinball
loss's optimality condition depends only on residual *signs*, that reduced
problem has the same solution as the original — provided the predicted signs
were right, which is what the fixup loop verifies.

Multiple τ additionally follows Chernozhukov, Fernández-Val & Melly (2022):
the fit at one quantile is an excellent predictor of residual signs at the
next, so the whole grid shares one subsample and one leverage band, and each τ
is seeded with the residuals from the previous one.  This mirrors
``quantreg::rq.fit.pfnb``.

Why the answers stay exact
--------------------------
The pinball loss is convex and positively homogeneous, hence subadditive:
``rho(sum u_i) <= sum rho(u_i)``.  So for *any* choice of glob sets the reduced
objective is a pointwise lower bound on the full one.  If at the reduced
optimum every globbed observation really does have the sign its glob assumed,
the loss is linear on each glob and the two objectives agree there — a
minimiser of a lower bound that touches the true objective minimises the true
objective too.  Nothing in that argument mentions τ spacing, the subsample
size, the band, or the random seed: those affect only how many fixup rounds are
needed.  A coarse grid is exactly as correct as a fine one, just slower.  The
one way to break it is to return before the sign check passes, which is why the
loop below always escalates rather than giving up.

References
----------
.. [1] Portnoy, S. and Koenker, R. (1997). "The Gaussian hare and the
       Laplacian tortoise." *Statistical Science* 12(4): 279-300.
.. [2] Chernozhukov, V., Fernández-Val, I. and Melly, B. (2022). "Fast
       algorithms for the quantile regression process." *Empirical Economics*
       62(1): 7-33.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

from pinball.linear._bootstrap import _ensure_rng
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
        Fixup rounds allowed before the subsample size is doubled.
    eps : float
        Floor for the bandwidth used to detect extreme residuals.
    random_state : int, numpy.random.RandomState or None
        Seed for the subsample draw.  Following the rest of the package
        (see :func:`pinball.linear._bootstrap.bootstrap`), pass an int for a
        reproducible fit.  The seed changes which observations are globbed, and
        hence the speed, but not the answer.
    """

    def __init__(
        self,
        inner_solver: BaseSolver | None = None,
        mm_factor: float = 0.8,
        max_bad_fixups: int = 3,
        eps: float = 1e-6,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.inner_solver = inner_solver or FNBSolver()
        self.mm_factor = mm_factor
        self.max_bad_fixups = max_bad_fixups
        self.eps = eps
        self.random_state = random_state

    @staticmethod
    def supports_multiple_quantiles() -> bool:
        return True

    # ------------------------------------------------------------------
    # Pieces shared by the single-τ and grid paths
    # ------------------------------------------------------------------

    def _initial_m(self, n: int, p: int) -> int:
        """Portnoy-Koenker subsample size, matching ``quantreg``'s *code*
        (``floor(n^(2/3) * sqrt(p))``) rather than its documentation."""
        return int(np.floor(n ** (2.0 / 3.0) * np.sqrt(p)))

    def _leverage_band(self, Xs: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Simultaneous confidence band ``sqrt(x_i' (Xs'Xs)^-1 x_i)``.

        Computed from a QR of ``Xs`` rather than a Cholesky of the Gram matrix
        ``Xs'Xs``: both give the same leverages, but forming the Gram matrix
        squares the condition number, so the Cholesky route fails outright on
        designs that are merely awkward (``cond(Xs) ~ 1e9`` becomes ``1e18``,
        i.e. numerically indefinite).  R's ``rq.fit.pfn`` uses the Gram route;
        this is a deliberate, behaviour-preserving improvement on it.
        """
        R = np.linalg.qr(Xs, mode="r")
        # R' Z = X'  =>  Z[:, i] = R^-T x_i, and band_i = ||Z[:, i]||.
        Z = solve_triangular(R, X.T, trans="T", lower=False)
        band = np.sqrt(np.einsum("ij,ij->j", Z, Z))
        # Floor the band itself, as R does, so the threshold and the studentised
        # residual agree about which observations are numerically flat.
        return np.maximum(self.eps, band)

    def _glob_and_fixup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
        r: np.ndarray,
        band: np.ndarray,
        m: int,
        **kwargs: Any,
    ) -> tuple[SolverResult | None, int, dict[str, Any]]:
        """One preprocessing pass at *tau*, seeded with residuals *r*.

        Returns ``(solution, m, diagnostics)``.  ``solution`` is ``None`` when
        the pass could not verify its globs, in which case ``m`` has been
        doubled and the caller should try again — never return an unverified
        fit.
        """
        n = X.shape[0]
        fit = self.inner_solver.solve

        M = self.mm_factor * m
        lo_q = max(1.0 / n, tau - M / (2.0 * n))
        hi_q = min(tau + M / (2.0 * n), (n - 1.0) / n)
        kappa = np.quantile(r / band, [lo_q, hi_q])

        s_l = r < band * kappa[0]
        s_u = r > band * kappa[1]

        bad_fixups = 0
        fixups = 0
        solution = None

        while bad_fixups < self.max_bad_fixups:
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
            n_globbed = int(xx.shape[0])

            solution = fit(xx, yy, tau, **kwargs)
            if solution.status != 0:
                warnings.warn(
                    f"inner solver reported status {solution.status} on the "
                    f"reduced problem at tau={tau:g}; preprocessing builds rows "
                    "that are sums over many observations, which can be badly "
                    "scaled.",
                    stacklevel=3,
                )

            r = y - X @ solution.coefficients
            su_bad = (r < 0) & s_u
            sl_bad = (r > 0) & s_l
            n_bad = int(np.count_nonzero(su_bad | sl_bad))

            if n_bad == 0:
                return solution, m, {
                    "n_globbed": n_globbed,
                    "fixups": fixups,
                    "bad_fixups": bad_fixups,
                }

            if n_bad > 0.1 * M:
                warnings.warn(
                    "Too many fixups — doubling subsample size m.", stacklevel=3
                )
                return None, 2 * m, {"n_globbed": n_globbed, "fixups": fixups,
                                     "bad_fixups": bad_fixups}

            s_u = s_u & ~su_bad
            s_l = s_l & ~sl_bad
            fixups += n_bad
            bad_fixups += 1

        # Budget exhausted with globs still unverified.  Escalate rather than
        # redraw at the same size: R's rq.fit.pfn loops here forever on data
        # that keeps producing a few bad signs, because nothing about the next
        # attempt is different.  Doubling guarantees termination, since m >= n
        # falls through to a full-data solve.
        warnings.warn(
            f"preprocessing could not verify its globs at tau={tau:g} within "
            f"{self.max_bad_fixups} fixup rounds — doubling subsample size m.",
            stacklevel=3,
        )
        return None, 2 * m, {"n_globbed": 0, "fixups": fixups, "bad_fixups": bad_fixups}

    def _finalize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        solution: SolverResult,
        diag: dict[str, Any],
    ) -> SolverResult:
        residuals = y - X @ solution.coefficients
        return SolverResult(
            coefficients=solution.coefficients,
            residuals=residuals,
            dual_solution=None,
            objective_value=solution.objective_value,
            status=solution.status,
            iterations=solution.iterations,
            solver_info={**solution.solver_info, "preprocessing": True, **diag},
        )

    # ------------------------------------------------------------------
    # Single τ
    # ------------------------------------------------------------------

    def _solve_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
        **kwargs: Any,
    ) -> SolverResult:
        n, p = X.shape
        rng = _ensure_rng(kwargs.pop("random_state", self.random_state))
        m = self._initial_m(n, p)

        while True:
            if m >= n:
                return self.inner_solver.solve(X, y, tau, **kwargs)

            idx = rng.choice(n, m, replace=False)
            Xs, ys = X[idx], y[idx]
            prelim = self.inner_solver.solve(Xs, ys, tau, **kwargs)
            band = self._leverage_band(Xs, X)
            r = y - X @ prelim.coefficients

            solution, m, diag = self._glob_and_fixup(X, y, tau, r, band, m, **kwargs)
            if solution is not None:
                return self._finalize(X, y, solution, diag)

    # ------------------------------------------------------------------
    # A grid of τ — Chernozhukov, Fernández-Val & Melly Algorithm 2
    # ------------------------------------------------------------------

    def _solve_multi_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        taus: Sequence[float],
        **kwargs: Any,
    ) -> list[SolverResult]:
        n, p = X.shape
        rng = _ensure_rng(kwargs.pop("random_state", self.random_state))
        m0 = self._initial_m(n, p)

        if m0 >= n:
            return [self.inner_solver.solve(X, y, t, **kwargs) for t in taus]

        # Ascending τ so each fit seeds the next; results are put back in the
        # caller's order at the end.
        order = np.argsort(np.asarray(taus, dtype=float), kind="stable")

        # One subsample and one leverage band for the whole grid.
        idx = rng.choice(n, m0, replace=False)
        Xs, ys = X[idx], y[idx]
        band = self._leverage_band(Xs, X)
        prelim = self.inner_solver.solve(Xs, ys, float(taus[order[0]]), **kwargs)
        r = y - X @ prelim.coefficients

        results: list[SolverResult | None] = [None] * len(taus)
        for j in order:
            tau = float(taus[j])
            m = m0                      # reset per τ, as quantreg's pfnb does
            while True:
                if m >= n:
                    solution = self.inner_solver.solve(X, y, tau, **kwargs)
                    diag: dict[str, Any] = {"n_globbed": n, "fixups": 0,
                                            "bad_fixups": 0}
                    break
                solution, m, diag = self._glob_and_fixup(
                    X, y, tau, r, band, m, **kwargs
                )
                if solution is not None:
                    break
            diag["warm_started"] = True
            results[j] = self._finalize(X, y, solution, diag)
            # The CFM step: carry this fit's residuals into the next quantile.
            r = results[j].residuals

        if any(res is None for res in results):  # pragma: no cover - unreachable
            raise RuntimeError(
                "preprocessing produced no result for some quantile; this is a "
                "bug — the per-τ loop only exits with a verified solution."
            )
        return results  # type: ignore[return-value]
