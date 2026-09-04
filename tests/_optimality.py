"""An exact optimality oracle for quantile regression fits.

Why this exists
---------------
Most solver tests compare one implementation against another, or against a
handful of stored reference numbers.  Neither answers the question "is this fit
actually the minimiser?", and neither helps when a solver returns a
plausible-looking wrong answer with no peer to compare against.

Quantile regression is a linear program, so it carries its own certificate.
:math:`\\hat\\beta` minimises :math:`\\sum_i \\rho_\\tau(y_i - x_i'\\beta)`
**iff** there is a dual vector *a* with

* ``X' a = 0``,
* ``a_i = tau``      where the residual is positive,
* ``a_i = tau - 1``  where the residual is negative,
* ``a_i`` in ``[tau - 1, tau]`` where the residual is zero.

That follows directly from ``0`` lying in the subdifferential of the pinball
loss.  It needs no reference implementation, no R, and no second solver.

Two conditions, not one
-----------------------
The dual depends only on the *signs* of the residuals, so dual feasibility
alone certifies the sign pattern rather than the coefficients.  Measured while
developing this module: checking dual feasibility on its own accepted a
deliberately wrong ``beta`` (perturbed by 1e-6) at four of five quantile
levels.  The basis residuals must *also* be shown to be zero.  Both conditions
are checked below; either one on its own is not a test.

Choosing the basis
------------------
A quantile regression fit interpolates exactly ``p`` observations, so the
basis is "the ``p`` smallest absolute residuals".  Do not select it with a
fixed tolerance -- a converged fit shows a clean spectral gap (measured on
``n=400, p=4``: four residuals at 1e-15..1e-10, then a jump to 1e-4..1e-2, a
ratio of 1e5 to 1e8), but the location of that gap moves with the problem
scale and the solver's own tolerance.  This module selects the ``p`` smallest
and *requires* the gap, reporting :data:`INCONCLUSIVE` when it is absent.

Limits
------
The certificate cannot resolve error below the solver's own convergence
tolerance: with the Frisch-Newton solver at ``eps=1e-6`` a coefficient error of
1e-9 is invisible, while 1e-7 and larger is caught.  It is a check on gross
correctness, not a precision oracle.  Use R parity or a cold-solve comparison
for the tighter regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The fit is optimal, both conditions verified.
CERTIFIED = "CERTIFIED"
#: The fit is demonstrably not optimal.
FAILED = "FAILED"
#: The certificate does not apply (degenerate basis, rank deficiency).  This is
#: **not** a pass -- it means "ask another oracle".
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class Certificate:
    """Outcome of :func:`certify`, with the diagnostics behind it."""

    status: str
    reason: str = ""
    interpolation: float | None = None
    box_slack: float | None = None
    dual_residual: float | None = None
    gap_ratio: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.status == CERTIFIED

    def __str__(self) -> str:
        bits = [f"{self.status}"]
        if self.reason:
            bits.append(self.reason)
        for name in ("interpolation", "box_slack", "dual_residual", "gap_ratio"):
            value = getattr(self, name)
            if value is not None:
                bits.append(f"{name}={value:.3e}")
        return " | ".join(bits)


def certify(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    tau: float,
    *,
    atol: float = 1e-8,
    box_tol: float = 1e-9,
    dual_tol: float = 1e-9,
    min_gap_ratio: float = 1e3,
    max_cond: float = 1e12,
) -> Certificate:
    """Check that *beta* is optimal for the *tau*-quantile regression of *y* on *X*.

    Parameters
    ----------
    atol
        Tolerance for the interpolation condition, relative to ``max|y|``.
        1e-8 is calibrated for the Frisch-Newton solver's default ``eps=1e-6``:
        genuine fits land at 1e-12..1e-9, errors of 1e-7 and up land at 1e-7
        and above.  Tighten only if the solver is run tighter.
    min_gap_ratio
        Required ratio between the ``(p+1)``-th and ``p``-th smallest absolute
        residuals.  Below this the basis is ambiguous and the result is
        :data:`INCONCLUSIVE`.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    beta = np.asarray(beta, dtype=np.float64).ravel()
    n, p = X.shape

    if not np.all(np.isfinite(beta)):
        return Certificate(FAILED, "beta is not finite")
    if n <= p:
        return Certificate(INCONCLUSIVE, f"n={n} <= p={p}: no interior basis")

    r = y - X @ beta
    absr = np.abs(r)
    order = np.argsort(absr)
    basis = order[:p]
    scale = max(float(np.abs(y).max()), 1.0)

    # --- gap: is the basis well separated from the rest? ------------------
    r_p = float(absr[order[p - 1]])
    r_next = float(absr[order[p]])
    gap_ratio = r_next / r_p if r_p > 0 else np.inf

    # --- condition 1: the basis observations are interpolated -------------
    interpolation = r_p / scale

    XZ = X[basis]
    cond = float(np.linalg.cond(XZ))
    if cond > max_cond:
        return Certificate(
            INCONCLUSIVE,
            f"basis design is ill-conditioned (cond={cond:.2e})",
            interpolation=interpolation,
            gap_ratio=gap_ratio,
            details={"cond": cond},
        )

    # Interpolation is checked *before* the gap, because failing it is decisive
    # on its own: an optimal fit interpolates p observations, so if even the p
    # smallest residuals are far from zero, beta is not a basic solution.  (Under
    # a non-unique optimum a point on the optimal face could interpolate fewer
    # than p; data in general position makes that a measure-zero concern, and it
    # would surface as a surprising FAILED rather than a silent pass.)
    if interpolation > atol:
        return Certificate(
            FAILED,
            f"does not interpolate p={p} observations "
            f"(|r|_(p)/scale = {interpolation:.3e} > {atol:g})",
            interpolation=interpolation,
            gap_ratio=gap_ratio,
            details={"smallest_abs_residuals": absr[order[: p + 3]].tolist()},
        )

    if gap_ratio < min_gap_ratio:
        # The basis cannot be identified: either the fit is nowhere near a vertex
        # (e.g. an unconverged interior-point iterate) or the problem is
        # genuinely degenerate.  No verdict is available -- and this is emphatically
        # not a pass.
        return Certificate(
            INCONCLUSIVE,
            f"no spectral gap after p={p} residuals "
            f"(|r|_(p)={r_p:.3e}, |r|_(p+1)={r_next:.3e})",
            interpolation=interpolation,
            gap_ratio=gap_ratio,
            details={"smallest_abs_residuals": absr[order[: p + 3]].tolist()},
        )

    # --- condition 2: a feasible dual exists ------------------------------
    off = np.ones(n, dtype=bool)
    off[basis] = False
    g = tau * X[off & (r > 0)].sum(axis=0) + (tau - 1.0) * X[off & (r < 0)].sum(axis=0)

    a_basis = np.linalg.solve(XZ.T, -g)

    # Dual feasibility, normalised so the number is dimensionless: |a| <= 1, so
    # dividing by the largest column sum of |X| bounds the achievable residual.
    a_full = np.zeros(n)
    a_full[basis] = a_basis
    a_full[off & (r > 0)] = tau
    a_full[off & (r < 0)] = tau - 1.0
    col_scale = max(float(np.abs(X).sum(axis=0).max()), 1.0)
    dual_residual = float(np.abs(X.T @ a_full).max() / col_scale)

    box_slack = float(min(a_basis.min() - (tau - 1.0), tau - a_basis.max()))

    ok_interp = interpolation <= atol
    ok_box = box_slack >= -box_tol
    ok_dual = dual_residual <= dual_tol

    if ok_interp and ok_box and ok_dual:
        status, reason = CERTIFIED, ""
    else:
        failed = []
        if not ok_interp:
            failed.append(f"basis residuals not zero ({interpolation:.3e} > {atol:g})")
        if not ok_box:
            failed.append(f"dual outside [tau-1, tau] (slack {box_slack:.3e})")
        if not ok_dual:
            failed.append(f"X'a != 0 ({dual_residual:.3e} > {dual_tol:g})")
        status, reason = FAILED, "; ".join(failed)

    return Certificate(
        status,
        reason,
        interpolation=interpolation,
        box_slack=box_slack,
        dual_residual=dual_residual,
        gap_ratio=gap_ratio,
        details={"basis": basis.tolist(), "cond": cond},
    )


def certify_dual(
    X: np.ndarray,
    residuals: np.ndarray,
    dual: np.ndarray,
    tau: float,
    *,
    box_tol: float = 1e-9,
    dual_tol: float = 1e-9,
    complementarity_tol: float = 1e-6,
) -> Certificate:
    """Cheap ``O(np)`` screen using a dual vector the solver already produced.

    ``FNBSolver`` fills in ``SolverResult.dual_solution``, so this costs one
    matrix-vector product rather than a factorisation.  It **screens**; it does
    not prove.  An interior-point iterate is strictly interior, so
    complementarity holds only to the solver's tolerance -- use :func:`certify`
    for a verdict.
    """
    X = np.asarray(X, dtype=np.float64)
    r = np.asarray(residuals, dtype=np.float64).ravel()
    a = np.asarray(dual, dtype=np.float64).ravel()

    col_scale = max(float(np.abs(X).sum(axis=0).max()), 1.0)
    dual_residual = float(np.abs(X.T @ a).max() / col_scale)
    box_slack = float(min(a.min() - (tau - 1.0), tau - a.max()))

    # Complementarity: away from the basis the dual must sit on the bound whose
    # side the residual is on.  The p basis observations are excluded -- their
    # residual is zero only up to rounding (~1e-16, so `r > 0` still classifies
    # them) while their dual is legitimately interior, and including them makes
    # this check report violations of order 1 on a perfectly good fit.
    p = X.shape[1]
    interior = np.argsort(np.abs(r))[:p]
    off = np.ones(r.shape[0], dtype=bool)
    off[interior] = False
    pos, neg = off & (r > 0), off & (r < 0)
    comp = 0.0
    if pos.any():
        comp = max(comp, float(np.abs(a[pos] - tau).max()))
    if neg.any():
        comp = max(comp, float(np.abs(a[neg] - (tau - 1.0)).max()))

    failed = []
    if dual_residual > dual_tol:
        failed.append(f"X'a != 0 ({dual_residual:.3e})")
    if box_slack < -box_tol:
        failed.append(f"dual outside [tau-1, tau] (slack {box_slack:.3e})")
    if comp > complementarity_tol:
        failed.append(f"complementarity violated ({comp:.3e})")

    return Certificate(
        FAILED if failed else CERTIFIED,
        "; ".join(failed),
        box_slack=box_slack,
        dual_residual=dual_residual,
        details={"complementarity": comp},
    )
