"""Convolution-smoothed quantile regression ("conquer").

The pinball loss is not differentiable at zero, which is why the other solvers
in this package are linear programs.  Convolution smoothing replaces it with

    L_h(b) = (1/n) sum_i (rho_tau * K_h)(y_i - x_i'b)

-- the check function convolved with a kernel of bandwidth ``h`` -- which is
twice differentiable and convex, so it can be minimised by gradient descent
instead.  The gradient has a closed form,

    grad L_h(b) = (1/n) X' [ Kbar(-r/h) - tau ],     r = y - X b

where ``Kbar`` is the kernel's CDF.  As ``h -> 0``, ``Kbar(-r/h)`` tends to
``1{r < 0}`` and the gradient tends to the ordinary quantile regression
subgradient.

**This is a different estimator, not a faster route to the same answer.**  It
targets a pseudo-parameter ``b_h(tau) = b(tau) - h^(s+1) B(tau) + o(h^(s+1))``:
the smoothing buys a lower asymptotic variance and a weaker dimension
requirement for asymptotic normality, and pays for it with a bias.  Measured
against the exact simplex solution on n=800, p=3 with the default bandwidth,
coefficients differ by ~7e-3 on a coefficient scale of ~3.

Two consequences worth stating plainly, because they are surprising if you
expect a drop-in solver:

* It is **not** covered by the optimality certificate in the test suite, and
  it will not agree with ``br``/``fn``/``pfn``, which all solve the exact LP.
* It is **not scale equivariant in y** at the default bandwidth.  ``h`` is a
  function of ``(n, p)`` alone, so rescaling ``y`` by ``c`` without also
  rescaling ``h`` changes the estimator (measured: 1.7e-3 relative at c=2).
  Scaling ``h`` by ``c`` restores the identity to ~3e-10.

Independent implementation
--------------------------
Written from the published algorithm, not ported: the R ``conquer`` package is
GPL-3 and this project is MIT, so no code was taken from it.  The bandwidth
default and the asymptotic variance form were determined by probing the R
package's *output* and are verified against it in the test suite.

References
----------
.. [1] He, X., Pan, X., Tan, K. M. and Zhou, W.-X. (2023). "Smoothed quantile
       regression with large-scale inference." *Journal of Econometrics*
       232(2): 367-388.
.. [2] Fernandes, M., Guerre, E. and Horta, E. (2021). "Smoothing quantile
       regressions." *Journal of Business & Economic Statistics* 39(1):
       338-357.
.. [3] Barzilai, J. and Borwein, J. M. (1988). "Two-point step size gradient
       methods." *IMA Journal of Numerical Analysis* 8(1): 141-148.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.special import expit, ndtr

from pinball.linear.solvers.base import BaseSolver, SolverResult

#: ``SolverResult.status`` value meaning the gradient descent ran out of
#: iterations before the gradient was small enough.
STATUS_NOT_CONVERGED = -99

KERNELS = ("Gaussian", "logistic", "uniform", "parabolic", "triangular")


def _kernel_cdf(u: np.ndarray, kernel: str) -> np.ndarray:
    """Integrated kernel ``Kbar(u) = int_{-inf}^{u} K(v) dv``."""
    if kernel == "Gaussian":
        return ndtr(u)
    if kernel == "logistic":
        return expit(u)
    # For the compactly supported kernels the CDF is a polynomial *only* inside
    # [-1, 1].  Do not lean on np.clip to handle the tails: these polynomials
    # are non-monotone outside the support and run the wrong way, so e.g. the
    # parabolic form gives -4 at u=3 and would clip to 0 where the answer is 1.
    # That silently inverts the gradient for every residual larger than h.
    inside = np.abs(u) <= 1.0
    if kernel == "uniform":
        return np.where(inside, 0.5 * (u + 1.0), (u > 0).astype(float))
    if kernel == "parabolic":
        # K(v) = 0.75 (1 - v^2) on [-1, 1]
        return np.where(
            inside, 0.75 * u - 0.25 * np.clip(u, -1.0, 1.0) ** 3 + 0.5,
            (u > 0).astype(float),
        )
    if kernel == "triangular":
        # K(v) = 1 - |v| on [-1, 1]
        uc = np.clip(u, -1.0, 1.0)
        return np.where(
            inside, 0.5 + uc - 0.5 * np.abs(uc) * uc, (u > 0).astype(float)
        )
    raise ValueError(f"Unknown kernel {kernel!r}. Choose one of {KERNELS}.")


def _kernel_pdf(u: np.ndarray, kernel: str) -> np.ndarray:
    """Kernel density ``K(u)``.  Needed for the Hessian and hence inference."""
    if kernel == "Gaussian":
        return np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)
    if kernel == "logistic":
        e = expit(u)
        return e * (1.0 - e)
    inside = np.abs(u) <= 1.0
    if kernel == "uniform":
        return np.where(inside, 0.5, 0.0)
    if kernel == "parabolic":
        return np.where(inside, 0.75 * (1.0 - u**2), 0.0)
    if kernel == "triangular":
        return np.where(inside, 1.0 - np.abs(u), 0.0)
    raise ValueError(f"Unknown kernel {kernel!r}. Choose one of {KERNELS}.")


def _kernel_absdev(t: np.ndarray, kernel: str) -> np.ndarray:
    """``G(t) = int |t - v| K(v) dv``.

    Used only to evaluate the smoothed loss, which the line search needs.
    Derived by hand from each kernel and checked against numerical quadrature
    to 2e-14 over 45 (kernel, t) pairs -- see tests/test_solver_conquer.py.
    """
    a = np.abs(t)
    if kernel == "Gaussian":
        return t * (2.0 * ndtr(t) - 1.0) + 2.0 * np.exp(-0.5 * t**2) / np.sqrt(2.0 * np.pi)
    if kernel == "logistic":
        # |t| + 2 log(1 + e^-|t|), written this way to avoid overflow.
        return a + 2.0 * np.log1p(np.exp(-a))
    if kernel == "uniform":
        return np.where(a <= 1.0, 0.5 * t**2 + 0.5, a)
    if kernel == "parabolic":
        return np.where(a <= 1.0, 0.75 * t**2 - 0.125 * t**4 + 0.375, a)
    if kernel == "triangular":
        return np.where(a <= 1.0, t**2 - a**3 / 3.0 + 1.0 / 3.0, a)
    raise ValueError(f"Unknown kernel {kernel!r}. Choose one of {KERNELS}.")


def smoothed_loss(
    X: np.ndarray, y: np.ndarray, beta: np.ndarray, tau: float, h: float, kernel: str
) -> float:
    """``L_h(beta) = (1/n) sum_i (rho_tau * K_h)(r_i)``.

    Uses ``rho_tau(u) = (tau - 1/2) u + |u|/2``, so the convolution splits into
    a linear part and ``(h/2) G(u/h)``.
    """
    r = y - X @ beta
    return float(np.mean((tau - 0.5) * r + 0.5 * h * _kernel_absdev(r / h, kernel)))


def default_bandwidth(n: int, p: int) -> float:
    """``h = ((log(n) + p) / n) ** 0.4``.

    This is the R ``conquer`` package's default, verified against it to six
    decimals across n in {200 .. 10000} and p in {2, 5, 10}.  ``p`` is the
    number of covariates *excluding* the intercept, matching R's convention.
    """
    return float(((np.log(n) + p) / n) ** 0.4)


def smoothed_gradient(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    tau: float,
    h: float,
    kernel: str,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """``grad L_h(beta) = (1/n) X' [Kbar(-r/h) - tau]``.

    ``weights`` reweights the per-observation scores, which is what the
    multiplier bootstrap needs.
    """
    r = y - X @ beta
    score = _kernel_cdf(-r / h, kernel) - tau
    if weights is not None:
        score = score * weights
    return X.T @ score / X.shape[0]


def smoothed_hessian(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    h: float,
    kernel: str,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """``H = (1/(n h)) X' diag(K(r/h)) X``, the Hessian of the smoothed loss."""
    r = y - X @ beta
    w = _kernel_pdf(r / h, kernel)
    if weights is not None:
        w = w * weights
    return (X.T * w) @ X / (X.shape[0] * h)


class ConquerSolver(BaseSolver):
    """Convolution-smoothed quantile regression by Barzilai-Borwein descent.

    Parameters
    ----------
    kernel : str
        One of ``"Gaussian"``, ``"logistic"``, ``"uniform"``, ``"parabolic"``,
        ``"triangular"``.
    bandwidth : float or None
        Smoothing bandwidth ``h``.  ``None`` uses :func:`default_bandwidth`.
        Larger ``h`` means more smoothing: lower variance, more bias.
    tol : float
        Convergence tolerance on the max-norm of the coefficient update.
    max_iter : int
        Iteration budget.  Exhausting it is reported, never silent.
    step_max : float
        Upper bound on the Barzilai-Borwein step, mirroring R conquer's
        ``stepBounded=TRUE`` default, which keeps a badly scaled first step
        from throwing the iterate far away from the optimum.

    Notes
    -----
    The minimiser is unique for a strictly positive kernel, so the answer does
    not depend on how it is found -- the descent scheme affects speed only.
    """

    def __init__(
        self,
        kernel: str = "Gaussian",
        bandwidth: float | None = None,
        tol: float = 1e-8,
        max_iter: int = 5000,
        step_max: float = 100.0,
    ) -> None:
        if kernel not in KERNELS:
            raise ValueError(f"Unknown kernel {kernel!r}. Choose one of {KERNELS}.")
        if bandwidth is not None and not bandwidth > 0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth}.")
        if not tol > 0:
            raise ValueError(f"tol must be positive, got {tol}.")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}.")
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.tol = tol
        self.max_iter = max_iter
        self.step_max = step_max

    def _resolve_bandwidth(self, X: np.ndarray, **kwargs: Any) -> float:
        h = kwargs.get("bandwidth", self.bandwidth)
        if h is not None:
            if not h > 0:
                raise ValueError(f"bandwidth must be positive, got {h}.")
            return float(h)
        n, p = X.shape
        # R conquer counts covariates without the intercept.  Detect a constant
        # column so an X that already carries one gets the same h as R would.
        n_cov = p - 1 if _has_intercept_column(X) else p
        return default_bandwidth(n, max(n_cov, 1))

    def _solve_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
        **kwargs: Any,
    ) -> SolverResult:
        n, p = X.shape
        kernel = kwargs.get("kernel", self.kernel)
        tol = float(kwargs.get("tol", self.tol))
        max_iter = int(kwargs.get("max_iter", self.max_iter))
        step_max = float(kwargs.get("step_max", self.step_max))
        h = self._resolve_bandwidth(X, **kwargs)

        beta, n_iter, grad_norm = _bb_descent(
            X, y, tau, h, kernel, tol, max_iter, step_max
        )

        converged = n_iter < max_iter
        status = 0
        if not converged:
            status = STATUS_NOT_CONVERGED
            warnings.warn(
                f"conquer did not converge in {max_iter} iterations "
                f"(final gradient norm {grad_norm:.3e}); the returned "
                "coefficients may not minimise the smoothed loss.",
                stacklevel=2,
            )

        residuals = y - X @ beta
        pos = np.maximum(residuals, 0.0)
        neg = np.maximum(-residuals, 0.0)
        # The pinball loss of the returned residuals, matching the contract every
        # other solver follows.  Note this is *not* the objective conquer
        # minimises -- that is the smoothed loss, available via smoothed_loss().
        obj = tau * np.sum(pos) + (1.0 - tau) * np.sum(neg)

        info: dict[str, Any] = {
            "bandwidth": h,
            "kernel": kernel,
            "gradient_norm": grad_norm,
            "converged": converged,
            "smoothed": True,
        }

        ci = kwargs.get("ci")
        if ci:
            info.update(
                confidence_intervals(
                    X,
                    y,
                    beta,
                    tau,
                    h,
                    kernel,
                    ci=ci if isinstance(ci, str) else "both",
                    alpha=float(kwargs.get("alpha", 0.05)),
                    n_boot=int(kwargs.get("n_boot", 1000)),
                    random_state=kwargs.get("random_state"),
                )
            )

        return SolverResult(
            coefficients=beta,
            residuals=residuals,
            dual_solution=None,
            objective_value=obj,
            status=status,
            iterations=n_iter,
            solver_info=info,
        )


def _has_intercept_column(X: np.ndarray) -> bool:
    """True if some column is constant and non-zero."""
    if X.shape[0] == 0:
        return False
    first = X[0]
    constant = np.all(first == X, axis=0)
    return bool(np.any(constant & (first != 0.0)))


def _bb_descent(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    h: float,
    kernel: str,
    tol: float,
    max_iter: int,
    step_max: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float]:
    """Minimise the smoothed loss.

    Starts from least squares and takes one Newton step -- the Hessian is only
    ``p x p`` and this makes the first Barzilai-Borwein step size well scaled,
    which is where a bare BB iteration is most fragile -- then switches to BB,
    whose per-iteration cost is ``O(np)`` rather than ``O(np^2)``.
    """
    n, p = X.shape
    if weights is None:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        sw = np.sqrt(weights)
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    grad = smoothed_gradient(X, y, beta, tau, h, kernel, weights)

    # One Newton step to get onto a sensible scale.
    H = smoothed_hessian(X, y, beta, h, kernel, weights)
    try:
        step = np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        step = np.linalg.lstsq(H, grad, rcond=None)[0]
    if not np.all(np.isfinite(step)):
        step = grad
    beta_new = beta - step

    # A conservative fallback step from the curvature at the starting point.
    eig = np.linalg.eigvalsh(H) if np.all(np.isfinite(H)) else np.array([1.0])
    lam = float(eig.max()) if eig.size and eig.max() > 0 else 1.0
    alpha_safe = 1.0 / lam
    alpha = alpha_safe

    n_iter = 1
    while n_iter < max_iter:
        grad_new = smoothed_gradient(X, y, beta_new, tau, h, kernel, weights)
        delta_b = beta_new - beta
        delta_g = grad_new - grad
        if np.max(np.abs(delta_b)) < tol:
            beta, grad = beta_new, grad_new
            break

        # A compactly supported kernel makes the gradient locally constant,
        # so delta_g can vanish and the Barzilai-Borwein ratio carries no
        # curvature information.  Reuse the last usable step rather than
        # jumping to step_max, which throws the iterate away from the optimum
        # and never recovers.
        denom = float(delta_b @ delta_g)
        if denom > 0:
            alpha = float(delta_b @ delta_b) / denom
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = alpha_safe
        alpha = min(alpha, step_max)

        beta, grad = beta_new, grad_new
        beta_new = beta - alpha * grad
        n_iter += 1
    else:
        beta, grad = beta_new, smoothed_gradient(
            X, y, beta_new, tau, h, kernel, weights
        )

    return beta, n_iter, float(np.max(np.abs(grad)))


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────

def asymptotic_cov(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    tau: float,
    h: float,
    kernel: str,
) -> np.ndarray:
    """Sandwich covariance ``H^-1 V H^-1 / n`` for the smoothed estimator.

    ``H`` is the Hessian of the smoothed loss and ``V`` the empirical outer
    product of the per-observation smoothed scores,

        V = (1/n) sum_i [Kbar(-r_i/h) - tau]^2 x_i x_i'

    Note ``V`` is the *empirical* score covariance, not the ``tau (1 - tau)``
    approximation that the unsmoothed theory suggests -- using the latter
    inflates the standard errors by 6-10% here.  Verified against R conquer's
    ``asyCI`` to a ratio of 1.00000 on every coefficient.
    """
    n = X.shape[0]
    r = y - X @ beta
    score = _kernel_cdf(-r / h, kernel) - tau
    H = smoothed_hessian(X, y, beta, h, kernel)
    V = (X.T * score**2) @ X / n
    H_inv = np.linalg.inv(H)
    return H_inv @ V @ H_inv / n


def multiplier_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    h: float,
    kernel: str,
    n_boot: int = 1000,
    random_state: int | np.random.RandomState | None = None,
    tol: float = 1e-8,
    max_iter: int = 5000,
    step_max: float = 100.0,
) -> np.ndarray:
    """Multiplier (weighted) bootstrap: ``(n_boot, p)`` array of refits.

    Each replicate reweights the per-observation losses by i.i.d. Exp(1)
    draws, which have mean and variance one.  Unlike resampling with
    replacement this keeps every observation in play, so no replicate can
    produce a rank-deficient design.

    Replicates that fail to converge are dropped rather than silently folded
    into the quantiles; the caller is warned if any were.
    """
    from pinball.linear._bootstrap import _ensure_rng

    rng = _ensure_rng(random_state)
    n = X.shape[0]
    draws = []
    n_failed = 0
    for _ in range(int(n_boot)):
        w = rng.exponential(1.0, size=n)
        b, n_iter, _ = _bb_descent(
            X, y, tau, h, kernel, tol, max_iter, step_max, weights=w
        )
        if n_iter < max_iter and np.all(np.isfinite(b)):
            draws.append(b)
        else:
            n_failed += 1
    if n_failed:
        warnings.warn(
            f"{n_failed} of {n_boot} bootstrap replicates did not converge and "
            "were discarded; the intervals are based on the rest.",
            stacklevel=2,
        )
    if not draws:
        raise RuntimeError("every bootstrap replicate failed to converge")
    return np.asarray(draws)


def confidence_intervals(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    tau: float,
    h: float,
    kernel: str,
    ci: str = "both",
    alpha: float = 0.05,
    n_boot: int = 1000,
    random_state: int | np.random.RandomState | None = None,
) -> dict[str, np.ndarray]:
    """Confidence intervals for a fitted conquer model.

    Mirrors the four interval types R's ``conquer`` reports:

    ``asymptotic``
        ``beta +/- z * sqrt(diag(asymptotic_cov))``.
    ``percentile``
        empirical quantiles of the bootstrap draws.
    ``pivotal``
        ``2 beta - `` the reversed percentile bounds (the "basic" bootstrap).
    ``normal``
        ``beta +/- z *`` the bootstrap standard deviation.

    Parameters
    ----------
    ci : {"asymptotic", "bootstrap", "both"}
        ``"bootstrap"`` returns the three resampling intervals, ``"asymptotic"``
        the sandwich one, ``"both"`` all four.
    """
    from scipy.stats import norm

    if ci not in ("asymptotic", "bootstrap", "both"):
        raise ValueError(
            f"ci must be 'asymptotic', 'bootstrap' or 'both', got {ci!r}."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    z = float(norm.ppf(1.0 - alpha / 2.0))
    out: dict[str, np.ndarray] = {}

    if ci in ("asymptotic", "both"):
        se = np.sqrt(np.diag(asymptotic_cov(X, y, beta, tau, h, kernel)))
        out["asymptotic"] = np.column_stack([beta - z * se, beta + z * se])
        out["asymptotic_se"] = se

    if ci in ("bootstrap", "both"):
        draws = multiplier_bootstrap(
            X, y, tau, h, kernel, n_boot=n_boot, random_state=random_state
        )
        lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        out["percentile"] = np.column_stack([lo, hi])
        # Basic/pivotal: reflect the percentile interval through the estimate.
        out["pivotal"] = np.column_stack([2.0 * beta - hi, 2.0 * beta - lo])
        sd = draws.std(axis=0, ddof=1)
        out["normal"] = np.column_stack([beta - z * sd, beta + z * sd])
        out["bootstrap_se"] = sd
        out["bootstrap_draws"] = draws

    return out
