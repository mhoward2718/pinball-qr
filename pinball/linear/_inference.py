"""Inference (standard errors, confidence intervals) for quantile regression.

Provides multiple methods for computing standard errors / confidence
intervals after fitting a quantile-regression model, following the
conventions of R's ``summary.rq``.

SE methods
----------
``"rank"``   — rank-inversion intervals (Koenker 1994 / Koenker-Machado 1999)
``"iid"``    — IID (Koenker-Bassett 1978) sandwich
``"nid"``    — Huber sandwich with local sparsity (default for n >= 1001)
``"ker"``    — Powell (1991) kernel sandwich
``"boot"``   — bootstrap (see :mod:`pinball._bootstrap`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm, t as student_t

from pinball.util.bandwidth import hall_sheather


@dataclass
class InferenceResult:
    """Summary table for a fitted quantile-regression model.

    Attributes
    ----------
    coefficients : ndarray, shape (p,)
    std_errors : ndarray, shape (p,)
    t_statistics : ndarray, shape (p,)
    p_values : ndarray, shape (p,)
    conf_int : ndarray, shape (p, 2)
        Lower / upper confidence bounds.
    se_method : str
    """

    coefficients: np.ndarray
    std_errors: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    conf_int: np.ndarray
    se_method: str
    feature_names: Optional[list[str]] = None

    def __repr__(self) -> str:
        lines = [f"InferenceResult(se_method={self.se_method!r})"]
        header = f"{'':>12s} {'Coef':>10s} {'Std Err':>10s} {'t':>10s} {'P>|t|':>10s} {'[0.025':>10s} {'0.975]':>10s}"
        lines.append(header)
        names = self.feature_names or [f"x{i}" for i in range(len(self.coefficients))]
        for i, name in enumerate(names):
            lines.append(
                f"{name:>12s} {self.coefficients[i]:10.4f} "
                f"{self.std_errors[i]:10.4f} {self.t_statistics[i]:10.4f} "
                f"{self.p_values[i]:10.4f} {self.conf_int[i, 0]:10.4f} "
                f"{self.conf_int[i, 1]:10.4f}"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# SE methods
# ──────────────────────────────────────────────────────────────────────────────

def _se_iid(
    X: np.ndarray,
    residuals: np.ndarray,
    tau: float,
    alpha: float = 0.05,
) -> InferenceResult:
    """IID (Koenker-Bassett 1978) covariance.

    Var(β̂) = τ(1−τ) (X'X)^{-1} / f(F^{-1}(τ))^2

    The sparsity function f(F^{-1}(τ)) is estimated via Hall-Sheather
    bandwidth and a kernel density estimate of the residuals.
    """
    n, p = X.shape
    h = hall_sheather(n, tau, alpha)
    sorted_resid = np.sort(residuals)
    # Bandwidth indices
    lo = max(int(np.floor(n * (tau - h))), 0)
    hi = min(int(np.ceil(n * (tau + h))), n - 1)
    # Sparsity estimate
    fhat = (2 * h) / (sorted_resid[hi] - sorted_resid[lo] + 1e-20)

    XtX_inv = np.linalg.inv(X.T @ X)
    cov = tau * (1 - tau) * XtX_inv / (fhat ** 2)
    se = np.sqrt(np.diag(cov))

    return _build_result(X, residuals, tau, se, "iid", alpha)


def _se_nid(
    X: np.ndarray,
    residuals: np.ndarray,
    tau: float,
    alpha: float = 0.05,
) -> InferenceResult:
    """Huber sandwich (non-IID) with local sparsity estimate."""
    n, p = X.shape
    h = hall_sheather(n, tau, alpha)

    # Sparsity: local density from kernel on residuals
    sorted_resid = np.sort(residuals)
    lo = max(int(np.floor(n * (tau - h))), 0)
    hi = min(int(np.ceil(n * (tau + h))), n - 1)
    fhat = (2 * h) / (sorted_resid[hi] - sorted_resid[lo] + 1e-20)

    # Huber sandwich: (X'X)^{-1} X' D X (X'X)^{-1}
    # where D = diag(psi_i^2), psi_i = tau - I(resid_i < 0)
    psi = tau - (residuals < 0).astype(np.float64)
    XtX_inv = np.linalg.inv(X.T @ X)
    # Efficient meat: X.T @ diag(psi^2) @ X = (psi[:,None] * X).T @ (psi[:,None] * X)
    Xw = psi[:, np.newaxis] * X
    meat = Xw.T @ Xw
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))

    return _build_result(X, residuals, tau, se, "nid", alpha)


def _se_ker(
    X: np.ndarray,
    residuals: np.ndarray,
    tau: float,
    alpha: float = 0.05,
) -> InferenceResult:
    r"""Powell (1991) kernel sandwich standard errors.

    Uses a Gaussian kernel on the residuals to estimate per-observation
    conditional density at zero, then forms the sandwich covariance:

    .. math::

        \hat{H}^{-1} = \bigl(\sum_i \hat{f}_i\, x_i x_i'\bigr)^{-1}

        \widehat{\mathrm{Cov}}(\hat\beta) =
            \tau(1-\tau)\;\hat{H}^{-1}(X'X)\hat{H}^{-1}

    Bandwidth follows Silverman's robust rule scaled by the normal
    quantile gap implied by the Hall-Sheather spacing.

    References
    ----------
    .. [1] Powell, J. (1991). "Estimation of monotonic regression models
           under quantile restrictions." *Nonparametric and Semiparametric
           Methods in Econometrics*, Cambridge Univ. Press.
    """
    n, p = X.shape
    h0 = hall_sheather(n, tau, alpha)

    # Clamp so tau ± h0 stays in (0, 1)
    h0 = min(h0, tau - 1e-6, 1 - tau - 1e-6)

    # Residual-scale bandwidth (Silverman's robust rule)
    z_hi = norm.ppf(min(tau + h0, 1 - 1e-10))
    z_lo = norm.ppf(max(tau - h0, 1e-10))
    sigma_hat = np.std(residuals, ddof=1)
    iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
    robust_scale = min(sigma_hat, iqr / 1.34) if iqr > 0 else sigma_hat
    bw = (z_hi - z_lo) * max(robust_scale, 1e-20)

    # Gaussian kernel density of residuals evaluated at zero
    fhat = norm.pdf(residuals / bw) / bw  # shape (n,)

    # Hessian: H = X' diag(fhat) X
    Xf = np.sqrt(fhat)[:, np.newaxis] * X
    H = Xf.T @ Xf
    H_inv = np.linalg.inv(H)

    # Sandwich: tau(1-tau) * H^{-1} X'X H^{-1}
    XtX = X.T @ X
    cov = tau * (1 - tau) * H_inv @ XtX @ H_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return _build_result(X, residuals, tau, se, "ker", alpha)


def _se_rank(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    tau: float,
    alpha: float = 0.05,
    iid: bool = True,
    interp: bool = True,
) -> InferenceResult:
    """Rank-inversion confidence intervals (Koenker 1994).

    Delegates to the BR solver with ``ci=True`` to obtain rank-test
    boundaries, then interpolates to get exact CI endpoints.

    Unlike the other ``_se_*`` functions this method produces confidence
    intervals directly (not via standard errors).  Pseudo-SE are derived
    as half the CI width divided by the critical value.

    Parameters
    ----------
    X, y : ndarray
        Design matrix and response.
    coefficients : ndarray (p,)
        Point estimate from a prior fit.
    tau : float
    alpha : float
    iid : bool
        ``True`` for the simpler IID formula; ``False`` for NID.
    interp : bool
        Linearly interpolate between the two bounding LP solutions.

    References
    ----------
    .. [1] Koenker, R. (1994). "Confidence intervals for regression quantiles."
    .. [2] Koenker, R. and Machado, J.A.F. (1999). *JASA* 94(448).
    """
    from pinball.solvers.br import BRSolver

    n, p = X.shape

    solver = BRSolver(ci=True, iid=iid, alpha=alpha, interp=interp)
    result = solver.solve(X, y, tau)

    ci_raw = result.solver_info["ci"]       # shape (4, p)
    tn_raw = result.solver_info["tnmat"]    # shape (4, p)
    cutoff = result.solver_info["cutoff"]

    ci_out = np.zeros((p, 2), dtype=np.float64)

    for j in range(p):
        ci_lo_vals = ci_raw[:2, j]   # two lower boundary coef values
        tn_lo_vals = tn_raw[:2, j]   # corresponding test statistics
        ci_hi_vals = ci_raw[2:, j]   # two upper boundary coef values
        tn_hi_vals = tn_raw[2:, j]

        if interp:
            # Interpolate lower bound
            ci_out[j, 0] = _interpolate_ci(
                ci_lo_vals[0], ci_lo_vals[1],
                tn_lo_vals[0], tn_lo_vals[1],
                cutoff,
            )
            # Interpolate upper bound
            ci_out[j, 1] = _interpolate_ci(
                ci_hi_vals[0], ci_hi_vals[1],
                tn_hi_vals[0], tn_hi_vals[1],
                cutoff,
            )
        else:
            # No interpolation — take the conservative boundary
            ci_out[j, 0] = ci_lo_vals[1]
            ci_out[j, 1] = ci_hi_vals[0]

    # Derive pseudo-SE from CI half-width
    df = n - p
    if df > 0:
        crit = student_t.ppf(1 - alpha / 2, df)
    else:
        crit = norm.ppf(1 - alpha / 2)
    half_width = (ci_out[:, 1] - ci_out[:, 0]) / 2.0
    se = half_width / max(crit, 1e-10)

    # Build result with rank-based CIs (not SE-derived CIs)
    coef = result.coefficients
    t_stats = coef / (se + 1e-30)
    if df > 0:
        p_vals = 2 * student_t.sf(np.abs(t_stats), df)
    else:
        p_vals = np.full(p, np.nan)

    return InferenceResult(
        coefficients=coef,
        std_errors=se,
        t_statistics=t_stats,
        p_values=p_vals,
        conf_int=ci_out,
        se_method="rank",
    )


def _interpolate_ci(
    beta_a: float,
    beta_b: float,
    tn_a: float,
    tn_b: float,
    cutoff: float,
) -> float:
    """Linearly interpolate between two LP boundary solutions.

    Given two coefficient values (beta_a, beta_b) and their rank-test
    statistics (tn_a, tn_b), find the coefficient value where the
    test statistic equals *cutoff* by linear interpolation.

    Follows R's quantreg ``summary.rq`` interpolation logic.
    """
    tn_a_abs = abs(tn_a)
    tn_b_abs = abs(tn_b)
    denom = abs(tn_a_abs - tn_b_abs)
    if denom < 1e-30:
        return (beta_a + beta_b) / 2.0
    # Linear interpolation: weight by distance to cutoff
    w = (cutoff - tn_b_abs) / denom
    return beta_b + w * (beta_a - beta_b)


def _build_result(
    X: np.ndarray,
    residuals: np.ndarray,
    tau: float,
    se: np.ndarray,
    method: str,
    alpha: float,
) -> InferenceResult:
    """Construct an InferenceResult from standard errors.

    The t-statistics, p-values and CIs are populated later by
    :func:`summary` once the actual coefficients are available.
    """
    p = X.shape[1]
    return InferenceResult(
        coefficients=np.zeros(p),  # placeholder — overwritten by summary()
        std_errors=se,
        t_statistics=np.zeros(p),
        p_values=np.ones(p),
        conf_int=np.zeros((p, 2)),
        se_method=method,
    )


def summary(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    tau: float,
    se: str = "auto",
    alpha: float = 0.05,
    feature_names: Optional[list[str]] = None,
    **kwargs,
) -> InferenceResult:
    """Compute a summary table with standard errors and confidence intervals.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix used in fitting (including intercept column if any).
    y : ndarray (n,)
        Response vector.
    coefficients : ndarray (p,)
        Fitted coefficient vector.
    tau : float
        Quantile level.
    se : str
        Standard-error method: ``"rank"``, ``"iid"``, ``"nid"``, ``"ker"``,
        ``"boot"``, or ``"auto"`` (selects ``"rank"`` for n < 1001,
        ``"nid"`` otherwise).
    alpha : float
        Significance level for confidence intervals.
    feature_names : list of str, optional
        Names for the coefficient table.
    **kwargs
        Extra arguments forwarded to the SE estimator.
        For ``se="rank"``: ``iid`` (bool), ``interp`` (bool).
        For ``se="boot"``: ``nboot`` (int), ``method`` (str),
        ``random_state``.

    Returns
    -------
    InferenceResult
    """
    n, p = X.shape
    residuals = y - X @ coefficients

    if se == "auto":
        se = "rank" if n < 1001 else "nid"

    # Rank-inversion: special path — builds its own InferenceResult
    if se == "rank":
        result = _se_rank(
            X, y, coefficients, tau, alpha,
            iid=kwargs.get("iid", True),
            interp=kwargs.get("interp", True),
        )
        result.feature_names = feature_names
        return result

    # Bootstrap: special path — delegates to _bootstrap module
    if se == "boot":
        from pinball._bootstrap import bootstrap as _boot
        nboot = kwargs.get("nboot", 200)
        bsmethod = kwargs.get("method", "xy")
        random_state = kwargs.get("random_state", None)
        br = _boot(
            X, y, tau=tau, nboot=nboot, method=bsmethod,
            random_state=random_state, alpha=alpha,
        )
        # Build InferenceResult from bootstrap
        t_stats = coefficients / (br.std_errors + 1e-30)
        df = n - p
        if df > 0:
            p_vals = 2 * student_t.sf(np.abs(t_stats), df)
        else:
            p_vals = np.full(p, np.nan)
        return InferenceResult(
            coefficients=coefficients,
            std_errors=br.std_errors,
            t_statistics=t_stats,
            p_values=p_vals,
            conf_int=br.conf_int,
            se_method="boot",
            feature_names=feature_names,
        )

    # Standard sandwich methods: iid, nid, ker
    if se == "iid":
        result = _se_iid(X, residuals, tau, alpha)
    elif se == "nid":
        result = _se_nid(X, residuals, tau, alpha)
    elif se == "ker":
        result = _se_ker(X, residuals, tau, alpha)
    else:
        raise ValueError(
            f"Unknown se method {se!r}. "
            "Choose from 'rank', 'iid', 'nid', 'ker', 'boot', 'auto'."
        )

    # Overwrite with actual coefficients
    result.coefficients = coefficients
    result.feature_names = feature_names

    # Compute t-stats and p-values
    result.t_statistics = coefficients / (result.std_errors + 1e-30)
    df = n - p
    if df > 0:
        result.p_values = 2 * student_t.sf(np.abs(result.t_statistics), df)
    else:
        result.p_values = np.full(p, np.nan)

    # Confidence interval
    if df > 0:
        crit = student_t.ppf(1 - alpha / 2, df)
    else:
        crit = norm.ppf(1 - alpha / 2)
    result.conf_int = np.column_stack([
        coefficients - crit * result.std_errors,
        coefficients + crit * result.std_errors,
    ])

    return result
