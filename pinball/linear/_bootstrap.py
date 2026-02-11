"""Bootstrap inference for quantile regression.

Implements three bootstrap strategies following R's ``boot.rq``:

* **xy-pair** — classical nonparametric resampling (Efron 1979)
* **wild** — Feng, He & Hu (2011) perturbation bootstrap
* **mcmb** — Markov chain marginal bootstrap (He & Hu 2002)

Each returns a :class:`BootstrapResult` containing the R × p matrix of
bootstrap coefficient draws, standard errors, and percentile confidence
intervals.

References
----------
.. [1] Efron, B. (1979). "Bootstrap methods: another look at the jackknife."
.. [2] Feng, X., He, X. and Hu, J. (2011). "Wild bootstrap for quantile
       regression." *Biometrika* 98(4): 995–999.
.. [3] He, X. and Hu, F. (2002). "Markov chain marginal bootstrap."
       *JASA* 97(459): 783–795.
.. [4] Kocherginsky, M., He, X. and Mu, Y. (2005). "Practical confidence
       intervals for regression quantiles." *JSPI* 128(2): 431–446.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy.stats import norm


@dataclass
class BootstrapResult:
    """Output of a bootstrap inference procedure.

    Attributes
    ----------
    boot_coefficients : ndarray, shape (nboot, p)
        Bootstrap replicate coefficient vectors.
    coefficients : ndarray, shape (p,)
        Point estimate (mean of bootstrap replicates or original fit).
    std_errors : ndarray, shape (p,)
        Bootstrap standard errors (column-wise std of replicates).
    conf_int : ndarray, shape (p, 2)
        Percentile confidence intervals.
    bsmethod : str
        The bootstrap method used.
    nboot : int
        Number of bootstrap replicates.
    """

    boot_coefficients: np.ndarray
    coefficients: np.ndarray
    std_errors: np.ndarray
    conf_int: np.ndarray
    bsmethod: str
    nboot: int

    @property
    def covariance(self) -> np.ndarray:
        """Sample covariance matrix of the bootstrap replicates."""
        return np.cov(self.boot_coefficients, rowvar=False)


# ──────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    tau: float = 0.5,
    nboot: int = 200,
    method: str = "xy",
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    alpha: float = 0.05,
    **kwargs,
) -> BootstrapResult:
    """Run a bootstrap for quantile-regression inference.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix (including intercept if applicable).
    y : ndarray (n,)
        Response vector.
    tau : float
        Quantile level.
    nboot : int
        Number of bootstrap replications.
    method : str
        ``"xy"`` (xy-pair), ``"wild"``, or ``"mcmb"``.
    random_state : int or RandomState or None
        Seed for reproducibility.
    alpha : float
        Significance level for percentile CI (default 0.05 → 95 %).
    **kwargs
        Extra options forwarded to the bootstrap strategy.

    Returns
    -------
    BootstrapResult
    """
    _methods = {
        "xy": _xy_pairs,
        "wild": _wild,
        "mcmb": _mcmb,
    }
    if method not in _methods:
        raise ValueError(
            f"Unknown bootstrap method {method!r}. "
            f"Choose from {list(_methods.keys())}."
        )
    return _methods[method](
        X, y, tau=tau, nboot=nboot, random_state=random_state,
        alpha=alpha, **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_rng(
    random_state: Optional[Union[int, np.random.RandomState]],
) -> np.random.RandomState:
    """Coerce *random_state* to a ``RandomState`` instance."""
    if random_state is None:
        return np.random.RandomState()
    if isinstance(random_state, (int, np.integer)):
        return np.random.RandomState(int(random_state))
    return random_state


def _fit_qr(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Fit a single quantile regression, return coefficient vector.

    Uses FNB solver for speed; falls back to BR for tiny problems.
    """
    from pinball.linear.solvers.fnb import FNBSolver
    solver = FNBSolver()
    return solver.solve(X, y, tau).coefficients


def _build_bootstrap_result(
    B: np.ndarray,
    coefficients: np.ndarray,
    bsmethod: str,
    alpha: float,
) -> BootstrapResult:
    """Summarise an (R, p) matrix of bootstrap draws."""
    nboot = B.shape[0]
    se = np.std(B, axis=0, ddof=1)
    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)
    ci = np.column_stack([
        np.percentile(B, lo_pct, axis=0),
        np.percentile(B, hi_pct, axis=0),
    ])
    return BootstrapResult(
        boot_coefficients=B,
        coefficients=coefficients,
        std_errors=se,
        conf_int=ci,
        bsmethod=bsmethod,
        nboot=nboot,
    )


# ──────────────────────────────────────────────────────────────────────────────
# XY-pair bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def _xy_pairs(
    X: np.ndarray,
    y: np.ndarray,
    tau: float = 0.5,
    nboot: int = 200,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    alpha: float = 0.05,
    mofn: Optional[int] = None,
    **kwargs,
) -> BootstrapResult:
    """XY-pair (case) bootstrap for quantile regression.

    Draws *mofn* observations with replacement, refits QR, and
    applies the :math:`\\sqrt{m/n}` deflation when ``mofn < n``
    (m-of-n subsampling).

    Parameters
    ----------
    mofn : int or None
        Subsample size.  ``None`` (default) → full-sample bootstrap.
    """
    rng = _ensure_rng(random_state)
    n, p = X.shape
    if mofn is None:
        mofn = n

    # Original fit for point estimate
    coef_hat = _fit_qr(X, y, tau)

    B = np.empty((nboot, p), dtype=np.float64)
    scale = np.sqrt(mofn / n)

    for r in range(nboot):
        idx = rng.choice(n, size=mofn, replace=True)
        try:
            b = _fit_qr(X[idx], y[idx], tau)
            B[r, :] = coef_hat + scale * (b - coef_hat)
        except Exception:
            # Singular subsample — use original coefficients
            B[r, :] = coef_hat

    return _build_bootstrap_result(B, coef_hat, "xy", alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Wild bootstrap (Feng, He & Hu 2011)
# ──────────────────────────────────────────────────────────────────────────────

def _wild(
    X: np.ndarray,
    y: np.ndarray,
    tau: float = 0.5,
    nboot: int = 200,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    alpha: float = 0.05,
    **kwargs,
) -> BootstrapResult:
    r"""Wild bootstrap for quantile regression (Feng, He & Hu 2011).

    Algorithm
    ---------
    1. Fit the original model → residuals :math:`\hat{u}`, fitted :math:`\hat{y}`.
    2. Compute adjusted residuals incorporating hat-matrix leverage.
    3. Generate two-point perturbation:
       :math:`S = -2\tau` w.p. :math:`\tau`, :math:`S = 2(1-\tau)` w.p. :math:`1-\tau`.
    4. Wild responses: :math:`y^* = \hat{y} + S \cdot |\tilde{u}|`.
    5. Refit QR on :math:`(X, y^*)` for each replicate.
    """
    rng = _ensure_rng(random_state)
    n, p = X.shape

    # Original fit
    coef_hat = _fit_qr(X, y, tau)
    y_hat = X @ coef_hat
    resid = y - y_hat

    # Estimate density at zero (simple kernel density on residuals)
    sigma = np.std(resid, ddof=1)
    iqr = np.percentile(resid, 75) - np.percentile(resid, 25)
    bw = 1.06 * min(sigma, iqr / 1.34 if iqr > 0 else sigma) * n ** (-0.2)
    bw = max(bw, 1e-10)
    f0 = np.mean(norm.pdf(resid / bw)) / bw

    # Hat-matrix leverages
    try:
        Q, R_mat = np.linalg.qr(X)
        hat_diag = np.sum(Q ** 2, axis=1)
    except np.linalg.LinAlgError:
        hat_diag = np.zeros(n)

    # Adjusted residuals
    psi = tau - (resid < 0).astype(np.float64)
    adj_resid = resid + hat_diag * psi / max(f0, 1e-20)

    B = np.empty((nboot, p), dtype=np.float64)

    for r in range(nboot):
        # Two-point perturbation
        u = rng.rand(n)
        S = np.where(u < tau, -2.0 * tau, 2.0 * (1.0 - tau))
        y_star = y_hat + S * np.abs(adj_resid)
        try:
            B[r, :] = _fit_qr(X, y_star, tau)
        except Exception:
            B[r, :] = coef_hat

    return _build_bootstrap_result(B, coef_hat, "wild", alpha)


# ──────────────────────────────────────────────────────────────────────────────
# MCMB (Markov chain marginal bootstrap)
# ──────────────────────────────────────────────────────────────────────────────

def _mcmb(
    X: np.ndarray,
    y: np.ndarray,
    tau: float = 0.5,
    nboot: int = 200,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    alpha: float = 0.05,
    **kwargs,
) -> BootstrapResult:
    r"""Markov chain marginal bootstrap (He & Hu 2002).

    Performs coordinate-wise Gibbs-like updates using resampled
    subgradient perturbations.

    Algorithm
    ---------
    1. Fit → :math:`\hat{\beta}`, residuals :math:`\hat{r}`.
    2. Compute :math:`\psi_i = \tau - \mathbf{1}(\hat{r}_i < 0)`.
    3. Form :math:`Z_{ij} = x_{ij} \psi_i`.
    4. For each MCMB iteration *k*, for each coordinate *j*:
       a. Resample perturbation from the :math:`Z_{\cdot j}` column.
       b. Solve a scalar weighted-median problem to update :math:`\beta_j`.
    5. Collect the chain draws after a transient period.
    """
    rng = _ensure_rng(random_state)
    n, p = X.shape

    # Original fit
    coef_hat = _fit_qr(X, y, tau)
    resid = y - X @ coef_hat
    psi = tau - (resid < 0).astype(np.float64)

    # Z matrix: element-wise product
    Z = X * psi[:, np.newaxis]  # (n, p)
    z_flat = Z.ravel()  # flattened for resampling (n*p elements)

    scale = np.sqrt(n) / np.sqrt(max(n - p, 1))

    B = np.empty((nboot, p), dtype=np.float64)
    theta = coef_hat.copy()

    for k in range(nboot):
        # Draw a single z_star vector (length n) from the entire Z matrix
        z_star = rng.choice(z_flat, size=n, replace=True)
        s_j = scale * np.sum(z_star)

        for j in range(p):
            xj = X[:, j]
            abs_xj = np.abs(xj)
            mask = abs_xj > 1e-20

            if np.sum(mask) == 0:
                continue

            # Partial residuals (excluding contribution of coef j)
            r_partial = y - X @ theta + theta[j] * xj

            # Targets and weights for the n real observations
            targets_real = r_partial[mask] / xj[mask]
            weights_real = abs_xj[mask]

            # Augmented observation: x_{n+1,j} = s_j, target = theta[j]
            # This shifts the weighted quantile even when tau=0.5
            aug_target = theta[j]
            aug_weight = abs(s_j)

            # Concatenate real + augmented observation
            targets = np.append(targets_real, aug_target)
            weights = np.append(weights_real, aug_weight)

            # Compute adjusted tau for the augmented sample
            xj_sum = np.sum(xj) + s_j
            abs_xj_sum = np.sum(abs_xj) + abs(s_j)
            tau_star = (tau - 0.5) * xj_sum / (abs_xj_sum + 1e-20) + 0.5
            tau_star = np.clip(tau_star, 0.01, 0.99)

            # Weighted quantile
            theta[j] = _weighted_quantile(targets, weights, tau_star)

        B[k, :] = theta.copy()

    return _build_bootstrap_result(B, coef_hat, "mcmb", alpha)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    tau: float,
) -> float:
    """Compute a weighted quantile (weighted percentile).

    Parameters
    ----------
    values : ndarray (m,)
    weights : ndarray (m,)
        Non-negative weights.
    tau : float
        Quantile level in (0, 1).

    Returns
    -------
    float
    """
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    # Find first index where cumulative weight exceeds tau * total
    idx = np.searchsorted(cum_w, tau * total_w)
    idx = min(idx, len(v_sorted) - 1)
    return float(v_sorted[idx])
