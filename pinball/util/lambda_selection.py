"""Automatic penalty-parameter selection for penalised quantile regression.

References
----------
.. [1] Belloni, A. and Chernozhukov, V. (2011). "ℓ1-penalized quantile
       regression in high-dimensional sparse models." *Annals of Statistics*.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def lambda_hat_bcv(
    X: np.ndarray,
    tau: float,
    c: float = 1.0,
    alpha: float = 0.05,
) -> float:
    """Belloni-Chernozhukov default λ for L1-penalised quantile regression.

    This is a *pivotal* choice that does not depend on unknown nuisance
    parameters and achieves near-oracle performance under approximate
    sparsity.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix.
    tau : float
        Quantile level.
    c : float
        Scaling constant (default 1.0; use larger values for
        conservative selection).
    alpha : float
        Nominal level controlling the probability bound.

    Returns
    -------
    float
        The recommended λ.
    """
    n, p = X.shape
    # Belloni-Chernozhukov (2011) Theorem 1
    sigma = np.sqrt(tau * (1 - tau))
    lam = c * sigma * norm.ppf(1 - alpha / (2 * p)) / np.sqrt(n)
    return lam
