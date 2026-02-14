"""Methods for computing bandwidth for sparsity / density estimation.

References
----------
.. [1] Bofinger, E. (1975). "Estimation of a density function using order
       statistics." *Australian Journal of Statistics* 17: 1–17.
.. [2] Chamberlain, G. (1994). "Quantile regression, censoring, and the
       structure of wages." In *Advances in Econometrics*, Vol. 1.
.. [3] Hall, P. and Sheather, S. (1988). "On the distribution of the
       Studentized quantile." *JRSS-B* 50: 381–391.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def hall_sheather(n: float, q: float, alpha: float = 0.05) -> float:
    """Hall-Sheather (1988) bandwidth.

    Parameters
    ----------
    n : int or float
        Number of observations.
    q : float
        Quantile level in (0, 1).
    alpha : float
        Significance level.

    Returns
    -------
    float
    """
    z = norm.ppf(q)
    num = 1.5 * norm.pdf(z) ** 2.0
    den = 2.0 * z ** 2.0 + 1.0
    h = n ** (-1.0 / 3) * norm.ppf(1.0 - alpha / 2.0) ** (2.0 / 3) * (num / den) ** (1.0 / 3)
    return h


def bofinger(n: float, q: float) -> float:
    """Bofinger (1975) bandwidth.

    Parameters
    ----------
    n : int or float
        Number of observations.
    q : float
        Quantile level in (0, 1).

    Returns
    -------
    float
    """
    num = 9.0 / 2 * norm.pdf(2 * norm.ppf(q)) ** 4
    den = (2 * norm.ppf(q) ** 2 + 1) ** 2
    h = n ** (-1.0 / 5) * (num / den) ** (1.0 / 5)
    return h


def chamberlain(n: float, q: float, alpha: float = 0.05) -> float:
    """Chamberlain (1994) bandwidth.

    Parameters
    ----------
    n : int or float
        Number of observations.
    q : float
        Quantile level in (0, 1).
    alpha : float
        Significance level.

    Returns
    -------
    float
    """
    return norm.ppf(1 - alpha / 2) * np.sqrt(q * (1 - q) / n)
