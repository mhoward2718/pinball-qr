"""Quantization-based conditional quantile estimator.

Implements the nonparametric conditional quantile estimator of
Charlier, Paindaveine & Saracco (2015) as a scikit-learn--compatible
estimator.

The algorithm:
1. Construct *n_grids* L_p-optimal quantization grids of size *N*
   for the covariate X using CLVQ (``choice_grid``).
2. For each grid, assign every training (X_i, Y_i) to its Voronoi cell
   and compute the sample ``tau``-quantile of Y within each cell.
3. Average the per-grid quantile estimates to obtain the final
   cell-level conditional quantile.
4. At prediction time, assign each new x to its nearest grid cell and
   return the corresponding averaged conditional quantile.

References
----------
.. [1] Charlier, I., Paindaveine, D. and Saracco, J. (2015).
       "Conditional quantile estimation through optimal quantization."
       *JSPI* 156, 14–30.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted, validate_data

from pinball.estimators._base import BaseQuantileEstimator
from pinball.nonparametric.quantization._clvq import choice_grid
from pinball.nonparametric.quantization._voronoi import (
    cell_quantiles,
    predict_quantiles,
    voronoi_assign,
)


class QuantizationQuantileEstimator(BaseQuantileEstimator):
    """Nonparametric conditional quantile estimator via optimal quantization.

    Parameters
    ----------
    tau : float, default 0.5
        Quantile level in (0, 1).
    N : int, default 20
        Number of points (centroids) in the quantization grid.
    n_grids : int, default 50
        Number of independent bootstrap grids.  The final estimate
        averages over all grids (like bagging).
    p : float, default 2
        Exponent of the L_p norm for the CLVQ algorithm.
    random_state : int or None, default None
        Seed for reproducibility.

    Attributes
    ----------
    grid_ : ndarray, shape (N,) or (d, N)
        Averaged optimal quantization grid (centroid of all grids).
    cell_quantiles_ : ndarray, shape (N, 1)
        Averaged conditional-quantile estimates per Voronoi cell.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    def __init__(
        self,
        tau: float = 0.5,
        N: int = 20,
        n_grids: int = 50,
        p: float = 2,
        random_state: int | None = None,
    ):
        self.tau = tau
        self.N = N
        self.n_grids = n_grids
        self.p = p
        self.random_state = random_state

    # ──────────────────────────────────────────────────────────────
    # fit
    # ──────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit the quantization-based conditional quantile estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = validate_data(self, X, y, y_numeric=True)
        n, d = X.shape

        if n < 2:
            raise ValueError(
                f"n_samples = {n} is too small.  Need at least 2 samples."
            )

        # Effective N: never exceed sample size
        N_eff = min(self.N, n)

        alpha = np.array([self.tau])

        # Prepare covariate for CLVQ: 1-D vector or (d, n) matrix
        if d == 1:
            X_clvq = X.ravel()
        else:
            X_clvq = X.T  # (d, n)

        grids = choice_grid(
            X_clvq, N_eff, n_grids=self.n_grids,
            p=self.p, random_state=self.random_state,
        )
        opt_grids = grids["optimal_grid"]
        # opt_grids shape: (N_eff, n_grids) for 1-D, (d, N_eff, n_grids) for d-D

        # Accumulate cell quantile estimates across grids
        n_grids = self.n_grids
        cell_q_accum = np.zeros((N_eff, 1), dtype=np.float64)
        cell_q_count = np.zeros((N_eff, 1), dtype=np.float64)

        for g in range(n_grids):
            if d == 1:
                grid_g = opt_grids[:, g]  # (N_eff,)
            else:
                grid_g = opt_grids[:, :, g]  # (d, N_eff)

            assignments = voronoi_assign(X_clvq, grid_g)
            cq = cell_quantiles(y, assignments, N_eff, alpha)
            # cq: (N_eff, 1) — NaN where no data in cell
            valid = ~np.isnan(cq)
            cell_q_accum[valid] += cq[valid]
            cell_q_count[valid] += 1.0

        # Average grid centroids across all bootstrap grids
        if d == 1:
            avg_grid = np.mean(opt_grids, axis=1)  # (N_eff,)
        else:
            avg_grid = np.mean(opt_grids, axis=2)  # (d, N_eff)

        # Average cell quantiles (NaN where no grid ever had data)
        with np.errstate(invalid="ignore"):
            avg_cell_q = np.where(
                cell_q_count > 0,
                cell_q_accum / cell_q_count,
                np.nan,
            )

        # Re-compute assignments and cell quantiles on the averaged grid
        # to get a single consistent estimator
        assignments_avg = voronoi_assign(X_clvq, avg_grid)
        cell_q_final = cell_quantiles(y, assignments_avg, N_eff, alpha)

        # Where the averaged grid's cells have data, use those;
        # otherwise fall back to the bootstrap average
        for j in range(N_eff):
            if np.isnan(cell_q_final[j, 0]) and not np.isnan(avg_cell_q[j, 0]):
                cell_q_final[j] = avg_cell_q[j]

        self.grid_ = avg_grid
        self.cell_quantiles_ = cell_q_final
        self.N_eff_ = N_eff
        return self

    # ──────────────────────────────────────────────────────────────
    # sklearn tags
    # ──────────────────────────────────────────────────────────────

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # The quantization estimator is a nonparametric method that
        # performs poorly on the tiny synthetic datasets (< 50 samples)
        # used by check_regressors_train.
        tags.regressor_tags.poor_score = True
        return tags

    # ──────────────────────────────────────────────────────────────
    # predict
    # ──────────────────────────────────────────────────────────────

    def predict(self, X):
        """Predict conditional quantiles at *X*.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples,)
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        n, d = X.shape

        if d == 1:
            x_q = X.ravel()
        else:
            x_q = X.T  # (d, m)

        preds = predict_quantiles(x_q, self.grid_, self.cell_quantiles_)
        # preds: (m, 1) → flatten
        return preds.ravel()
