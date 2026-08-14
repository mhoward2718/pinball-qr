"""Quantization-based conditional quantile estimator.

Implements the nonparametric conditional quantile estimator of
Charlier, Paindaveine & Saracco (2015) as a scikit-learn--compatible
estimator.

The algorithm:
1. Construct *n_grids* independent, L_p-optimal quantization grids of
   size *N* for the covariate X using CLVQ (``choice_grid``).
2. For each grid *separately*, assign every training (X_i, Y_i) to its
   Voronoi cell and compute the sample ``tau``-quantile of Y within
   each cell -- this gives *n_grids* independent cell-quantile tables.
3. At prediction time, assign each new x to its nearest cell *within
   each grid separately*, look up that grid's cell-quantile estimate,
   and average the resulting *n_grids* predicted values.

Bagging happens in prediction space (averaging predicted quantile
values across grids), not in grid-geometry space. The N points of two
independently-initialised/updated grids have no correspondence to each
other (grid A's 7th point and grid B's 7th point represent unrelated
regions), so averaging grid-point *positions* across grids is invalid
-- it collapses the quantizer toward the data's centroid instead of
covering its support. This mirrors R's ``QuantifQuantile`` reference
implementation, which never averages grid positions either.

References
----------
.. [1] Charlier, I., Paindaveine, D. and Saracco, J. (2015).
       "Conditional quantile estimation through optimal quantization."
       *JSPI* 156, 14–30.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.utils.validation import check_is_fitted, validate_data

from pinball.estimators._base import BaseQuantileEstimator
from pinball.nonparametric.quantization._clvq import choice_grid
from pinball.nonparametric.quantization._voronoi import cell_quantiles, voronoi_assign


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
    grids_ : ndarray, shape (N, n_grids) or (d, N, n_grids)
        The ``n_grids`` independent optimal quantization grids.
    cell_quantiles_ : ndarray, shape (N, n_grids)
        Per-grid conditional-quantile estimate for each Voronoi cell.
        ``cell_quantiles_[j, g]`` is grid *g*'s ``tau``-quantile estimate
        for its cell *j*; NaN where that cell had no training data.
    N_eff_ : int
        Effective grid size actually used (``min(N, n_samples)``).
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
        X_clvq = X.ravel() if d == 1 else X.T  # noqa: SIM108

        grids = choice_grid(
            X_clvq, N_eff, n_grids=self.n_grids,
            p=self.p, random_state=self.random_state,
        )
        opt_grids = grids["optimal_grid"]
        # opt_grids shape: (N_eff, n_grids) for 1-D, (d, N_eff, n_grids) for d-D

        # Each grid gets its own independent cell-quantile table. These are
        # kept separate (not merged into one grid) -- see the module
        # docstring for why averaging grid *positions* across independently
        # constructed grids is invalid.
        cell_quantiles_per_grid = np.full((N_eff, self.n_grids), np.nan, dtype=np.float64)
        for g in range(self.n_grids):
            grid_g = opt_grids[:, g] if d == 1 else opt_grids[:, :, g]
            assignments = voronoi_assign(X_clvq, grid_g)
            cq = cell_quantiles(y, assignments, N_eff, alpha)  # (N_eff, 1)
            cell_quantiles_per_grid[:, g] = cq[:, 0]

        self.grids_ = opt_grids
        self.cell_quantiles_ = cell_quantiles_per_grid
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

        x_q = X.ravel() if d == 1 else X.T

        # Bag in prediction space: assign each query point to its nearest
        # cell *within each grid separately*, look up that grid's own
        # cell-quantile estimate, then average the n_grids predictions.
        preds_per_grid = np.empty((n, self.n_grids), dtype=np.float64)
        for g in range(self.n_grids):
            grid_g = self.grids_[:, g] if d == 1 else self.grids_[:, :, g]
            assignments = voronoi_assign(x_q, grid_g)
            preds_per_grid[:, g] = self.cell_quantiles_[assignments, g]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(preds_per_grid, axis=1)
