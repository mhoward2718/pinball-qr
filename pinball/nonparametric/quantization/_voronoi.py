"""Voronoi assignment and cell-conditional quantile computation.

Provides the low-level building blocks that the quantization-based
conditional quantile estimator uses:

- ``voronoi_assign``: map each data point to its nearest grid point.
- ``cell_quantiles``: compute sample quantiles of *Y* within Voronoi cells.
- ``predict_quantiles``: predict conditional quantiles at new query points.
"""

from __future__ import annotations

import numpy as np


def voronoi_assign(X: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Assign each point in *X* to the nearest point in *grid*.

    Parameters
    ----------
    X : ndarray, shape (n,) or (d, n)
        Data points.
    grid : ndarray, shape (N,) or (d, N)
        Grid / codebook centroids.

    Returns
    -------
    ndarray of int, shape (n,)
        Index into *grid* of the nearest centroid for each point in *X*.
    """
    X = np.asarray(X, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    if X.ndim == 1:
        # 1-D: |X[i] - grid[j]|, shape (n, N)
        dists = np.abs(X[:, np.newaxis] - grid[np.newaxis, :])
    elif X.ndim == 2:
        d, n = X.shape
        d_g, N = grid.shape
        assert d == d_g, "X and grid must have the same number of rows (d)."
        # (d, n, 1) - (d, 1, N) → (d, n, N) → sum over d → (n, N)
        diff = X[:, :, np.newaxis] - grid[:, np.newaxis, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=0))
    else:
        raise ValueError("X must be 1-D or 2-D.")

    return np.argmin(dists, axis=1)


def cell_quantiles(
    Y: np.ndarray,
    assignments: np.ndarray,
    n_cells: int,
    alpha: np.ndarray,
) -> np.ndarray:
    """Compute sample quantiles of *Y* within each Voronoi cell.

    Parameters
    ----------
    Y : ndarray, shape (n,)
        Response values.
    assignments : ndarray of int, shape (n,)
        Cell membership index for each observation (from ``voronoi_assign``).
    n_cells : int
        Number of cells in the grid (``N``).
    alpha : ndarray, shape (q,)
        Quantile levels, each in (0, 1).

    Returns
    -------
    ndarray, shape (n_cells, q)
        ``result[j, k]`` is the ``alpha[k]``-th quantile of Y in cell *j*.
        NaN for empty cells.
    """
    q = len(alpha)
    result = np.full((n_cells, q), np.nan)
    for j in range(n_cells):
        mask = assignments == j
        if np.any(mask):
            result[j] = np.quantile(Y[mask], alpha)
    return result


def predict_quantiles(
    x_new: np.ndarray,
    grid: np.ndarray,
    cell_quants: np.ndarray,
) -> np.ndarray:
    """Predict conditional quantiles at query points.

    Each query point is assigned to its nearest grid cell, and the
    pre-computed cell quantile is returned.

    Parameters
    ----------
    x_new : ndarray, shape (m,) or (d, m)
        Query points.
    grid : ndarray, shape (N,) or (d, N)
        Grid centroids.
    cell_quants : ndarray, shape (N, q)
        Pre-computed quantile values for each cell (from ``cell_quantiles``).

    Returns
    -------
    ndarray, shape (m, q)
        Predicted conditional quantiles at each query point.
    """
    idx = voronoi_assign(x_new, grid)
    return cell_quants[idx]
