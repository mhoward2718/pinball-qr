"""Optimal quantization grid construction via CLVQ.

Implements the Competitive Learning Vector Quantization (CLVQ) stochastic
gradient algorithm for constructing L_p-optimal quantization grids.
Ported from R's ``QuantifQuantile::choice.grid``.

The algorithm starts from random initial grids drawn from the data, then
iteratively moves the nearest grid point toward each incoming data point
using a decaying step size.  After one pass through the (bootstrapped)
data the grid approximates the L_p-optimal N-point quantizer of X.

References
----------
.. [1] Pages, G. (1998). "A space quantization method for numerical
       integration." *J. Comp. Appl. Math.* 89(1), 1–38.
.. [2] Charlier, I., Paindaveine, D. and Saracco, J. (2015).
       "Conditional quantile estimation through optimal quantization."
       *JSPI* 156, 14–30.
"""

from __future__ import annotations

import numpy as np


def choice_grid(
    X: np.ndarray,
    N: int,
    n_grids: int = 1,
    p: int | float = 2,
    random_state: int | np.random.RandomState | None = None,
) -> dict[str, np.ndarray]:
    """Construct optimal quantization grids for *X* via CLVQ.

    Parameters
    ----------
    X : ndarray, shape (n,) or (d, n)
        Data to quantize.  A 1-D array is treated as univariate;
        a 2-D array with *d* rows is *d*-dimensional.
    N : int
        Number of points in each quantization grid.
    n_grids : int, default 1
        Number of independent grids to construct.
    p : float, default 2
        Exponent of the L_p norm.  ``p=2`` gives the classical CLVQ.
    random_state : int, RandomState, or None
        Seed for reproducibility.

    Returns
    -------
    dict
        ``"initial_grid"`` — the random starting grids.
        ``"optimal_grid"`` — the optimised grids after one CLVQ pass.
        Both have shape ``(N, n_grids)`` (1-D) or ``(d, N, n_grids)`` (d-D).
    """
    # ── validation ──────────────────────────────────────────────────
    if not (isinstance(N, (int, np.integer)) and N > 0):
        raise ValueError("N must be a positive integer.")
    if not (isinstance(n_grids, (int, np.integer)) and n_grids > 0):
        raise ValueError("n_grids must be a positive integer.")
    if p < 1:
        raise ValueError("p must be at least 1.")

    rng = np.random.RandomState(random_state) if not isinstance(
        random_state, np.random.RandomState
    ) else random_state

    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        return _choice_grid_1d(X, N, n_grids, p, rng)
    elif X.ndim == 2:
        return _choice_grid_nd(X, N, n_grids, p, rng)
    else:
        raise ValueError("X must be 1-D or 2-D.")


# ======================================================================
# 1-D implementation
# ======================================================================

def _choice_grid_1d(
    X: np.ndarray,
    N: int,
    ng: int,
    p: float,
    rng: np.random.RandomState,
) -> dict[str, np.ndarray]:
    """CLVQ for univariate X.  Mirrors R's ``choice.grid`` vector branch."""
    n = len(X)

    # Bootstrap stimuli: ng rows, each a resample of X
    stimuli = X[rng.randint(0, n, size=(ng, n))]  # (ng, n)

    # Initial grids: each column is N unique-ish points from X
    unique_X = np.unique(X)
    hat_X = np.empty((N, ng), dtype=np.float64)
    for g in range(ng):
        idx = rng.choice(len(unique_X), size=N, replace=(len(unique_X) < N))
        hat_X[:, g] = unique_X[idx]
    hat_X0 = hat_X.copy()  # save initial grids

    # ── step-size schedule ──────────────────────────────────────────
    a_gamma = 4.0 * N
    b_gamma = np.pi ** 2 * N ** (-2)

    # Initial distortion for each grid → gamma_0
    # Project every X[i] onto its nearest grid point in hat_X0
    # dists: (n, N, ng) would be huge; do per-grid instead
    gamma0 = np.empty(ng, dtype=np.float64)
    for g in range(ng):
        dists = np.abs(X[:, np.newaxis] - hat_X0[:, g][np.newaxis, :])
        nearest_dist = np.min(dists, axis=1)
        gamma0[g] = np.mean(nearest_dist ** 2)
    gamma0 = np.clip(gamma0, 0.005, 1.0)

    # gamma[g, t] = gamma0[g] * a / (a + gamma0[g] * b * t)
    t_vals = np.arange(n + 1, dtype=np.float64)  # 0 .. n
    gamma = gamma0[:, np.newaxis] * a_gamma / (
        a_gamma + gamma0[:, np.newaxis] * b_gamma * t_vals[np.newaxis, :]
    )  # (ng, n+1)

    # ── CLVQ update loop ───────────────────────────────────────────
    for i in range(n):
        # stimuli[g, i] is the incoming point for grid g
        stim = stimuli[:, i]  # (ng,)

        # Distance from stimulus to each grid point: (N, ng)
        diff = hat_X - stim[np.newaxis, :]  # (N, ng)
        dists = np.abs(diff)  # L2 in 1-D = abs

        # Nearest grid point per grid
        imin = np.argmin(dists, axis=0)  # (ng,)

        # Update: hat_X[imin[g], g] -= gamma[g, i+1] * direction
        for g in range(ng):
            j = imin[g]
            d_vec = hat_X[j, g] - stim[g]
            dist_val = np.abs(d_vec)
            if dist_val == 0.0:
                continue  # R: no-op when distance is zero
            if p == 2:
                hat_X[j, g] -= gamma[g, i + 1] * d_vec
            else:
                hat_X[j, g] -= (
                    gamma[g, i + 1] * d_vec * dist_val ** (p - 2)
                )

    return {"initial_grid": hat_X0, "optimal_grid": hat_X}


# ======================================================================
# d-D implementation
# ======================================================================

def _choice_grid_nd(
    X: np.ndarray,
    N: int,
    ng: int,
    p: float,
    rng: np.random.RandomState,
) -> dict[str, np.ndarray]:
    """CLVQ for d-dimensional X.  Mirrors R's ``choice.grid`` matrix branch."""
    d, n = X.shape

    # Bootstrap stimuli: (d, n, ng) — each grid gets its own resample
    idx = rng.randint(0, n, size=(ng, n))  # (ng, n)
    stimuli = np.empty((d, n, ng), dtype=np.float64)
    for g in range(ng):
        stimuli[:, :, g] = X[:, idx[g]]

    # Initial grids: (d, N, ng) — N random distinct columns from X
    hat_X = np.empty((d, N, ng), dtype=np.float64)
    for g in range(ng):
        cols = rng.choice(n, size=N, replace=False)
        hat_X[:, :, g] = X[:, cols]
    hat_X0 = hat_X.copy()

    # ── step-size schedule ──────────────────────────────────────────
    a_gamma = 4.0 * N ** (1.0 / d)
    b_gamma = np.pi ** 2 * N ** (-2.0 / d)

    # gamma_0: half the minimum pairwise distance between grid points
    gamma0 = np.full(ng, np.inf)
    for i in range(N - 1):
        for j in range(i + 1, N):
            dd = np.sqrt(
                np.sum((hat_X0[:, i, :] - hat_X0[:, j, :]) ** 2, axis=0)
            ) / 2.0  # (ng,)
            gamma0 = np.minimum(gamma0, dd)
    gamma0 = np.clip(gamma0, 0.005, 1.0)

    t_vals = np.arange(n + 1, dtype=np.float64)
    gamma = gamma0[:, np.newaxis] * a_gamma / (
        a_gamma + gamma0[:, np.newaxis] * b_gamma * t_vals[np.newaxis, :]
    )  # (ng, n+1)

    # ── CLVQ update loop ───────────────────────────────────────────
    for i in range(n):
        # stim[:, g] is the d-dim incoming point for grid g
        stim = stimuli[:, i, :]  # (d, ng)

        # Distance from stimulus to each grid point: (N, ng)
        # hat_X[:, :, g] is (d, N), stim[:, g] is (d,)
        diff = hat_X - stim[:, np.newaxis, :]  # (d, N, ng)
        dists = np.sqrt(np.sum(diff ** 2, axis=0))  # (N, ng)

        # Nearest grid point per grid
        imin = np.argmin(dists, axis=0)  # (ng,)

        for g in range(ng):
            j = imin[g]
            d_vec = hat_X[:, j, g] - stim[:, g]  # (d,)
            dist_val = np.sqrt(np.sum(d_vec ** 2))
            if dist_val == 0.0:
                continue
            if p == 2:
                hat_X[:, j, g] -= gamma[g, i + 1] * d_vec
            else:
                hat_X[:, j, g] -= (
                    gamma[g, i + 1] * d_vec * dist_val ** (p - 2)
                )

    return {"initial_grid": hat_X0, "optimal_grid": hat_X}
