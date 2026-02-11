"""TDD tests for Voronoi assignment + cell-conditional quantile estimation.

These test the low-level building blocks that the full estimator relies on:
  - voronoi_assign : project data onto nearest grid point
  - cell_quantiles : sample quantiles within each Voronoi cell
  - predict_quantiles : predict conditional quantiles at new query points
"""

import numpy as np
import pytest


# ============================================================
# 1.  voronoi_assign
# ============================================================

class TestVoronoiAssign:
    """Assign each data point to its nearest grid point."""

    def test_1d_simple(self):
        from pinball.nonparametric.quantization._voronoi import voronoi_assign
        grid = np.array([0.0, 1.0, 2.0])          # 3 centroids
        X = np.array([0.1, 0.9, 1.6, 2.1, -0.3])  # 5 points
        idx = voronoi_assign(X, grid)
        np.testing.assert_array_equal(idx, [0, 1, 2, 2, 0])

    def test_2d_simple(self):
        from pinball.nonparametric.quantization._voronoi import voronoi_assign
        # grid: (d=2, N=3)
        grid = np.array([[0, 1, 2],
                         [0, 1, 2]], dtype=float)
        # X: (d=2, n=4)
        X = np.array([[0.1, 0.9, 1.8, 0.4],
                       [0.2, 1.1, 2.0, 0.3]], dtype=float)
        idx = voronoi_assign(X, grid)
        np.testing.assert_array_equal(idx, [0, 1, 2, 0])

    def test_output_length_matches_n(self):
        from pinball.nonparametric.quantization._voronoi import voronoi_assign
        rng = np.random.RandomState(42)
        X = rng.randn(200)
        grid = np.linspace(-3, 3, 10)
        idx = voronoi_assign(X, grid)
        assert idx.shape == (200,)

    def test_all_indices_valid(self):
        from pinball.nonparametric.quantization._voronoi import voronoi_assign
        rng = np.random.RandomState(42)
        X = rng.randn(200)
        grid = np.linspace(-3, 3, 10)
        idx = voronoi_assign(X, grid)
        assert np.all(idx >= 0)
        assert np.all(idx < 10)

    def test_ties_go_to_lower_index(self):
        """When equidistant, prefer the first (lower-index) grid point."""
        from pinball.nonparametric.quantization._voronoi import voronoi_assign
        grid = np.array([0.0, 2.0])
        X = np.array([1.0])  # equidistant to both
        idx = voronoi_assign(X, grid)
        assert idx[0] == 0  # argmin returns first


# ============================================================
# 2.  cell_quantiles
# ============================================================

class TestCellQuantiles:
    """Compute sample quantiles of Y within each Voronoi cell."""

    def test_basic_shape(self):
        from pinball.nonparametric.quantization._voronoi import cell_quantiles
        Y = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        assignments = np.array([0, 0, 1, 1, 2, 2])
        alpha = np.array([0.25, 0.5, 0.75])
        result = cell_quantiles(Y, assignments, n_cells=3, alpha=alpha)
        # shape should be (n_cells, len(alpha))
        assert result.shape == (3, 3)

    def test_single_alpha(self):
        from pinball.nonparametric.quantization._voronoi import cell_quantiles
        Y = np.array([10, 20, 30, 40], dtype=float)
        assignments = np.array([0, 0, 1, 1])
        alpha = np.array([0.5])
        result = cell_quantiles(Y, assignments, n_cells=2, alpha=alpha)
        assert result.shape == (2, 1)
        # Median of [10, 20] and [30, 40]
        np.testing.assert_allclose(result[0, 0], 15.0)
        np.testing.assert_allclose(result[1, 0], 35.0)

    def test_empty_cell_returns_nan(self):
        from pinball.nonparametric.quantization._voronoi import cell_quantiles
        Y = np.array([1.0, 2.0, 3.0])
        assignments = np.array([0, 0, 0])  # only cell 0 occupied
        alpha = np.array([0.5])
        result = cell_quantiles(Y, assignments, n_cells=3, alpha=alpha)
        assert np.isfinite(result[0, 0])  # cell 0 has data
        assert np.isnan(result[1, 0])     # cell 1 empty
        assert np.isnan(result[2, 0])     # cell 2 empty

    def test_quantile_ordering(self):
        """For the same cell, quantile at alpha=0.25 ≤ 0.5 ≤ 0.75."""
        from pinball.nonparametric.quantization._voronoi import cell_quantiles
        rng = np.random.RandomState(99)
        Y = rng.randn(100)
        assignments = np.zeros(100, dtype=int)
        alpha = np.array([0.25, 0.5, 0.75])
        result = cell_quantiles(Y, assignments, n_cells=1, alpha=alpha)
        assert result[0, 0] <= result[0, 1] <= result[0, 2]


# ============================================================
# 3.  predict_quantiles
# ============================================================

class TestPredictQuantiles:
    """Predict conditional quantiles at query points."""

    def test_1d_shape(self):
        from pinball.nonparametric.quantization._voronoi import (
            predict_quantiles,
        )
        grid = np.array([0.0, 1.0, 2.0])
        # cell_quants: (3 cells, 2 alphas)
        cell_quants = np.array([[0.1, 0.2],
                                [0.3, 0.4],
                                [0.5, 0.6]])
        x_new = np.array([0.1, 0.9, 1.8])
        result = predict_quantiles(x_new, grid, cell_quants)
        assert result.shape == (3, 2)  # (n_query, n_alpha)

    def test_1d_correct_lookup(self):
        from pinball.nonparametric.quantization._voronoi import (
            predict_quantiles,
        )
        grid = np.array([0.0, 1.0, 2.0])
        cell_quants = np.array([[10.0, 20.0],
                                [30.0, 40.0],
                                [50.0, 60.0]])
        # 0.1 → cell 0, 1.1 → cell 1, 1.9 → cell 2
        x_new = np.array([0.1, 1.1, 1.9])
        result = predict_quantiles(x_new, grid, cell_quants)
        np.testing.assert_array_equal(result[0], [10.0, 20.0])
        np.testing.assert_array_equal(result[1], [30.0, 40.0])
        np.testing.assert_array_equal(result[2], [50.0, 60.0])

    def test_2d_correct_lookup(self):
        from pinball.nonparametric.quantization._voronoi import (
            predict_quantiles,
        )
        # grid: (2, 2) — two 2-D centroids
        grid = np.array([[0.0, 1.0],
                         [0.0, 1.0]], dtype=float)
        cell_quants = np.array([[5.0], [15.0]])
        # x close to centroid 0 and centroid 1
        x_new = np.array([[0.1, 0.9],
                          [0.1, 0.9]], dtype=float)
        result = predict_quantiles(x_new, grid, cell_quants)
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result[0], [5.0])
        np.testing.assert_array_equal(result[1], [15.0])


# ============================================================
# 4.  Integration: assign → quantile → predict round-trip
# ============================================================

class TestVoronoiRoundTrip:
    """End-to-end: data generation → Voronoi → cell quantiles → prediction."""

    def test_1d_round_trip(self):
        from pinball.nonparametric.quantization._voronoi import (
            cell_quantiles,
            predict_quantiles,
            voronoi_assign,
        )
        rng = np.random.RandomState(7)
        # Two clusters: X ~ [-1, 0] and X ~ [1, 2]
        X = np.concatenate([rng.uniform(-1, 0, 50), rng.uniform(1, 2, 50)])
        Y = X ** 2 + rng.randn(100) * 0.1
        grid = np.array([-0.5, 1.5])

        idx = voronoi_assign(X, grid)
        cq = cell_quantiles(Y, idx, n_cells=2, alpha=np.array([0.5]))
        pred = predict_quantiles(np.array([-0.3, 1.3]), grid, cq)

        # Median of X^2 near -0.5 should be ~0.25, near 1.5 should be ~2.25
        assert pred.shape == (2, 1)
        assert 0.0 < pred[0, 0] < 1.0   # roughly 0.25
        assert 1.0 < pred[1, 0] < 4.0   # roughly 2.25
