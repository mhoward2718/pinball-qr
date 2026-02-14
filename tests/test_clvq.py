"""TDD tests for the CLVQ optimal quantization grid construction.

Tests the ``choice_grid`` function which implements the Competitive
Learning Vector Quantization algorithm (Pages 1998), ported from
R's ``QuantifQuantile::choice.grid``.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# choice_grid output shape and basic invariants
# ---------------------------------------------------------------------------

class TestChoiceGridShape:
    """Output dimensionality and type."""

    def test_univariate_shape(self):
        """1-D X → grid shape (N, n_grids)."""
        from pinball.nonparametric.quantization._clvq import choice_grid

        rng = np.random.RandomState(42)
        X = rng.uniform(-2, 2, size=300)
        result = choice_grid(X, N=10, n_grids=5, random_state=42)
        grid = result["optimal_grid"]
        assert grid.shape == (10, 5)

    def test_multivariate_shape(self):
        """d-D X → grid shape (d, N, n_grids)."""
        from pinball.nonparametric.quantization._clvq import choice_grid

        rng = np.random.RandomState(42)
        X = rng.uniform(-2, 2, size=(2, 300))
        result = choice_grid(X, N=10, n_grids=5, random_state=42)
        grid = result["optimal_grid"]
        assert grid.shape == (2, 10, 5)

    def test_single_grid(self):
        """Default n_grids=1 still works."""
        from pinball.nonparametric.quantization._clvq import choice_grid

        X = np.linspace(-2, 2, 200)
        result = choice_grid(X, N=8, n_grids=1, random_state=0)
        assert result["optimal_grid"].shape == (8, 1)

    def test_returns_initial_and_optimal(self):
        """Result dict contains both initial and optimal grids."""
        from pinball.nonparametric.quantization._clvq import choice_grid

        X = np.linspace(-2, 2, 200)
        result = choice_grid(X, N=8, n_grids=3, random_state=0)
        assert "initial_grid" in result
        assert "optimal_grid" in result
        assert result["initial_grid"].shape == result["optimal_grid"].shape


# ---------------------------------------------------------------------------
# CLVQ convergence properties
# ---------------------------------------------------------------------------

class TestChoiceGridConvergence:
    """The optimized grid should be better than the initial random grid."""

    @staticmethod
    def _distortion_1d(X, grid_col):
        """Mean squared distance from each X to its nearest grid point."""
        dists = np.abs(X[:, np.newaxis] - grid_col[np.newaxis, :])
        return np.mean(np.min(dists, axis=1) ** 2)

    def test_distortion_decreases_univariate(self):
        """Optimal grid has lower distortion than initial grid (1-D)."""
        from pinball.nonparametric.quantization._clvq import choice_grid

        rng = np.random.RandomState(644972)
        X = rng.uniform(-2, 2, size=300)
        result = choice_grid(X, N=10, n_grids=5, random_state=644972)

        for g in range(5):
            d_init = self._distortion_1d(X, result["initial_grid"][:, g])
            d_opt = self._distortion_1d(X, result["optimal_grid"][:, g])
            assert d_opt <= d_init, (
                f"Grid {g}: optimal distortion {d_opt:.4f} > "
                f"initial {d_init:.4f}"
            )

    @staticmethod
    def _distortion_nd(X, grid):
        """Mean squared distance from each X column to nearest grid col."""
        # X: (d, n), grid: (d, N) → dists: (n, N)
        dists = np.sqrt(
            np.sum(
                (X[:, :, np.newaxis] - grid[:, np.newaxis, :]) ** 2,
                axis=0,
            )
        )
        return np.mean(np.min(dists, axis=1) ** 2)

    def test_distortion_decreases_multivariate(self):
        """Optimal grid has lower distortion than initial grid (2-D)."""
        from pinball.nonparametric.quantization._clvq import choice_grid

        rng = np.random.RandomState(42)
        X = rng.uniform(-2, 2, size=(2, 500))
        result = choice_grid(X, N=15, n_grids=3, random_state=42)

        for g in range(3):
            d_init = self._distortion_nd(
                X, result["initial_grid"][:, :, g]
            )
            d_opt = self._distortion_nd(
                X, result["optimal_grid"][:, :, g]
            )
            assert d_opt <= d_init


# ---------------------------------------------------------------------------
# Grid points should stay within data range
# ---------------------------------------------------------------------------

class TestChoiceGridRange:
    """Grid points should remain in the convex hull of the data."""

    def test_1d_grid_within_data_range(self):
        from pinball.nonparametric.quantization._clvq import choice_grid

        rng = np.random.RandomState(42)
        X = rng.uniform(-2, 2, size=300)
        result = choice_grid(X, N=10, n_grids=5, random_state=42)
        grid = result["optimal_grid"]
        # Allow small numerical tolerance outside data range
        assert np.all(grid >= X.min() - 0.5)
        assert np.all(grid <= X.max() + 0.5)

    def test_2d_grid_within_data_range(self):
        from pinball.nonparametric.quantization._clvq import choice_grid

        rng = np.random.RandomState(42)
        X = rng.uniform(-2, 2, size=(2, 300))
        result = choice_grid(X, N=10, n_grids=3, random_state=42)
        grid = result["optimal_grid"]
        for dim in range(2):
            assert np.all(grid[dim] >= X[dim].min() - 0.5)
            assert np.all(grid[dim] <= X[dim].max() + 0.5)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestChoiceGridDeterminism:
    """Same random_state produces identical grids."""

    def test_reproducible_1d(self):
        from pinball.nonparametric.quantization._clvq import choice_grid

        X = np.random.RandomState(0).uniform(-2, 2, 200)
        r1 = choice_grid(X, N=10, n_grids=3, random_state=99)
        r2 = choice_grid(X, N=10, n_grids=3, random_state=99)
        np.testing.assert_array_equal(
            r1["optimal_grid"], r2["optimal_grid"]
        )

    def test_reproducible_2d(self):
        from pinball.nonparametric.quantization._clvq import choice_grid

        X = np.random.RandomState(0).uniform(-2, 2, (2, 200))
        r1 = choice_grid(X, N=10, n_grids=3, random_state=99)
        r2 = choice_grid(X, N=10, n_grids=3, random_state=99)
        np.testing.assert_array_equal(
            r1["optimal_grid"], r2["optimal_grid"]
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestChoiceGridValidation:
    """Input validation matches R's checks (translated to English)."""

    def test_N_must_be_positive_integer(self):
        from pinball.nonparametric.quantization._clvq import choice_grid

        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="N must be a positive integer"):
            choice_grid(X, N=0)
        with pytest.raises(ValueError, match="N must be a positive integer"):
            choice_grid(X, N=-1)
        with pytest.raises(ValueError, match="N must be a positive integer"):
            choice_grid(X, N=2.5)

    def test_p_must_be_at_least_1(self):
        from pinball.nonparametric.quantization._clvq import choice_grid

        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="p must be at least 1"):
            choice_grid(X, N=2, p=0.5)

    def test_n_grids_must_be_positive_integer(self):
        from pinball.nonparametric.quantization._clvq import choice_grid

        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="n_grids must be a positive integer"):
            choice_grid(X, N=2, n_grids=0)


# ---------------------------------------------------------------------------
# Lp norm parameter
# ---------------------------------------------------------------------------

class TestChoiceGridLpNorm:
    """p != 2 should also produce valid grids."""

    def test_p1_produces_valid_grid(self):
        from pinball.nonparametric.quantization._clvq import choice_grid

        X = np.random.RandomState(42).uniform(-2, 2, 200)
        result = choice_grid(X, N=10, n_grids=2, p=1, random_state=42)
        assert result["optimal_grid"].shape == (10, 2)
        # Grid should still be finite
        assert np.all(np.isfinite(result["optimal_grid"]))

    def test_p3_produces_valid_grid(self):
        from pinball.nonparametric.quantization._clvq import choice_grid

        X = np.random.RandomState(42).uniform(-2, 2, 200)
        result = choice_grid(X, N=10, n_grids=2, p=3, random_state=42)
        assert result["optimal_grid"].shape == (10, 2)
        assert np.all(np.isfinite(result["optimal_grid"]))
