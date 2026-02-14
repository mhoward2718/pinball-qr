"""Tests for the Frisch-Newton interior-point solver."""


import numpy as np
import pytest

from pinball.linear.solvers.fnb import FNBSolver


class TestFNBSolver:

    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(42)
        n, p = 50, 3
        X = rng.randn(n, p)
        y = X @ [1, 2, 3] + rng.randn(n) * 0.5
        return X, y

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="beta must be in"):
            FNBSolver(beta=1.5)

    def test_invalid_beta_zero(self):
        with pytest.raises(ValueError, match="beta must be in"):
            FNBSolver(beta=0.0)

    def test_tau_near_zero_raises(self, data):
        X, y = data
        solver = FNBSolver()
        with pytest.raises(ValueError, match="FNB requires tau"):
            solver.solve(X, y, tau=1e-8)

    def test_tau_near_one_raises(self, data):
        X, y = data
        solver = FNBSolver()
        with pytest.raises(ValueError, match="FNB requires tau"):
            solver.solve(X, y, tau=1 - 1e-8)

    def test_default_params(self):
        s = FNBSolver()
        assert s.beta == 0.99995
        assert s.eps == 1e-6


# ──────────────────────────────────────────────────────────────────────
# Integration test (requires compiled Fortran)
# ──────────────────────────────────────────────────────────────────────

def _has_native():
    try:
        from pinball._native import rqfnb  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
class TestFNBSolverIntegration:
    """End-to-end tests using the real Fortran solver."""

    def test_engel_median(self):
        from pinball.datasets import load_engel
        data = load_engel()
        X = np.column_stack([np.ones(len(data.target)), data.data])
        y = data.target
        solver = FNBSolver()
        result = solver.solve(X, y, tau=0.5)

        assert result.status == 0
        assert result.coefficients.shape == (2,)
        # Known R result: intercept ≈ 81.48, slope ≈ 0.5602
        np.testing.assert_allclose(result.coefficients, [81.48, 0.5602], atol=1.0)

    def test_multiple_quantiles(self):
        from pinball.datasets import load_engel
        data = load_engel()
        X = np.column_stack([np.ones(len(data.target)), data.data])
        y = data.target

        solver = FNBSolver()
        for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result = solver.solve(X, y, tau=tau)
            assert result.status == 0
            assert result.coefficients.shape == (2,)
            # Slope should be positive (food exp increases with income)
            assert result.coefficients[1] > 0
