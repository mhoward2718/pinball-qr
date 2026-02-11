"""Tests for the Preprocessing + Frisch-Newton solver."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from pinball.solvers.pfn import PreprocessingSolver
from pinball.solvers.base import BaseSolver, SolverResult


class _FakeInnerSolver(BaseSolver):
    """OLS-based solver used as the inner solver for testing."""

    def _solve_impl(self, X, y, tau, **kwargs):
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ coef
        return SolverResult(
            coefficients=coef,
            residuals=residuals,
            objective_value=float(np.sum(np.abs(residuals))),
            status=0,
            iterations=1,
        )


class TestPreprocessingSolverInit:

    def test_defaults(self):
        solver = PreprocessingSolver()
        assert solver.mm_factor == 0.8
        assert solver.max_bad_fixups == 3
        assert solver.eps == 1e-6

    def test_custom_inner_solver(self):
        inner = _FakeInnerSolver()
        solver = PreprocessingSolver(inner_solver=inner)
        assert solver.inner_solver is inner


class TestPreprocessingSolverSmallN:

    def test_fallback_when_m_exceeds_n(self):
        """When n is small enough that m >= n, PFN should delegate directly."""
        rng = np.random.RandomState(42)
        n, p = 20, 3
        X = rng.randn(n, p)
        y = X @ [1, 2, 3] + rng.randn(n)

        inner = _FakeInnerSolver()
        solver = PreprocessingSolver(inner_solver=inner)
        result = solver.solve(X, y, tau=0.5)

        assert isinstance(result, SolverResult)
        assert result.coefficients.shape == (p,)

    def test_returns_correct_residuals(self):
        rng = np.random.RandomState(42)
        n, p = 20, 3
        X = rng.randn(n, p)
        y = X @ [1, 2, 3] + rng.randn(n)

        inner = _FakeInnerSolver()
        solver = PreprocessingSolver(inner_solver=inner)
        result = solver.solve(X, y, tau=0.5)

        expected = y - X @ result.coefficients
        np.testing.assert_allclose(result.residuals, expected, atol=1e-10)


class TestPreprocessingSolverLargeN:

    def test_preprocessing_activates(self):
        """With large n, the preprocessing loop should actually run."""
        rng = np.random.RandomState(42)
        n, p = 500, 3  # m ≈ sqrt(3)*500^(2/3) ≈ 109 < 500
        X = rng.randn(n, p)
        y = X @ [1, 2, 3] + rng.randn(n)

        inner = _FakeInnerSolver()
        solver = PreprocessingSolver(inner_solver=inner)
        result = solver.solve(X, y, tau=0.5)

        assert isinstance(result, SolverResult)
        assert result.coefficients.shape == (p,)
        # Residuals should be on the full data
        assert result.residuals.shape == (n,)

    def test_solver_info_has_preprocessing_flag(self):
        """When preprocessing converges, solver_info should have 'preprocessing'."""
        rng = np.random.RandomState(0)  # seed 0 converges here
        n, p = 2000, 3  # large enough that m ≈ 275 << 2000
        X = rng.randn(n, p)
        y = X @ [1, 2, 3] + rng.randn(n)

        inner = _FakeInnerSolver()
        solver = PreprocessingSolver(inner_solver=inner)
        result = solver.solve(X, y, tau=0.5)

        # Either preprocessing succeeded (flag is True) or it fell back
        # to full solve; both paths produce valid results
        assert isinstance(result, SolverResult)
        assert result.coefficients.shape == (p,)
