"""Tests for the solver base class and SolverResult."""

import numpy as np
import pytest

from pinball.linear.solvers.base import BaseSolver, SolverResult

# ──────────────────────────────────────────────────────────────────────
# Concrete mock solver for testing the ABC contract
# ──────────────────────────────────────────────────────────────────────

class _MockSolver(BaseSolver):
    """Trivially returns OLS coefficients for testing the interface."""

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


# ──────────────────────────────────────────────────────────────────────
# SolverResult
# ──────────────────────────────────────────────────────────────────────

class TestSolverResult:

    def test_construction(self):
        coef = np.array([1.0, 2.0])
        resid = np.array([0.1, -0.1, 0.05])
        r = SolverResult(coefficients=coef, residuals=resid)
        np.testing.assert_array_equal(r.coefficients, coef)
        np.testing.assert_array_equal(r.residuals, resid)
        assert r.status == 0
        assert r.iterations == 0
        assert r.solver_info == {}
        assert r.dual_solution is None

    def test_frozen(self):
        r = SolverResult(
            coefficients=np.array([1.0]),
            residuals=np.array([0.0]),
        )
        with pytest.raises(AttributeError):
            r.status = 99


# ──────────────────────────────────────────────────────────────────────
# BaseSolver interface contract
# ──────────────────────────────────────────────────────────────────────

class TestBaseSolverContract:

    @pytest.fixture
    def solver(self):
        return _MockSolver()

    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(42)
        n, p = 50, 3
        X = rng.randn(n, p)
        y = X @ [1, 2, 3] + rng.randn(n) * 0.1
        return X, y

    def test_solve_returns_solver_result(self, solver, data):
        X, y = data
        result = solver.solve(X, y, tau=0.5)
        assert isinstance(result, SolverResult)

    def test_coefficients_shape(self, solver, data):
        X, y = data
        result = solver.solve(X, y, tau=0.5)
        assert result.coefficients.shape == (X.shape[1],)

    def test_residuals_shape(self, solver, data):
        X, y = data
        result = solver.solve(X, y, tau=0.5)
        assert result.residuals.shape == (X.shape[0],)

    def test_invalid_tau_zero(self, solver, data):
        X, y = data
        with pytest.raises(ValueError, match="tau must be in"):
            solver.solve(X, y, tau=0.0)

    def test_invalid_tau_one(self, solver, data):
        X, y = data
        with pytest.raises(ValueError, match="tau must be in"):
            solver.solve(X, y, tau=1.0)

    def test_invalid_tau_negative(self, solver, data):
        X, y = data
        with pytest.raises(ValueError, match="tau must be in"):
            solver.solve(X, y, tau=-0.5)

    def test_X_not_2d(self, solver):
        with pytest.raises(ValueError, match="2-D"):
            solver.solve(np.array([1, 2, 3]), np.array([1, 2, 3]), tau=0.5)

    def test_X_y_shape_mismatch(self, solver):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2])
        with pytest.raises(ValueError, match="incompatible shapes"):
            solver.solve(X, y, tau=0.5)

    def test_n_lt_2_raises(self, solver):
        X = np.array([[1, 2, 3]])
        y = np.array([1])
        with pytest.raises(ValueError, match="1 sample"):
            solver.solve(X, y, tau=0.5)

    def test_dtypes_coerced_to_float64(self, solver):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        y = np.array([1, 2, 3], dtype=np.int32)
        result = solver.solve(X, y, tau=0.5)
        assert result.coefficients.dtype == np.float64

    def test_supports_multiple_quantiles_default_false(self, solver):
        assert solver.supports_multiple_quantiles() is False

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            BaseSolver()
