"""Tests for the L1-penalised (Lasso) quantile regression solver."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pinball.linear.solvers.lasso import LassoSolver
from pinball.linear.solvers.base import SolverResult


class TestLassoSolverInit:

    def test_default_lambda_none(self):
        solver = LassoSolver()
        assert solver.lambda_ is None

    def test_custom_lambda(self):
        solver = LassoSolver(lambda_=0.5)
        assert solver.lambda_ == 0.5

    def test_penalize_intercept_default_false(self):
        solver = LassoSolver()
        assert solver.penalize_intercept is False


class TestLassoSolverSolve:

    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(42)
        n, p = 50, 3
        X = rng.randn(n, p)
        y = X @ [1, 2, 3] + rng.randn(n) * 0.5
        return X, y

    def test_augments_design_matrix(self, data):
        """Verify the augmented matrix has n+p rows."""
        X, y = data
        n, p = X.shape
        solver = LassoSolver(lambda_=0.1)

        # Mock the inner FNB solver
        mock_fnb = MagicMock()
        mock_fnb.solve.return_value = SolverResult(
            coefficients=np.array([1.0, 2.0, 3.0]),
            residuals=np.zeros(n + p),
            objective_value=0.0,
            status=0,
            iterations=10,
            solver_info={},
        )
        solver._fnb = mock_fnb

        result = solver.solve(X, y, tau=0.5)

        # FNB should have been called with augmented matrix (n+p, p)
        call_args = mock_fnb.solve.call_args
        X_aug = call_args[0][0]
        y_aug = call_args[0][1]
        assert X_aug.shape == (n + p, p)
        assert y_aug.shape == (n + p,)

    def test_no_penalize_intercept(self, data):
        """First penalty weight should be 0 when penalize_intercept=False."""
        X, y = data
        n, p = X.shape
        solver = LassoSolver(lambda_=0.5, penalize_intercept=False)

        mock_fnb = MagicMock()
        mock_fnb.solve.return_value = SolverResult(
            coefficients=np.ones(p),
            residuals=np.zeros(n + p),
            objective_value=0.0, status=0, iterations=5,
        )
        solver._fnb = mock_fnb

        solver.solve(X, y, tau=0.5)

        X_aug = mock_fnb.solve.call_args[0][0]
        # Bottom block is diag(pen); pen[0] should be 0
        penalty_block = X_aug[n:, :]
        assert penalty_block[0, 0] == 0.0  # intercept not penalised
        assert penalty_block[1, 1] == 0.5  # others penalised

    def test_residuals_on_original_data(self, data):
        X, y = data
        n, p = X.shape
        coef = np.array([1.0, 2.0, 3.0])
        solver = LassoSolver(lambda_=0.1)

        mock_fnb = MagicMock()
        mock_fnb.solve.return_value = SolverResult(
            coefficients=coef,
            residuals=np.zeros(n + p),
            objective_value=0.0, status=0, iterations=5,
        )
        solver._fnb = mock_fnb

        result = solver.solve(X, y, tau=0.5)
        expected_resid = y - X @ coef
        np.testing.assert_allclose(result.residuals, expected_resid)

    def test_lambda_in_solver_info(self, data):
        X, y = data
        n, p = X.shape
        solver = LassoSolver(lambda_=0.42)

        mock_fnb = MagicMock()
        mock_fnb.solve.return_value = SolverResult(
            coefficients=np.ones(p),
            residuals=np.zeros(n + p),
            objective_value=0.0, status=0, iterations=1,
            solver_info={},
        )
        solver._fnb = mock_fnb

        result = solver.solve(X, y, tau=0.5)
        assert result.solver_info["lambda"] == 0.42

    def test_auto_lambda_positive(self, data):
        """When lambda_=None, BCV lambda should be used (positive)."""
        X, y = data
        n, p = X.shape
        solver = LassoSolver(lambda_=None)

        mock_fnb = MagicMock()
        mock_fnb.solve.return_value = SolverResult(
            coefficients=np.ones(p),
            residuals=np.zeros(n + p),
            objective_value=0.0, status=0, iterations=1,
            solver_info={},
        )
        solver._fnb = mock_fnb

        result = solver.solve(X, y, tau=0.5)
        assert result.solver_info["lambda"] > 0
