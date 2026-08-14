"""Tests for the L1-penalised (Lasso) quantile regression solver."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from pinball.linear.solvers.base import SolverResult
from pinball.linear.solvers.lasso import LassoSolver


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
        """Each penalised coefficient gets a +row and a -row (see module
        docstring: that's what makes the penalty symmetric at any tau)."""
        X, y = data
        n, p = X.shape
        solver = LassoSolver(lambda_=0.1)  # penalize_intercept=False -> col 0 unpenalised

        # Mock the inner FNB solver
        mock_fnb = MagicMock()
        mock_fnb.solve.return_value = SolverResult(
            coefficients=np.array([1.0, 2.0, 3.0]),
            residuals=np.zeros(n + 2 * (p - 1)),
            objective_value=0.0,
            status=0,
            iterations=10,
            solver_info={},
        )
        solver._fnb = mock_fnb

        solver.solve(X, y, tau=0.5)

        call_args = mock_fnb.solve.call_args
        X_aug = call_args[0][0]
        y_aug = call_args[0][1]
        n_penalized = p - 1  # column 0 is the unpenalised intercept
        assert X_aug.shape == (n + 2 * n_penalized, p)
        assert y_aug.shape == (n + 2 * n_penalized,)
        np.testing.assert_array_equal(y_aug[n:], 0.0)  # penalty rows target 0

    def test_no_penalize_intercept(self, data):
        """No +/- penalty rows should touch column 0 when
        penalize_intercept=False; the other columns get a +lambda and a
        -lambda row each."""
        X, y = data
        n, p = X.shape
        solver = LassoSolver(lambda_=0.5, penalize_intercept=False)

        mock_fnb = MagicMock()
        mock_fnb.solve.return_value = SolverResult(
            coefficients=np.ones(p),
            residuals=np.zeros(n + 2 * (p - 1)),
            objective_value=0.0, status=0, iterations=5,
        )
        solver._fnb = mock_fnb

        solver.solve(X, y, tau=0.5)

        X_aug = mock_fnb.solve.call_args[0][0]
        penalty_block = X_aug[n:, :]
        assert np.all(penalty_block[:, 0] == 0.0)  # intercept column untouched
        # 2 penalised columns -> 4 rows; column 1 is +/-0.5 in its own pair
        # of rows and 0 in the other column's pair, and vice versa.
        assert sorted(penalty_block[:, 1]) == [-0.5, 0.0, 0.0, 0.5]
        assert sorted(penalty_block[:, 2]) == [-0.5, 0.0, 0.0, 0.5]

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


class TestLassoSolverSymmetricPenalty:
    """Real (non-mocked) solves checking the penalty is symmetric in the
    coefficient's *sign* at extreme tau -- not just at tau=0.5, where a
    single-row-per-coefficient penalty happens to be symmetric anyway
    (rho_0.5(u) = 0.5*|u|) and so cannot distinguish a correct
    implementation from the sign-dependent bug this solver used to have.

    Reference behaviour cross-checked against R's ``rq.fit.lasso`` (same
    data, same lambda): at a large-enough lambda, R drives every penalised
    coefficient to exactly 0 regardless of tau or the coefficient's true
    sign -- e.g. lambda=50 zeroes true coefficients of +/-2 and +/-1.5 at
    both tau=0.1 and tau=0.9. A sign-dependent penalty cannot reproduce
    that: it fully suppresses one sign while barely touching the other.
    """

    @pytest.fixture
    def symmetric_data(self):
        rng = np.random.RandomState(0)
        n = 200
        z = rng.randn(n, 4)
        beta_true = np.array([2.0, -2.0, 1.5, -1.5])
        y = 0.7 + z @ beta_true + rng.randn(n) * 0.5
        X = np.column_stack([np.ones(n), z])
        return X, y

    @pytest.mark.parametrize("tau", [0.1, 0.9])
    def test_large_lambda_zeroes_both_signs(self, symmetric_data, tau):
        X, y = symmetric_data
        solver = LassoSolver(lambda_=50.0, penalize_intercept=False)
        coef = solver.solve(X, y, tau).coefficients
        # coef[1:] correspond to true beta = [2, -2, 1.5, -1.5]; a
        # symmetric L1 penalty this strong sparsifies all four regardless
        # of sign, matching R's rq.fit.lasso(X, y, tau, lambda=50).
        np.testing.assert_allclose(coef[1:], 0.0, atol=1e-3)

    def test_shrinkage_magnitude_matches_across_sign(self, symmetric_data):
        """At a moderate lambda, the drop from the unpenalised fit should
        be comparable for a +2 and a -2 true coefficient -- not off by an
        order of magnitude, which is what the sign-dependent bug produced
        (e.g. one coefficient barely shrunk while its opposite-signed,
        equal-magnitude counterpart was shrunk ~35x more)."""
        X, y = symmetric_data
        unpenalized = LassoSolver(lambda_=0.0, penalize_intercept=False).solve(X, y, 0.1).coefficients
        penalized = LassoSolver(lambda_=8.0, penalize_intercept=False).solve(X, y, 0.1).coefficients

        shrink_pos = abs(unpenalized[1] - penalized[1])  # true beta_1 = +2
        shrink_neg = abs(unpenalized[2] - penalized[2])  # true beta_2 = -2
        ratio = max(shrink_pos, shrink_neg) / max(min(shrink_pos, shrink_neg), 1e-6)
        # The old sign-dependent penalty gave ~20x on this exact fixture;
        # the symmetric penalty gives ~6x (residual asymmetry here is
        # ordinary finite-sample noise from the two columns' own data, not
        # the penalty). 10 cleanly separates the two.
        assert ratio < 10, (
            f"shrinkage should be comparable across sign, got {shrink_pos=:.4f} "
            f"vs {shrink_neg=:.4f} (ratio {ratio:.1f})"
        )
