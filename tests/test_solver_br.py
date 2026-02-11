"""Tests for the Barrodale-Roberts solver.

These tests run without requiring the Fortran extension by mocking the
native ``rqbr`` call.  Integration tests that actually call Fortran are
marked with ``@pytest.mark.fortran`` and skipped when the extension is
not available.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from pinball.solvers.br import BRSolver, _derive_br_params, _get_wls_weights
from pinball.solvers.base import SolverResult


# ──────────────────────────────────────────────────────────────────────
# Parameter derivation
# ──────────────────────────────────────────────────────────────────────

class TestDeriveBRParams:

    def test_single_quantile_shapes(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [10, 20, 30],
                       [1, 10, 100], [7, 5, 3]], dtype=np.float64)
        y = np.array([10, 100, 1000, 5, 25], dtype=np.float64)
        params = _derive_br_params(X, y, tau=0.5)

        assert params["m"] == 5
        assert params["nn"] == np.int32(3)
        assert params["m5"] == np.int32(10)
        assert params["n3"] == np.int32(6)
        assert params["n4"] == np.int32(7)
        assert params["t"] == 0.5
        assert params["nsol"] == np.int32(2)
        assert params["ndsol"] == np.int32(2)
        assert params["sol"].shape == (6, 2)
        assert params["dsol"].shape == (5, 2)
        assert params["ci"].shape == (4, 3)
        assert params["tnmat"].shape == (4, 3)

    def test_full_process_shapes(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [10, 20, 30],
                       [1, 10, 100], [7, 5, 3]], dtype=np.float64)
        y = np.array([10, 100, 1000, 5, 25], dtype=np.float64)
        params = _derive_br_params(X, y, tau=None)

        assert params["t"] == -1.0
        assert params["nsol"] == np.int32(15)
        assert params["ndsol"] == np.int32(15)
        assert params["sol"].shape == (6, 15)
        assert params["dsol"].shape == (5, 15)

    def test_tolerance(self):
        X = np.eye(3, dtype=np.float64)
        X = np.vstack([X, X])  # 6 x 3
        y = np.ones(6, dtype=np.float64)
        params = _derive_br_params(X, y, tau=0.5)
        expected_tol = np.finfo(np.float64).eps ** (2.0 / 3.0)
        assert abs(params["toler"] - expected_tol) < 1e-20

    def test_arrays_are_correct_dtypes(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y = np.array([1, 2, 3, 4], dtype=np.float64)
        params = _derive_br_params(X, y, tau=0.5)
        assert params["a"].dtype == np.float64
        assert params["s"].dtype == np.int32
        assert params["h"].dtype == np.int32


# ──────────────────────────────────────────────────────────────────────
# BRSolver (with mocked Fortran)
# ──────────────────────────────────────────────────────────────────────

class TestBRSolver:

    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(42)
        n, p = 50, 3
        X = rng.randn(n, p)
        y = X @ [1, 2, 3] + rng.randn(n) * 0.5
        return X, y

    def _mock_rqbr_return(self, n, p):
        """Fake rqbr output tuple (11 elements)."""
        return (
            0,                                # flag
            np.array([1.0, 2.0, 3.0]),        # coef
            np.zeros(n),                      # resid
            np.zeros((p + 3, 2)),             # sol
            np.zeros((n, 2)),                 # dsol
            np.int32(0),                      # lsol
            np.zeros((p, 2), dtype=np.int32), # h
            np.zeros(p),                      # qn
            np.float64(0),                    # cutoff
            np.zeros((4, p)),                 # ci
            np.zeros((4, p)),                 # tnmat
        )

    def test_solve_returns_solver_result(self, data):
        X, y = data
        n, p = X.shape
        solver = BRSolver()
        with patch("pinball.solvers.br.rqbr", create=True) as mock:
            mock.return_value = self._mock_rqbr_return(n, p)
            with patch.dict("sys.modules", {"pinball._native": MagicMock(rqbr=mock)}):
                with patch("pinball.solvers.br.BRSolver._solve_impl") as mock_impl:
                    # Direct mock of the implementation
                    mock_impl.return_value = SolverResult(
                        coefficients=np.array([1.0, 2.0, 3.0]),
                        residuals=np.zeros(n),
                    )
                    result = solver.solve(X, y, tau=0.5)
        assert isinstance(result, SolverResult)

    def test_singular_matrix_raises(self):
        X = np.array([[1, 1, 3], [4, 4, 6], [10, 10, 30],
                       [1, 1, 100], [7, 7, 3]], dtype=np.float64)
        y = np.array([10, 100, 1000, 5, 25], dtype=np.float64)
        solver = BRSolver()
        with pytest.raises(ValueError, match="Singular design matrix"):
            solver.solve(X, y, tau=0.5)

    def test_supports_multiple_quantiles(self):
        assert BRSolver.supports_multiple_quantiles() is True


# ──────────────────────────────────────────────────────────────────────
# WLS weights (unit-testable without Fortran)
# ──────────────────────────────────────────────────────────────────────

class TestGetWLSWeights:

    def test_weights_positive(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y = np.array([1, 2, 3, 4], dtype=np.float64)

        solver_mock = MagicMock()
        solver_mock.solve.side_effect = [
            SolverResult(coefficients=np.array([0.5, 0.6]), residuals=y * 0),
            SolverResult(coefficients=np.array([0.3, 0.4]), residuals=y * 0),
        ]

        with patch("pinball.solvers.br.BRSolver", return_value=solver_mock):
            weights = _get_wls_weights(X, y, tau=0.5)

        assert np.all(weights > 0)

    def test_eps_floor(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y = np.array([1, 2, 3, 4], dtype=np.float64)

        # Coefficients that produce zero dyhat
        solver_mock = MagicMock()
        solver_mock.solve.side_effect = [
            SolverResult(coefficients=np.array([1.0, 1.0]), residuals=y * 0),
            SolverResult(coefficients=np.array([1.0, 1.0]), residuals=y * 0),
        ]
        eps = np.finfo(np.float64).eps ** (2.0 / 3.0)

        with patch("pinball.solvers.br.BRSolver", return_value=solver_mock):
            weights = _get_wls_weights(X, y, tau=0.5)

        # When dyhat is zero, weights should be at least eps
        assert np.all(weights >= eps)


# ──────────────────────────────────────────────────────────────────────
# Integration tests (require compiled Fortran)
# ──────────────────────────────────────────────────────────────────────

def _has_native():
    try:
        from pinball._native import rqbr
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
class TestBRSolverIntegration:
    """End-to-end tests using the real Fortran solver."""

    def test_engel_median(self):
        from pinball.datasets import load_engel
        data = load_engel()
        X = np.column_stack([np.ones(len(data.target)), data.data])
        y = data.target
        solver = BRSolver()
        result = solver.solve(X, y, tau=0.5)

        assert result.status == 0
        assert result.coefficients.shape == (2,)
        # Known R result: intercept ≈ 81.48, slope ≈ 0.5602
        np.testing.assert_allclose(result.coefficients, [81.48, 0.5602], atol=1.0)
