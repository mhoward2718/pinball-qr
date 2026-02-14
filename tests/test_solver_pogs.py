"""Tests for the POGS ADMM solver for quantile regression.

TDD: These tests were written *before* the implementation.
They verify the ``POGSSolver`` class against the ``BaseSolver`` contract
and ensure correct formulation of the quantile regression problem in
POGS graph-form.

Test structure (SOLID)
---------------------
* ``TestPOGSSolverContract`` — Liskov Substitution: POGSSolver is a
  proper BaseSolver subclass.
* ``TestPOGSSolverValidation`` — Single Responsibility: solver-specific
  validation logic.
* ``TestPOGSGraphFormFormulation`` — verifies the mathematical mapping
  from (X, y, tau) to POGS FunctionObj arrays is correct.
* ``TestPOGSSolverIntegration`` — end-to-end tests with the native
  POGS library (skipped if the native library is not built).
"""

from unittest.mock import patch

import numpy as np
import pytest

from pinball.linear.solvers.base import BaseSolver, SolverResult

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _has_native_pogs():
    """Return True if the native _pogs_native library is available."""
    try:
        from pinball.linear.solvers.pogs import _find_native_library
        _find_native_library()
        return True
    except OSError:
        return False


def _make_data(n=50, p=3, seed=42):
    """Generate simple regression data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    true_beta = rng.randn(p)
    y = X @ true_beta + rng.randn(n) * 0.5
    return X, y


# ──────────────────────────────────────────────────────────────────────
# Contract tests — Liskov Substitution
# ──────────────────────────────────────────────────────────────────────

class TestPOGSSolverContract:
    """POGSSolver must be a proper BaseSolver subclass."""

    def test_is_base_solver_subclass(self):
        from pinball.linear.solvers.pogs import POGSSolver
        assert issubclass(POGSSolver, BaseSolver)

    def test_instantiation_default_params(self):
        from pinball.linear.solvers.pogs import POGSSolver
        solver = POGSSolver()
        assert solver.abs_tol == 1e-4
        assert solver.rel_tol == 1e-4
        assert solver.max_iter == 2500
        assert solver.rho == 1.0
        assert solver.verbose == 0
        assert solver.adaptive_rho is True

    def test_instantiation_custom_params(self):
        from pinball.linear.solvers.pogs import POGSSolver
        solver = POGSSolver(
            abs_tol=1e-6,
            rel_tol=1e-6,
            max_iter=5000,
            rho=2.0,
            verbose=1,
            adaptive_rho=False,
        )
        assert solver.abs_tol == 1e-6
        assert solver.rel_tol == 1e-6
        assert solver.max_iter == 5000
        assert solver.rho == 2.0
        assert solver.verbose == 1
        assert solver.adaptive_rho is False

    def test_supports_multiple_quantiles_is_false(self):
        from pinball.linear.solvers.pogs import POGSSolver
        assert POGSSolver.supports_multiple_quantiles() is False

    def test_solve_method_exists(self):
        from pinball.linear.solvers.pogs import POGSSolver
        solver = POGSSolver()
        assert callable(getattr(solver, "solve", None))


# ──────────────────────────────────────────────────────────────────────
# Validation tests — Single Responsibility
# ──────────────────────────────────────────────────────────────────────

class TestPOGSSolverValidation:
    """Solver-specific input validation."""

    def test_tau_zero_raises(self):
        from pinball.linear.solvers.pogs import POGSSolver
        X, y = _make_data()
        solver = POGSSolver()
        with pytest.raises(ValueError, match="tau must be in"):
            solver.solve(X, y, tau=0.0)

    def test_tau_one_raises(self):
        from pinball.linear.solvers.pogs import POGSSolver
        X, y = _make_data()
        solver = POGSSolver()
        with pytest.raises(ValueError, match="tau must be in"):
            solver.solve(X, y, tau=1.0)

    def test_incompatible_shapes_raises(self):
        from pinball.linear.solvers.pogs import POGSSolver
        X = np.ones((10, 3))
        y = np.ones(5)
        solver = POGSSolver()
        with pytest.raises(ValueError, match="incompatible shapes"):
            solver.solve(X, y, tau=0.5)

    def test_1d_X_raises(self):
        from pinball.linear.solvers.pogs import POGSSolver
        X = np.ones(10)
        y = np.ones(10)
        solver = POGSSolver()
        with pytest.raises(ValueError, match="2-D"):
            solver.solve(X, y, tau=0.5)

    def test_native_lib_missing_raises_os_error(self):
        import pinball.linear.solvers.pogs as pogs_mod
        from pinball.linear.solvers.pogs import POGSSolver

        X, y = _make_data()
        solver = POGSSolver()
        # Reset the cached library so it tries to reload
        old_lib = pogs_mod._lib
        pogs_mod._lib = None
        try:
            with patch(
                "pinball.linear.solvers.pogs._find_native_library",
                side_effect=OSError("not found"),
            ), pytest.raises(OSError, match="not found"):
                solver.solve(X, y, tau=0.5)
        finally:
            pogs_mod._lib = old_lib


# ──────────────────────────────────────────────────────────────────────
# Graph-form formulation tests
# ──────────────────────────────────────────────────────────────────────

class TestPOGSGraphFormFormulation:
    """Verify the quantile regression → POGS graph-form mapping.

    The pinball loss is: rho_tau(u) = 0.5 * |u| + (tau - 0.5) * u

    Standard QR minimises rho_tau(y - Xb), so in POGS graph-form
    with y_var = Xb:

    f_i(y_var) = 0.5 * |y_var - response_i| + (0.5 - tau) * y_var
    uses FunctionObj(kAbs, a=1, b=response_i, c=0.5, d=0.5-tau, e=0)

    And g_j(x_j) = 0 uses FunctionObj(kZero).
    """

    def test_build_f_functions_length(self):
        from pinball.linear.solvers.pogs import _build_graph_form
        X, y = _make_data(n=20, p=3)
        f, g = _build_graph_form(X, y, tau=0.5)
        assert len(f) == 20

    def test_build_g_functions_length(self):
        from pinball.linear.solvers.pogs import _build_graph_form
        X, y = _make_data(n=20, p=3)
        f, g = _build_graph_form(X, y, tau=0.5)
        assert len(g) == 3

    def test_f_function_type_is_kAbs(self):
        from pinball.linear.solvers.pogs import _build_graph_form
        X, y = _make_data(n=10, p=2)
        f, g = _build_graph_form(X, y, tau=0.5)
        # All f_i should use the kAbs base function
        for fi in f:
            assert fi.h == 0  # kAbs = 0 in the Function enum

    def test_g_function_type_is_kZero(self):
        from pinball.linear.solvers.pogs import _build_graph_form
        X, y = _make_data(n=10, p=2)
        f, g = _build_graph_form(X, y, tau=0.5)
        # All g_j should use the kZero base function
        for gj in g:
            assert gj.h == 15  # kZero = 15 in the Function enum

    def test_f_offsets_match_y(self):
        from pinball.linear.solvers.pogs import _build_graph_form
        y = np.array([1.0, 2.0, 3.0])
        X = np.ones((3, 1))
        f, g = _build_graph_form(X, y, tau=0.5)
        for i, fi in enumerate(f):
            assert fi.b == pytest.approx(y[i])

    def test_f_linear_term_at_median(self):
        """At tau=0.5, the linear term d should be 0."""
        from pinball.linear.solvers.pogs import _build_graph_form
        X, y = _make_data(n=10, p=2)
        f, g = _build_graph_form(X, y, tau=0.5)
        for fi in f:
            assert fi.d == pytest.approx(0.0)

    def test_f_linear_term_at_tau_0_25(self):
        """At tau=0.25, d = 0.5 - tau = 0.25."""
        from pinball.linear.solvers.pogs import _build_graph_form
        X, y = _make_data(n=10, p=2)
        f, g = _build_graph_form(X, y, tau=0.25)
        for fi in f:
            assert fi.d == pytest.approx(0.25)

    def test_f_linear_term_at_tau_0_75(self):
        """At tau=0.75, d = 0.5 - tau = -0.25."""
        from pinball.linear.solvers.pogs import _build_graph_form
        X, y = _make_data(n=10, p=2)
        f, g = _build_graph_form(X, y, tau=0.75)
        for fi in f:
            assert fi.d == pytest.approx(-0.25)

    def test_f_scale_c_is_half(self):
        """c = 0.5 in the kAbs formulation."""
        from pinball.linear.solvers.pogs import _build_graph_form
        X, y = _make_data(n=10, p=2)
        f, g = _build_graph_form(X, y, tau=0.5)
        for fi in f:
            assert fi.c == pytest.approx(0.5)

    def test_f_a_is_one(self):
        """a = 1.0 (no rescaling of the argument)."""
        from pinball.linear.solvers.pogs import _build_graph_form
        X, y = _make_data(n=10, p=2)
        f, g = _build_graph_form(X, y, tau=0.5)
        for fi in f:
            assert fi.a == pytest.approx(1.0)

    def test_f_e_is_zero(self):
        """e = 0.0 (no quadratic penalty)."""
        from pinball.linear.solvers.pogs import _build_graph_form
        X, y = _make_data(n=10, p=2)
        f, g = _build_graph_form(X, y, tau=0.5)
        for fi in f:
            assert fi.e == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────
# Mocked solve tests — Dependency Inversion
# ──────────────────────────────────────────────────────────────────────

class TestPOGSSolverMocked:
    """Tests using a mocked pogs backend to verify wiring without
    requiring the actual pogs C library."""

    def _mock_solve_graph_form(self, A, f, g, **kwargs):
        """Fake _solve_graph_form that returns realistic-looking output."""
        m, n = A.shape
        return {
            "x": np.ones(n),
            "y": A @ np.ones(n),
            "l": np.zeros(m),
            "optval": 0.42,
            "iterations": 100,
            "status": 0,
        }

    def test_returns_solver_result(self):
        from pinball.linear.solvers.pogs import POGSSolver
        X, y = _make_data()
        solver = POGSSolver()
        with patch(
            "pinball.linear.solvers.pogs._call_pogs",
            side_effect=self._mock_solve_graph_form,
        ):
            result = solver.solve(X, y, tau=0.5)
        assert isinstance(result, SolverResult)

    def test_result_coefficients_shape(self):
        from pinball.linear.solvers.pogs import POGSSolver
        X, y = _make_data(n=50, p=3)
        solver = POGSSolver()
        with patch(
            "pinball.linear.solvers.pogs._call_pogs",
            side_effect=self._mock_solve_graph_form,
        ):
            result = solver.solve(X, y, tau=0.5)
        assert result.coefficients.shape == (3,)

    def test_result_residuals_shape(self):
        from pinball.linear.solvers.pogs import POGSSolver
        X, y = _make_data(n=50, p=3)
        solver = POGSSolver()
        with patch(
            "pinball.linear.solvers.pogs._call_pogs",
            side_effect=self._mock_solve_graph_form,
        ):
            result = solver.solve(X, y, tau=0.5)
        assert result.residuals.shape == (50,)

    def test_result_objective_value_set(self):
        from pinball.linear.solvers.pogs import POGSSolver
        X, y = _make_data()
        solver = POGSSolver()
        with patch(
            "pinball.linear.solvers.pogs._call_pogs",
            side_effect=self._mock_solve_graph_form,
        ):
            result = solver.solve(X, y, tau=0.5)
        assert result.objective_value == pytest.approx(0.42)

    def test_result_iterations_set(self):
        from pinball.linear.solvers.pogs import POGSSolver
        X, y = _make_data()
        solver = POGSSolver()
        with patch(
            "pinball.linear.solvers.pogs._call_pogs",
            side_effect=self._mock_solve_graph_form,
        ):
            result = solver.solve(X, y, tau=0.5)
        assert result.iterations == 100

    def test_solver_kwargs_forwarded(self):
        """Verify constructor params are forwarded to the POGS call."""
        from pinball.linear.solvers.pogs import POGSSolver
        X, y = _make_data()
        solver = POGSSolver(abs_tol=1e-8, max_iter=5000, rho=2.0)

        captured = {}

        def spy(A, f, g, **kwargs):
            captured.update(kwargs)
            return self._mock_solve_graph_form(A, f, g, **kwargs)

        with patch("pinball.linear.solvers.pogs._call_pogs", side_effect=spy):
            solver.solve(X, y, tau=0.5)

        assert captured["abs_tol"] == pytest.approx(1e-8)
        assert captured["max_iter"] == 5000
        assert captured["rho"] == pytest.approx(2.0)


# ──────────────────────────────────────────────────────────────────────
# Registry tests — Open/Closed Principle
# ──────────────────────────────────────────────────────────────────────

class TestPOGSRegistration:
    """The POGS solver should always be available via the solver registry
    since it uses a vendored native library."""

    def test_registered_as_pogs(self):
        from pinball.linear.solvers import list_solvers
        assert "pogs" in list_solvers()

    def test_get_solver_returns_pogs_instance(self):
        from pinball.linear.solvers import get_solver
        from pinball.linear.solvers.pogs import POGSSolver
        solver = get_solver("pogs")
        assert isinstance(solver, POGSSolver)

    def test_module_always_importable(self):
        """The POGS module should always be importable (no external deps)."""
        from pinball.linear.solvers.pogs import POGSSolver
        assert POGSSolver is not None


# ──────────────────────────────────────────────────────────────────────
# Integration tests (require pogs installed)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_native_pogs(), reason="native POGS library not built")
class TestPOGSSolverIntegration:
    """End-to-end tests using the real POGS solver."""

    def test_engel_median(self):
        """Median coefficients should match BR/FNB on Engel data."""
        from pinball.datasets import load_engel
        from pinball.linear.solvers.pogs import POGSSolver

        data = load_engel()
        X = np.column_stack([np.ones(len(data.target)), data.data])
        y = data.target
        solver = POGSSolver(abs_tol=1e-4, rel_tol=1e-3, max_iter=10000)
        result = solver.solve(X, y, tau=0.5)

        assert result.status == 0
        assert result.coefficients.shape == (2,)
        # Compare to known R result: intercept ≈ 81.48, slope ≈ 0.5602
        np.testing.assert_allclose(
            result.coefficients, [81.48, 0.5602], atol=2.0
        )

    def test_simple_data_tau_0_5(self):
        """On perfectly linear data, coefficients should be close to true."""
        from pinball.linear.solvers.pogs import POGSSolver

        rng = np.random.RandomState(123)
        n = 200
        X = np.column_stack([np.ones(n), rng.randn(n)])
        true_beta = np.array([3.0, 2.0])
        y = X @ true_beta + rng.randn(n) * 0.01

        solver = POGSSolver(abs_tol=1e-4, rel_tol=1e-3, max_iter=10000)
        result = solver.solve(X, y, tau=0.5)

        assert result.status == 0
        np.testing.assert_allclose(result.coefficients, true_beta, atol=0.1)

    def test_different_quantiles(self):
        """Higher quantile → larger intercept on location-shift data."""
        from pinball.linear.solvers.pogs import POGSSolver

        rng = np.random.RandomState(99)
        n = 300
        X = np.column_stack([np.ones(n), rng.randn(n)])
        y = X @ [5.0, 1.0] + rng.randn(n).clip(-3, 3)

        solver = POGSSolver(abs_tol=1e-4, rel_tol=1e-3, max_iter=10000)
        intercepts = {}
        for tau in [0.1, 0.5, 0.9]:
            result = solver.solve(X, y, tau=tau)
            assert result.status == 0
            intercepts[tau] = result.coefficients[0]

        # For location-shift model, the intercept should be monotone in tau
        assert intercepts[0.1] < intercepts[0.5] < intercepts[0.9]

    def test_agrees_with_br_solver(self):
        """POGS and BR should give similar coefficients on well-scaled data."""
        from pinball.linear.solvers.br import BRSolver
        from pinball.linear.solvers.pogs import POGSSolver

        # Use well-conditioned synthetic data (Engel has a huge condition
        # number due to the intercept column vs. income ~400–5000, which
        # makes ADMM convergence slow on the extrapolated intercept).
        rng = np.random.RandomState(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.randn(n)])
        true_beta = np.array([5.0, 2.0])
        y = X @ true_beta + rng.randn(n)

        pogs_solver = POGSSolver(abs_tol=1e-4, rel_tol=1e-3, max_iter=10000)
        br_solver = BRSolver()

        for tau in [0.25, 0.5, 0.75]:
            pogs_result = pogs_solver.solve(X, y, tau=tau)
            br_result = br_solver.solve(X, y, tau=tau)
            np.testing.assert_allclose(
                pogs_result.coefficients,
                br_result.coefficients,
                atol=0.5,
                err_msg=f"tau={tau}: POGS and BR disagree",
            )

    def test_estimator_integration(self):
        """POGSSolver works through the QuantileRegressor interface."""
        from pinball import QuantileRegressor
        from pinball.datasets import load_engel

        data = load_engel()
        model = QuantileRegressor(tau=0.5, method="pogs")
        model.fit(data.data, data.target)
        assert hasattr(model, "coef_")
        preds = model.predict(data.data)
        assert preds.shape == data.target.shape
