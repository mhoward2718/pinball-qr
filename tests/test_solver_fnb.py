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


# ──────────────────────────────────────────────────────────────────────
# Regression tests: failure must never be silent
#
# `lpfnb` (fortran/rqfnb.f) leaves its iteration loop on either a small
# duality gap *or* an exhausted budget, and only the former is a success.
# It never sets `info` in the latter case -- `info` is written only by
# stepy's dposv -- so before these checks existed an unconverged fit was
# indistinguishable from a good one.  Observed in practice: a warm-started
# solve returned coefficients wrong in the second decimal, silently.
#
# The iteration limit is hard to provoke with real data (the interior
# point drives the gap to true zero, so *any* positive eps is eventually
# met), so the budget path is exercised by faking the Fortran return.
# ──────────────────────────────────────────────────────────────────────

class TestFNBReportsFailure:

    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(42)
        n, p = 50, 3
        X = rng.randn(n, p)
        y = X @ [1, 2, 3] + rng.randn(n) * 0.5
        return X, y

    def test_non_positive_eps_rejected(self):
        """eps <= 0 makes lpfnb's loop test unsatisfiable and it returns NaN
        coefficients with info = 0.  Reject the parameter instead."""
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="eps must be positive"):
                FNBSolver(eps=bad)

    @pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
    def test_converged_solve_is_clean(self, data):
        """A well-posed solve must not trip either guard."""
        X, y = data
        result = FNBSolver().solve(X, y, tau=0.5)
        assert result.status == 0
        assert result.solver_info["converged"] is True
        assert result.iterations < 500

    @pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
    def test_dual_solution_certifies_optimality(self, data):
        """The returned dual must satisfy X' a = 0 with a in [tau-1, tau] --
        that is exactly the subgradient optimality condition, so a caller can
        check the fit without re-solving."""
        X, y = data
        for tau in (0.1, 0.5, 0.9):
            result = FNBSolver().solve(X, y, tau=tau)
            a = result.dual_solution
            assert a is not None
            assert a.min() >= (tau - 1) - 1e-9
            assert a.max() <= tau + 1e-9
            scale = np.abs(X).sum(axis=0).max()
            np.testing.assert_allclose(X.T @ a / scale, 0.0, atol=1e-9)

    @pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
    def test_iteration_limit_warns_and_flags(self, data, monkeypatch):
        """Budget exhausted -> warn, and report it in status."""
        from pinball.linear.solvers import fnb as fnb_mod

        X, y = data
        n, p = X.shape

        def fake_rqfnb(a, c, rhs, d, u, beta, eps, wn, wp, nit, info):
            nit_out = np.array([fnb_mod._LPFNB_MAXIT, 3, n], dtype=np.int32)
            return (a, c, rhs, d, u, wn, wp, nit_out, np.int32(0))

        monkeypatch.setattr("pinball._native.rqfnb", fake_rqfnb, raising=False)
        with pytest.warns(UserWarning, match="iteration limit"):
            result = FNBSolver().solve(X, y, tau=0.5)
        assert result.status == fnb_mod.STATUS_NOT_CONVERGED
        assert result.solver_info["converged"] is False

    @pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
    def test_non_finite_coefficients_warn_and_flag(self, data, monkeypatch):
        """NaN coefficients with info = 0 must not pass as a normal result."""
        from pinball.linear.solvers import fnb as fnb_mod

        X, y = data
        n, p = X.shape

        def fake_rqfnb(a, c, rhs, d, u, beta, eps, wn, wp, nit, info):
            wp_out = np.zeros((p, p + 3), order="F")
            wp_out[:, 0] = np.nan
            return (a, c, rhs, d, u, wn, wp_out,
                    np.array([7, 3, n], dtype=np.int32), np.int32(0))

        monkeypatch.setattr("pinball._native.rqfnb", fake_rqfnb, raising=False)
        with pytest.warns(UserWarning, match="non-finite"):
            result = FNBSolver().solve(X, y, tau=0.5)
        assert result.status == fnb_mod.STATUS_NOT_FINITE
