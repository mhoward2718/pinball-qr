"""Tests for the optimality oracle, and for the solvers it judges.

The first class tests the tester: an oracle that never fails is worthless, so
the negative controls here matter more than the positive ones.
"""

import numpy as np
import pytest

from tests._optimality import CERTIFIED, FAILED, INCONCLUSIVE, certify, certify_dual


def _has_native():
    try:
        from pinball._native import rqfnb  # noqa: F401
        return True
    except Exception:
        return False


def _problem(n=400, p=4, seed=0, scale=1.0):
    rng = np.random.RandomState(seed)
    X = np.column_stack([np.ones(n), rng.randn(n, p - 1)])
    y = scale * (X @ np.arange(1.0, p + 1) + rng.randn(n))
    return X, y


native_only = pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")


@native_only
class TestCertifierItself:
    """Negative controls first -- these are what give the oracle teeth."""

    @pytest.mark.parametrize("tau", [0.01, 0.1, 0.5, 0.9, 0.99])
    def test_certifies_a_real_solution(self, tau):
        from pinball.linear.solvers.fnb import FNBSolver

        X, y = _problem()
        beta = FNBSolver().solve(X, y, tau).coefficients
        cert = certify(X, y, beta, tau)
        assert cert.status == CERTIFIED, str(cert)

    @pytest.mark.parametrize("tau", [0.01, 0.5, 0.99])
    @pytest.mark.parametrize("bump", [1e-3, 1e-5])
    def test_rejects_a_perturbed_solution(self, tau, bump):
        """The essential negative control.  A perturbed beta keeps the same
        residual *signs*, so the dual stays feasible and dual feasibility alone
        would still accept it -- only the interpolation condition catches it.
        (Measured: with the dual check alone, a 1e-6 bump passed at 4 of 5 tau.)"""
        from pinball.linear.solvers.fnb import FNBSolver

        X, y = _problem()
        beta = FNBSolver().solve(X, y, tau).coefficients.copy()
        beta[1] += bump
        cert = certify(X, y, beta, tau)
        assert cert.status != CERTIFIED, f"bump={bump} slipped through: {cert}"

    def test_resolution_floor_is_documented_not_accidental(self):
        """The oracle cannot see error below the solver's own tolerance, and
        pretending otherwise would make it a flaky test rather than a strict one.

        With FNB at eps=1e-6 and this problem's scale, a 1e-7 coefficient bump
        lands at ~9e-9 relative -- under the 1e-8 interpolation tolerance -- so
        it certifies.  That is the honest floor: use R parity or a cold-solve
        comparison when finer resolution is needed.
        """
        from pinball.linear.solvers.fnb import FNBSolver

        X, y = _problem()
        beta = FNBSolver().solve(X, y, 0.5).coefficients.copy()

        beta_tiny = beta.copy()
        beta_tiny[1] += 1e-9
        assert certify(X, y, beta_tiny, 0.5).status == CERTIFIED

        beta_seen = beta.copy()
        beta_seen[1] += 1e-4
        assert certify(X, y, beta_seen, 0.5).status != CERTIFIED

    def test_rejects_a_solution_for_the_wrong_quantile(self):
        from pinball.linear.solvers.fnb import FNBSolver

        X, y = _problem()
        beta = FNBSolver().solve(X, y, 0.5).coefficients
        assert certify(X, y, beta, 0.9).status == FAILED

    def test_rejects_non_finite(self):
        X, y = _problem()
        assert certify(X, y, np.full(X.shape[1], np.nan), 0.5).status == FAILED

    def test_inconclusive_not_certified_on_rank_deficiency(self):
        """A duplicated column must never come back CERTIFIED."""
        X, y = _problem(n=200, p=3)
        X = np.column_stack([X, X[:, 1]])          # exact collinearity
        beta = np.zeros(X.shape[1])
        assert certify(X, y, beta, 0.5).status in (INCONCLUSIVE, FAILED)

    def test_inconclusive_when_basis_has_no_gap(self):
        """An arbitrary beta has no interpolated basis; report inconclusive
        rather than inventing a verdict."""
        X, y = _problem()
        cert = certify(X, y, np.ones(X.shape[1]), 0.5)
        assert cert.status in (INCONCLUSIVE, FAILED)
        assert cert.status != CERTIFIED

    def test_scale_invariance_of_the_verdict(self):
        """Multiplying y by 1e6 must not change the verdict -- the tolerance is
        relative, and a certifier that only works at unit scale is a trap."""
        from pinball.linear.solvers.fnb import FNBSolver

        for scale in (1e-3, 1.0, 1e6):
            X, y = _problem(scale=scale)
            beta = FNBSolver().solve(X, y, 0.5).coefficients
            assert certify(X, y, beta, 0.5).status == CERTIFIED, f"scale={scale}"


@native_only
class TestSolversAreOptimal:
    """Apply the oracle to every exact solver, including at extreme tau."""

    #: `pfn` needs a looser interpolation tolerance than the direct solvers, and
    #: the reason is structural rather than a defect: preprocessing replaces
    #: whole groups of observations with a single *summed* pseudo-observation,
    #: so the reduced design contains rows whose norm is orders of magnitude
    #: larger than an ordinary row.  The inner solver's eps therefore buys less
    #: accuracy on the reduced problem than it would on the original.  Measured
    #: over six subsampling seeds at tau=0.5: pfn's objective sits within 1.9e-8
    #: of the simplex optimum (~1.7e-10 relative) while its interpolation
    #: residual reaches 5e-8, i.e. above the 1e-8 default.
    _ATOL = {"br": 1e-8, "fn": 1e-8, "fnb": 1e-8, "pfn": 1e-6}

    @pytest.mark.parametrize("method", ["br", "fn", "fnb", "pfn"])
    @pytest.mark.parametrize("tau", [0.01, 0.25, 0.5, 0.75, 0.99])
    def test_solver_returns_an_optimal_fit(self, method, tau):
        from pinball.linear.solvers import get_solver

        X, y = _problem(n=300, p=3)
        beta = get_solver(method).solve(X, y, tau).coefficients
        cert = certify(X, y, beta, tau, atol=self._ATOL[method])
        assert cert.status == CERTIFIED, f"{method} at tau={tau}: {cert}"

    @pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
    def test_pfn_objective_matches_the_exact_optimum(self, tau):
        """Preprocessing is meant to be exact, so judge it on the objective --
        which is well defined even when the argmin is not -- rather than on the
        interpolation residual, which its summed glob rows inflate."""
        from pinball.linear.solvers import get_solver

        X, y = _problem(n=300, p=3)
        exact = get_solver("br").solve(X, y, tau).objective_value
        got = get_solver("pfn").solve(X, y, tau).objective_value
        assert got >= exact - 1e-9, "pfn beat the exact optimum -- impossible"
        assert (got - exact) / max(abs(exact), 1.0) < 1e-8

    def test_engel_fit_is_optimal(self):
        """The dataset the reference coefficients come from."""
        from pinball.datasets import load_engel
        from pinball.linear.solvers.fnb import FNBSolver

        data = load_engel()
        X = np.column_stack([np.ones(len(data.target)), data.data])
        y = data.target
        for tau in (0.1, 0.5, 0.9):
            beta = FNBSolver().solve(X, y, tau).coefficients
            assert certify(X, y, beta, tau).status == CERTIFIED

    @pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
    def test_solver_supplied_dual_screens_clean(self, tau):
        from pinball.linear.solvers.fnb import FNBSolver

        X, y = _problem(n=300, p=3)
        result = FNBSolver().solve(X, y, tau)
        cert = certify_dual(X, result.residuals, result.dual_solution, tau)
        assert cert.status == CERTIFIED, str(cert)
