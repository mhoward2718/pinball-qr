"""Multi-quantile preprocessing (Chernozhukov, Fernández-Val & Melly Alg. 2).

A warning about what these tests can and cannot show.

Preprocessing is *unconditionally* exact: the fixup loop verifies every globbed
sign and escalates until it is satisfied, so a badly wired warm start does not
produce wrong answers — it produces right answers slowly.  That means no
assertion on coefficients, objectives, or optimality can distinguish "CFM
implemented correctly" from "CFM implemented wrongly and rescued by the fixup
loop".  Those tests are still worth having (they pin the exactness guarantee),
but the test that actually exercises the warm start is
:meth:`TestWarmStartActuallyWarmStarts.test_warm_start_reduces_work`, which
asserts on counters.
"""

import numpy as np
import pytest

from tests._optimality import CERTIFIED, certify


def _has_native():
    try:
        from pinball._native import rqfnb  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_native(), reason="Fortran extension not built"
)

# Preprocessing builds rows that are sums over many observations, so the reduced
# problem is worse conditioned than the original and the inner solver's eps buys
# less accuracy.  Measured against the simplex optimum: objective within ~2e-8
# absolute, coefficients within ~5e-8.  See tests/test_optimality.py.
COEF_ATOL = 1e-6


def _problem(n=4000, p=4, seed=7):
    rng = np.random.RandomState(seed)
    X = np.column_stack([np.ones(n), rng.randn(n, p - 1)])
    y = X @ np.arange(1.0, p + 1) + rng.standard_t(3, n)
    return X, y


class TestExactness:
    """The grid path must agree with solving each τ on its own."""

    def test_matches_cold_per_tau_solves(self):
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem()
        taus = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        warm = PreprocessingSolver(random_state=0).solve_multi(X, y, taus)
        for tau, got in zip(taus, warm):
            cold = PreprocessingSolver(random_state=1).solve(X, y, tau)
            np.testing.assert_allclose(
                got.coefficients, cold.coefficients, atol=COEF_ATOL,
                err_msg=f"tau={tau}: warm-started fit differs from a cold one",
            )

    def test_matches_the_exact_simplex_optimum(self):
        """Objective values, which stay meaningful even where the argmin is not
        unique — the sharper assertion under degeneracy."""
        from pinball.linear.solvers import get_solver
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem()
        taus = [0.05, 0.5, 0.95]
        warm = PreprocessingSolver(random_state=0).solve_multi(X, y, taus)
        for tau, got in zip(taus, warm):
            exact = get_solver("br").solve(X, y, tau).objective_value
            assert got.objective_value >= exact - 1e-8, "beat the true optimum"
            assert (got.objective_value - exact) / abs(exact) < 1e-8

    @pytest.mark.parametrize("tau", [0.01, 0.5, 0.99])
    def test_every_fit_is_certified_optimal(self, tau):
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem()
        taus = [0.01, 0.2, 0.5, 0.8, 0.99]
        results = PreprocessingSolver(random_state=0).solve_multi(X, y, taus)
        got = results[taus.index(tau)]
        cert = certify(X, y, got.coefficients, tau, atol=1e-6)
        assert cert.status == CERTIFIED, f"tau={tau}: {cert}"

    def test_correctness_does_not_depend_on_grid_fineness(self):
        """A coarse grid is exactly as correct as a fine one — only slower.
        Exactness comes from the sign check, which knows nothing about spacing."""
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem()
        solo = PreprocessingSolver(random_state=0).solve(X, y, 0.5).coefficients
        coarse = PreprocessingSolver(random_state=0).solve_multi(
            X, y, [0.1, 0.5, 0.9])[1].coefficients
        fine_taus = list(np.round(np.linspace(0.1, 0.9, 33), 4))
        fine = PreprocessingSolver(random_state=0).solve_multi(
            X, y, fine_taus)[fine_taus.index(0.5)].coefficients
        np.testing.assert_allclose(coarse, solo, atol=COEF_ATOL)
        np.testing.assert_allclose(fine, solo, atol=COEF_ATOL)

    def test_results_come_back_in_input_order(self):
        """τ are sorted internally so each fit can seed the next; the caller
        must still get their own order back."""
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem()
        shuffled = [0.9, 0.1, 0.5, 0.75]
        got = PreprocessingSolver(random_state=0).solve_multi(X, y, shuffled)
        assert len(got) == len(shuffled)
        for tau, res in zip(shuffled, got):
            solo = PreprocessingSolver(random_state=0).solve(X, y, tau)
            np.testing.assert_allclose(
                res.coefficients, solo.coefficients, atol=COEF_ATOL,
                err_msg=f"result for tau={tau} is not in the requested position",
            )

    def test_repeated_tau_is_allowed(self):
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem()
        got = PreprocessingSolver(random_state=0).solve_multi(X, y, [0.5, 0.5])
        assert len(got) == 2
        np.testing.assert_allclose(
            got[0].coefficients, got[1].coefficients, atol=1e-12
        )

    def test_invalid_tau_rejected_wherever_it_sits(self):
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem(n=500)
        with pytest.raises(ValueError, match="tau"):
            PreprocessingSolver().solve_multi(X, y, [0.5, 1.5])

    def test_small_n_falls_back_to_full_data_solves(self):
        from pinball.linear.solvers import get_solver
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem(n=60, p=3)          # m0 >= n, so no preprocessing
        taus = [0.25, 0.5, 0.75]
        got = PreprocessingSolver(random_state=0).solve_multi(X, y, taus)
        for tau, res in zip(taus, got):
            exact = get_solver("fn").solve(X, y, tau).coefficients
            np.testing.assert_allclose(res.coefficients, exact, atol=1e-9)


class TestWarmStartActuallyWarmStarts:

    def test_warm_start_reduces_work(self):
        """The only test here that can tell a working warm start from a broken
        one, because exactness hides the difference everywhere else.

        The grid path does one preliminary subsample fit for the whole grid and
        seeds each τ with the previous fit's residuals; the cold path repeats
        that preliminary fit at every τ.  So the grid path must do strictly less
        inner-solver work.
        """
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem(n=20000, p=5)
        taus = list(np.round(np.linspace(0.02, 0.98, 49), 4))

        warm = PreprocessingSolver(random_state=0).solve_multi(X, y, taus)
        cold = [PreprocessingSolver(random_state=0).solve(X, y, t) for t in taus]

        warm_iters = sum(r.iterations for r in warm)
        cold_iters = sum(r.iterations for r in cold)
        warm_rows = sum(r.solver_info["n_globbed"] for r in warm)
        cold_rows = sum(r.solver_info["n_globbed"] for r in cold)

        assert warm_iters < cold_iters, (
            f"warm start did not reduce inner iterations "
            f"({warm_iters} vs {cold_iters}) — is the residual carry wired up?"
        )
        assert warm_rows <= cold_rows, (
            f"warm start did not shrink the reduced problems "
            f"({warm_rows} vs {cold_rows} rows)"
        )

    def test_grid_path_is_flagged_in_solver_info(self):
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem()
        got = PreprocessingSolver(random_state=0).solve_multi(X, y, [0.25, 0.75])
        for res in got:
            assert res.solver_info["warm_started"] is True
            assert "fixups" in res.solver_info
            assert "n_globbed" in res.solver_info

    def test_solver_advertises_batch_support(self):
        from pinball.linear.solvers.pfn import PreprocessingSolver

        assert PreprocessingSolver.supports_multiple_quantiles() is True


class TestReproducibility:

    def test_same_seed_gives_identical_fits(self):
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem()
        a = PreprocessingSolver(random_state=3).solve_multi(X, y, [0.25, 0.5])
        b = PreprocessingSolver(random_state=3).solve_multi(X, y, [0.25, 0.5])
        for ra, rb in zip(a, b):
            np.testing.assert_allclose(ra.coefficients, rb.coefficients, atol=0.0)

    def test_different_seeds_agree_on_the_answer(self):
        """The seed picks the subsample, so it changes the path, not the fit."""
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem()
        a = PreprocessingSolver(random_state=1).solve(X, y, 0.3)
        b = PreprocessingSolver(random_state=99).solve(X, y, 0.3)
        np.testing.assert_allclose(
            a.coefficients, b.coefficients, atol=COEF_ATOL
        )

    def test_seed_reaches_the_solver_through_the_estimator(self):
        """`get_solver` takes no constructor kwargs, so `solver_options` is the
        route — check it actually works end to end."""
        from pinball.linear import QuantileRegressor

        X, y = _problem()
        Xf = X[:, 1:]
        a = QuantileRegressor(tau=0.4, method="pfn",
                              solver_options={"random_state": 5}).fit(Xf, y)
        b = QuantileRegressor(tau=0.4, method="pfn",
                              solver_options={"random_state": 5}).fit(Xf, y)
        np.testing.assert_allclose(a.coef_, b.coef_, atol=0.0)


class TestTermination:

    def test_exhausted_fixup_budget_still_terminates(self):
        """With no fixup rounds allowed, the old code left `m` unchanged and
        redrew the same-sized subsample forever.  It must escalate instead."""
        from pinball.linear.solvers.pfn import PreprocessingSolver

        X, y = _problem(n=1500, p=3)
        solver = PreprocessingSolver(max_bad_fixups=0, random_state=0)
        with pytest.warns(UserWarning):
            result = solver.solve(X, y, 0.5)
        assert np.all(np.isfinite(result.coefficients))
        cert = certify(X, y, result.coefficients, 0.5, atol=1e-6)
        assert cert.status == CERTIFIED, str(cert)
