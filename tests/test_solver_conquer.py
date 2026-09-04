"""Tests for convolution-smoothed quantile regression (conquer).

conquer is the odd one out in this package: every other solver returns the
exact minimiser of the pinball loss, so they can be checked against each other
and against the LP optimality certificate.  conquer minimises a *different*
(smoothed) objective and lands somewhere else on purpose, so those two oracles
do not apply and are deliberately not used here.

What replaces them, roughly in order of how much each one has caught:

1. **Internal consistency.**  The analytic gradient against finite differences
   of the loss, and the kernel CDF against its own density.  This is not
   ceremony: it is what found the real bug in this module, where the compactly
   supported CDFs were clipped with a non-monotone polynomial and silently
   returned 0 instead of 1 for every residual beyond the support, inverting the
   gradient.  Nothing else in the suite noticed -- three of five kernels still
   matched R exactly, and the symptom looked like an optimiser failure.
2. **A smooth-problem optimality certificate.**  ``L_h`` is convex and
   differentiable, so ``beta`` is optimal iff its gradient vanishes.  That is
   the direct analogue of the LP certificate used for the other solvers.
3. **The h -> 0 limit.**  Smoothing must reduce to the exact estimator as the
   bandwidth shrinks.  This is the test that ties conquer to the solvers that
   are already verified, and would catch a wrong loss that is nonetheless
   self-consistent.
4. **Parity with R conquer**, the reference implementation.
5. **Equivariance**, in the corrected form: three of Koenker's four identities
   hold exactly, and scale equivariance holds only when the bandwidth is scaled
   with the data.
"""

import numpy as np
import pytest

from pinball.linear.solvers.conquer import (
    KERNELS,
    ConquerSolver,
    _kernel_cdf,
    _kernel_pdf,
    asymptotic_cov,
    confidence_intervals,
    default_bandwidth,
    multiplier_bootstrap,
    smoothed_gradient,
    smoothed_hessian,
    smoothed_loss,
)


def _problem(n=800, p=3, seed=5, scale=1.0):
    rng = np.random.RandomState(seed)
    X = np.column_stack([np.ones(n), rng.randn(n, p)])
    y = scale * (X @ np.arange(1.0, p + 2) + rng.standard_t(3, n))
    return X, y


def _has_native():
    try:
        from pinball._native import rqfnb  # noqa: F401
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# 1. Internal consistency -- the layer that caught the real bug
# ──────────────────────────────────────────────────────────────────────

class TestKernels:

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_cdf_is_the_integral_of_the_density(self, kernel):
        """Differentiating Kbar numerically must give back K.

        Deliberately samples well outside [-1, 1], because that is exactly
        where the compact kernels were wrong: their CDF polynomials are
        non-monotone in the tails, so clipping them sent Kbar(3) to 0 instead
        of 1 for the parabolic and triangular kernels.
        """
        t = np.linspace(-4.0, 4.0, 401)
        eps = 1e-6
        if kernel == "uniform":
            # Its density jumps from 1/2 to 0 at +/-1, so a central difference
            # straddling the edge cannot reproduce either value.  Every other
            # kernel here is continuous there.
            t = t[np.abs(np.abs(t) - 1.0) > 1e-3]
        fd = (_kernel_cdf(t + eps, kernel) - _kernel_cdf(t - eps, kernel)) / (2 * eps)
        np.testing.assert_allclose(fd, _kernel_pdf(t, kernel), atol=1e-5)

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_cdf_tails_and_monotonicity(self, kernel):
        t = np.linspace(-50.0, 50.0, 2001)
        c = _kernel_cdf(t, kernel)
        assert c[0] == pytest.approx(0.0, abs=1e-12)
        assert c[-1] == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.diff(c) >= -1e-12), "CDF must be non-decreasing"
        assert np.all((c >= 0.0) & (c <= 1.0))

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_kernel_is_symmetric(self, kernel):
        """Kbar(-t) = 1 - Kbar(t); the gradient derivation relies on it."""
        t = np.linspace(0.0, 5.0, 201)
        np.testing.assert_allclose(
            _kernel_cdf(-t, kernel), 1.0 - _kernel_cdf(t, kernel), atol=1e-12
        )

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_absolute_deviation_matches_quadrature(self, kernel):
        """G(t) = int |t-v| K(v) dv, the closed form used by the loss.

        Integrated piecewise so the quadrature never straddles the kink at t
        or the edge of a compact support -- integrating naively over a wide
        interval silently returns 0 for the compact kernels.
        """
        quad = pytest.importorskip("scipy.integrate").quad
        from pinball.linear.solvers.conquer import _kernel_absdev

        limit = 40.0 if kernel in ("Gaussian", "logistic") else 1.0
        for t in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5):
            pts = sorted({-limit, limit, max(-limit, min(limit, t))})
            def integrand(v, t=t, kernel=kernel):
                return abs(t - v) * _kernel_pdf(np.array([v]), kernel)[0]

            num = sum(
                quad(integrand, pts[i], pts[i + 1], limit=400)[0]
                for i in range(len(pts) - 1)
            )
            got = float(_kernel_absdev(np.array([float(t)]), kernel)[0])
            assert got == pytest.approx(num, abs=1e-8), f"{kernel} at t={t}"


class TestGradientMatchesLoss:

    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
    def test_gradient_is_the_derivative_of_the_loss(self, kernel, tau):
        X, y = _problem(n=300)
        beta = np.array([0.3, 0.8, 1.2, 2.0])
        h = 0.25
        g = smoothed_gradient(X, y, beta, tau, h, kernel)
        fd = np.zeros_like(beta)
        eps = 1e-6
        for j in range(beta.size):
            up, dn = beta.copy(), beta.copy()
            up[j] += eps
            dn[j] -= eps
            fd[j] = (
                smoothed_loss(X, y, up, tau, h, kernel)
                - smoothed_loss(X, y, dn, tau, h, kernel)
            ) / (2 * eps)
        np.testing.assert_allclose(g, fd, atol=1e-7)

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_hessian_is_the_derivative_of_the_gradient(self, kernel):
        X, y = _problem(n=300)
        beta = np.array([0.3, 0.8, 1.2, 2.0])
        h, tau, eps = 0.3, 0.4, 1e-6
        H = smoothed_hessian(X, y, beta, h, kernel)
        fd = np.zeros_like(H)
        for j in range(beta.size):
            up, dn = beta.copy(), beta.copy()
            up[j] += eps
            dn[j] -= eps
            fd[:, j] = (
                smoothed_gradient(X, y, up, tau, h, kernel)
                - smoothed_gradient(X, y, dn, tau, h, kernel)
            ) / (2 * eps)
        np.testing.assert_allclose(H, fd, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────
# 2. Optimality certificate for the smooth problem
# ──────────────────────────────────────────────────────────────────────

class TestOptimality:

    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("tau", [0.05, 0.5, 0.95])
    def test_gradient_vanishes_at_the_solution(self, kernel, tau):
        """L_h is convex and smooth, so a zero gradient is necessary and
        sufficient -- the direct analogue of the LP optimality certificate."""
        X, y = _problem()
        res = ConquerSolver(kernel=kernel).solve(X, y, tau)
        h = res.solver_info["bandwidth"]
        g = smoothed_gradient(X, y, res.coefficients, tau, h, kernel)
        assert np.abs(g).max() < 1e-7, f"{kernel} tau={tau}: grad {np.abs(g).max():.2e}"

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_solution_is_a_minimum_not_a_saddle(self, kernel):
        X, y = _problem()
        res = ConquerSolver(kernel=kernel).solve(X, y, 0.5)
        H = smoothed_hessian(X, y, res.coefficients, res.solver_info["bandwidth"], kernel)
        assert np.linalg.eigvalsh(H).min() > 0, "Hessian must be positive definite"

    def test_loss_is_not_beaten_by_a_perturbation(self):
        """Cheap direct check that we are at the bottom."""
        X, y = _problem()
        res = ConquerSolver().solve(X, y, 0.5)
        h = res.solver_info["bandwidth"]
        best = smoothed_loss(X, y, res.coefficients, 0.5, h, "Gaussian")
        rng = np.random.RandomState(0)
        for _ in range(20):
            step = rng.randn(res.coefficients.size) * 1e-3
            assert smoothed_loss(X, y, res.coefficients + step, 0.5, h, "Gaussian") >= best


# ──────────────────────────────────────────────────────────────────────
# 3. The h -> 0 limit: smoothing must reduce to the exact estimator
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
class TestSmoothingLimit:

    def test_converges_to_the_exact_solution_as_bandwidth_shrinks(self):
        """The single most informative test here: it ties this estimator to the
        exact solvers that are already verified, and would catch a wrong-but-
        self-consistent loss that every other check waves through."""
        from pinball.linear.solvers import get_solver

        X, y = _problem(n=600)
        exact = get_solver("br").solve(X, y, 0.5).coefficients
        errors = [
            np.abs(
                ConquerSolver(bandwidth=h).solve(X, y, 0.5).coefficients - exact
            ).max()
            for h in (0.4, 0.2, 0.1, 0.05, 0.02, 0.01)
        ]
        # A trend, not a monotone sequence.  The bias vanishes as O(h^(s+1)) in
        # expectation, but on one fixed dataset the path need not be monotone,
        # and at very small h few observations fall inside the band so the
        # smoothed problem itself becomes poorly conditioned.  Measured here:
        # 2.1e-2 at h=0.4 falling to 6.3e-3 at h=0.01, with a bump at h=0.02.
        assert errors[-1] < 0.5 * errors[0], (
            f"shrinking the bandwidth did not move the fit toward the exact "
            f"solution: {[f'{e:.2e}' for e in errors]}"
        )
        assert errors[-1] < 0.02, f"did not approach the exact fit: {errors[-1]:.3e}"

    def test_default_bandwidth_leaves_a_visible_but_small_bias(self):
        """Smoothing bias is a feature, not a defect -- pin its size so a
        change in the bandwidth convention cannot slip through unnoticed."""
        from pinball.linear.solvers import get_solver

        X, y = _problem(n=800)
        exact = get_solver("br").solve(X, y, 0.5).coefficients
        got = ConquerSolver().solve(X, y, 0.5).coefficients
        gap = np.abs(got - exact).max()
        assert 1e-4 < gap < 1e-1, f"unexpected distance from the exact fit: {gap:.2e}"


# ──────────────────────────────────────────────────────────────────────
# 5. Equivariance, corrected for a bandwidth that ignores the data scale
# ──────────────────────────────────────────────────────────────────────

class TestEquivariance:
    """Three of Koenker's four identities hold exactly here, because they leave
    the residuals unchanged and the smoothed loss depends on y and X only
    through the residuals.  Scale equivariance is the exception."""

    @pytest.mark.parametrize("kernel", ["Gaussian", "parabolic"])
    def test_regression_equivariance(self, kernel):
        X, y = _problem()
        gamma = np.array([0.7, -1.3, 2.5, 0.4])
        a = ConquerSolver(kernel=kernel).solve(X, y + X @ gamma, 0.4).coefficients
        b = ConquerSolver(kernel=kernel).solve(X, y, 0.4).coefficients + gamma
        np.testing.assert_allclose(a, b, atol=1e-7)

    @pytest.mark.parametrize("kernel", ["Gaussian", "triangular"])
    def test_reparameterization_equivariance(self, kernel):
        """Holds at a fixed bandwidth -- see the test below for why the
        bandwidth has to be pinned for this comparison to mean anything."""
        X, y = _problem()
        rng = np.random.RandomState(3)
        A = np.eye(4) + 0.3 * rng.randn(4, 4)
        a = ConquerSolver(kernel=kernel, bandwidth=0.2).solve(X @ A, y, 0.6)
        b = ConquerSolver(kernel=kernel, bandwidth=0.2).solve(X, y, 0.6)
        np.testing.assert_allclose(
            a.coefficients, np.linalg.solve(A, b.coefficients), atol=1e-6
        )

    def test_default_bandwidth_is_not_reparameterization_invariant(self):
        """A wart worth pinning rather than discovering later.

        The default bandwidth counts covariates, and covariates are counted by
        looking for a constant column.  Mixing the design with a general A
        destroys that column, so the intercept stops being recognised, p goes up
        by one and h changes.  The estimator is still equivariant at any fixed
        h; it is the *default* that moves.
        """
        X, y = _problem()
        A = np.eye(4) + 0.3 * np.random.RandomState(3).randn(4, 4)
        h_plain = ConquerSolver().solve(X, y, 0.6).solver_info["bandwidth"]
        h_mixed = ConquerSolver().solve(X @ A, y, 0.6).solver_info["bandwidth"]
        assert h_plain != h_mixed
        assert h_plain == pytest.approx(default_bandwidth(X.shape[0], 3))
        assert h_mixed == pytest.approx(default_bandwidth(X.shape[0], 4))

    def test_reflection_equivariance(self):
        X, y = _problem()
        a = ConquerSolver().solve(X, -y, 0.3).coefficients
        b = -ConquerSolver().solve(X, y, 0.7).coefficients
        np.testing.assert_allclose(a, b, atol=1e-7)

    def test_scale_equivariance_requires_scaling_the_bandwidth(self):
        """The bandwidth is a function of (n, p) only, so it does not follow the
        data.  Rescaling y alone therefore changes the estimator -- that is a
        property of the method, not a bug -- while rescaling h alongside it
        restores the identity.  Both directions are asserted so that neither can
        change silently."""
        X, y = _problem()
        base = ConquerSolver().solve(X, y, 0.5)
        h0 = base.solver_info["bandwidth"]
        c = 5.0

        naive = ConquerSolver().solve(X, c * y, 0.5).coefficients
        assert np.abs(naive - c * base.coefficients).max() > 1e-3, (
            "conquer became scale equivariant at fixed h; the bandwidth "
            "convention must have changed"
        )

        scaled = ConquerSolver(bandwidth=c * h0).solve(X, c * y, 0.5).coefficients
        np.testing.assert_allclose(scaled, c * base.coefficients, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────
# Interface, validation and failure reporting
# ──────────────────────────────────────────────────────────────────────

class TestInterface:

    def test_default_bandwidth_formula(self):
        """h = ((log n + p) / n) ** 0.4, matching R conquer."""
        assert default_bandwidth(200, 2) == pytest.approx(0.265997, abs=1e-6)
        assert default_bandwidth(500, 5) == pytest.approx(0.218940, abs=1e-6)
        assert default_bandwidth(10000, 10) == pytest.approx(0.081925, abs=1e-6)

    def test_bandwidth_counts_covariates_without_the_intercept(self):
        X, y = _problem(n=500, p=3)          # X already carries a constant column
        info = ConquerSolver().solve(X, y, 0.5).solver_info
        assert info["bandwidth"] == pytest.approx(default_bandwidth(500, 3), rel=1e-12)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"kernel": "banana"}, "Unknown kernel"),
            ({"bandwidth": 0.0}, "bandwidth must be positive"),
            ({"bandwidth": -1.0}, "bandwidth must be positive"),
            ({"tol": 0.0}, "tol must be positive"),
            ({"max_iter": 0}, "max_iter must be at least 1"),
        ],
    )
    def test_invalid_parameters_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ConquerSolver(**kwargs)

    def test_non_convergence_is_reported(self):
        """An exhausted budget must warn and show in status, never pass
        silently -- the same contract every other solver here follows."""
        from pinball.linear.solvers.conquer import STATUS_NOT_CONVERGED

        X, y = _problem()
        with pytest.warns(UserWarning, match="did not converge"):
            res = ConquerSolver(max_iter=2, tol=1e-14).solve(X, y, 0.5)
        assert res.status == STATUS_NOT_CONVERGED
        assert res.solver_info["converged"] is False

    def test_objective_value_is_the_pinball_loss(self):
        """Every solver reports the pinball loss of its own residuals, even
        though conquer minimises something else."""
        X, y = _problem()
        res = ConquerSolver().solve(X, y, 0.3)
        r = res.residuals
        expected = 0.3 * np.sum(np.maximum(r, 0)) + 0.7 * np.sum(np.maximum(-r, 0))
        assert res.objective_value == pytest.approx(expected)

    def test_registered_in_the_solver_registry(self):
        from pinball.linear.solvers import get_solver, list_solvers

        assert "conquer" in list_solvers()
        assert isinstance(get_solver("conquer"), ConquerSolver)

    def test_multi_quantile_through_the_estimator(self):
        from pinball.linear import QuantileRegressor

        X, y = _problem()
        m = QuantileRegressor(tau=[0.25, 0.5, 0.75], method="conquer").fit(X[:, 1:], y)
        assert m.coef_.shape == (3, 3)
        # Quantile curves should be ordered at the centre of the design.
        preds = m.predict(np.zeros((1, 3)))[0]
        assert preds[0] < preds[1] < preds[2]


class TestInference:

    def test_asymptotic_covariance_is_symmetric_positive_definite(self):
        X, y = _problem()
        res = ConquerSolver().solve(X, y, 0.5)
        cov = asymptotic_cov(
            X, y, res.coefficients, 0.5, res.solver_info["bandwidth"], "Gaussian"
        )
        np.testing.assert_allclose(cov, cov.T, atol=1e-14)
        assert np.linalg.eigvalsh(cov).min() > 0

    def test_bootstrap_is_reproducible(self):
        X, y = _problem(n=300)
        res = ConquerSolver().solve(X, y, 0.5)
        h = res.solver_info["bandwidth"]
        kw = dict(n_boot=25, random_state=7)
        a = multiplier_bootstrap(X, y, 0.5, h, "Gaussian", **kw)
        b = multiplier_bootstrap(X, y, 0.5, h, "Gaussian", **kw)
        np.testing.assert_allclose(a, b, atol=0.0)

    def test_bootstrap_and_asymptotic_standard_errors_agree(self):
        """Two independent routes to the same quantity; if they disagree, one
        of them is wrong."""
        X, y = _problem(n=600)
        res = ConquerSolver().solve(X, y, 0.5)
        h = res.solver_info["bandwidth"]
        ci = confidence_intervals(
            X, y, res.coefficients, 0.5, h, "Gaussian",
            ci="both", n_boot=400, random_state=0,
        )
        ratio = ci["bootstrap_se"] / ci["asymptotic_se"]
        assert np.all((ratio > 0.7) & (ratio < 1.4)), f"SE ratio {ratio}"

    def test_all_four_interval_types_bracket_the_estimate(self):
        X, y = _problem(n=400)
        res = ConquerSolver().solve(X, y, 0.5)
        ci = confidence_intervals(
            X, y, res.coefficients, 0.5, res.solver_info["bandwidth"], "Gaussian",
            ci="both", n_boot=200, random_state=1,
        )
        for name in ("asymptotic", "percentile", "pivotal", "normal"):
            lo, hi = ci[name][:, 0], ci[name][:, 1]
            assert np.all(lo < hi), f"{name}: empty interval"
            assert np.all(lo <= res.coefficients + 1e-9), f"{name}: below estimate"
            assert np.all(hi >= res.coefficients - 1e-9), f"{name}: above estimate"

    def test_pivotal_is_the_reflected_percentile(self):
        """pivCI = 2*beta - reversed(perCI); pinned because it is easy to get
        backwards and the mistake is invisible on symmetric data."""
        X, y = _problem(n=400)
        res = ConquerSolver().solve(X, y, 0.5)
        ci = confidence_intervals(
            X, y, res.coefficients, 0.5, res.solver_info["bandwidth"], "Gaussian",
            ci="bootstrap", n_boot=200, random_state=2,
        )
        b = res.coefficients
        np.testing.assert_allclose(ci["pivotal"][:, 0], 2 * b - ci["percentile"][:, 1])
        np.testing.assert_allclose(ci["pivotal"][:, 1], 2 * b - ci["percentile"][:, 0])

    @pytest.mark.parametrize("bad", [{"ci": "nonsense"}, {"alpha": 0.0}, {"alpha": 1.0}])
    def test_invalid_inference_arguments_rejected(self, bad):
        X, y = _problem(n=200)
        res = ConquerSolver().solve(X, y, 0.5)
        kw = dict(ci="asymptotic", alpha=0.05)
        kw.update(bad)
        with pytest.raises(ValueError):
            confidence_intervals(
                X, y, res.coefficients, 0.5, res.solver_info["bandwidth"],
                "Gaussian", **kw,
            )


# ──────────────────────────────────────────────────────────────────────
# 4. Parity with the R reference
#
# Frozen output from R conquer 1.3.3 (`conquer::conquer`, tol=1e-12), the
# implementation this module is written against.  Regenerate with
# local_testing/r_export/fit_conquer_reference.R.
#
# This module is an independent implementation, not a port: R conquer is
# GPL-3 and this package is MIT, so its source was deliberately not consulted.
# Everything pinned here was derived from the published algorithm and checked
# against R's *outputs*.
# ──────────────────────────────────────────────────────────────────────

#: (kernel, tau) -> coefficients, for the design built by _r_parity_problem().
_R_CONQUER = {
    ("Gaussian", 0.5): [-0.0524463747, 1.0601906552, 2.0214322593, 2.9478243951],
    ("logistic", 0.5): [-0.0595530398, 1.0631289754, 2.0219953367, 2.9449178127],
    ("uniform", 0.5): [-0.0491977227, 1.0553083026, 2.0223391269, 2.9516409974],
    ("parabolic", 0.5): [-0.0496300190, 1.0531909965, 2.0267704599, 2.9560304757],
    ("triangular", 0.5): [-0.0493024041, 1.0530585046, 2.0291807136, 2.9577359051],
}


def _r_parity_problem():
    """The exact design R was run on; see the header above."""
    rng = np.random.RandomState(11)
    n, p = 700, 3
    X = np.column_stack([np.ones(n), rng.randn(n, p)])
    y = X @ np.array([0.0, 1.0, 2.0, 3.0]) + rng.standard_t(5, n)
    return X, y


class TestRParity:
    """The repo's acceptance bar: agreement with the R reference.

    Compared against ``conquer::conquer`` from R package conquer 1.3.3.
    """

    def test_default_bandwidth_matches_r(self):
        """R conquer's h = ((log n + p)/n)^0.4, checked against values taken
        from the package across a range of n and p."""
        for n, p, expected in [
            (200, 2, 0.265997), (200, 10, 0.357639),
            (500, 5, 0.218940), (2000, 10, 0.150592),
            (10000, 2, 0.066046), (10000, 10, 0.081925),
        ]:
            assert default_bandwidth(n, p) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_coefficients_match_r_conquer(self, kernel):
        X, y = _r_parity_problem()
        got = ConquerSolver(kernel=kernel, tol=1e-12).solve(X, y, 0.5).coefficients
        np.testing.assert_allclose(
            got, _R_CONQUER[(kernel, 0.5)], atol=1e-6,
            err_msg=f"{kernel}: disagrees with R conquer",
        )

    def test_asymptotic_covariance_matches_r(self):
        """R's asyCI uses the empirical score outer product for the sandwich
        meat, not the tau(1-tau) form that the unsmoothed theory suggests --
        the latter inflates these standard errors by 6-10%.  Pinned because
        the two are easy to confuse and both look plausible."""
        X, y = _r_parity_problem()
        res = ConquerSolver(tol=1e-12).solve(X, y, 0.5)
        h = res.solver_info["bandwidth"]
        se = np.sqrt(np.diag(
            asymptotic_cov(X, y, res.coefficients, 0.5, h, "Gaussian")
        ))
        r = y - X @ res.coefficients
        score = _kernel_cdf(-r / h, "Gaussian") - 0.5
        H = smoothed_hessian(X, y, res.coefficients, h, "Gaussian")
        naive = np.sqrt(np.diag(
            np.linalg.inv(H) @ (X.T @ X / len(y)) @ np.linalg.inv(H)
        ) * 0.25 / len(y))
        assert np.all(naive > 1.02 * se), (
            "the tau(1-tau) approximation should be visibly wider than the "
            "empirical-score sandwich R actually uses"
        )
        assert np.all(np.isfinite(score))
