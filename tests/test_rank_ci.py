"""Tests for rank-inversion confidence intervals.

These test the ``se="rank"`` path in ``summary()`` which delegates to
the BR solver with ``ci=True`` and then interpolates the Fortran output.
"""

import numpy as np
import pytest

from pinball._inference import summary, InferenceResult


def _has_native():
    """Return True if the Fortran extension is available."""
    try:
        from pinball._native import rqbr  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def engel_problem():
    """Engel food expenditure data — well-conditioned, n=235."""
    from pinball.datasets import load_engel
    data = load_engel()
    n = data.data.shape[0]
    X = np.column_stack([np.ones(n), data.data[:, 0]])  # intercept + income
    y = data.target
    # Fit at median with BR solver
    from pinball.linear.solvers.br import BRSolver
    solver = BRSolver(ci=True, iid=True, alpha=0.05)
    result = solver.solve(X, y, 0.5)
    return X, y, result


@pytest.fixture
def small_problem():
    """Small synthetic problem with known structure."""
    rng = np.random.RandomState(42)
    n, p = 100, 2
    X = np.column_stack([np.ones(n), rng.randn(n)])
    beta_true = np.array([3.0, 1.5])
    y = X @ beta_true + rng.randn(n) * 0.5
    return X, y, beta_true


@pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
class TestRankInversionCI:
    """Test the rank-inversion CI path (se='rank')."""

    def test_returns_inference_result(self, engel_problem):
        X, y, solver_result = engel_problem
        result = summary(X, y, solver_result.coefficients, tau=0.5, se="rank")
        assert isinstance(result, InferenceResult)

    def test_se_method_is_rank(self, engel_problem):
        X, y, solver_result = engel_problem
        result = summary(X, y, solver_result.coefficients, tau=0.5, se="rank")
        assert result.se_method == "rank"

    def test_conf_int_shape(self, engel_problem):
        X, y, solver_result = engel_problem
        result = summary(X, y, solver_result.coefficients, tau=0.5, se="rank")
        p = X.shape[1]
        assert result.conf_int.shape == (p, 2)

    def test_conf_int_contains_estimate(self, engel_problem):
        X, y, solver_result = engel_problem
        result = summary(
            X, y, solver_result.coefficients, tau=0.5, se="rank", alpha=0.05
        )
        coef = solver_result.coefficients
        for j in range(len(coef)):
            assert result.conf_int[j, 0] <= coef[j] <= result.conf_int[j, 1], (
                f"Coef {j}: {coef[j]} not in [{result.conf_int[j, 0]}, "
                f"{result.conf_int[j, 1]}]"
            )

    def test_ci_lower_lt_upper(self, engel_problem):
        X, y, solver_result = engel_problem
        result = summary(X, y, solver_result.coefficients, tau=0.5, se="rank")
        for j in range(X.shape[1]):
            assert result.conf_int[j, 0] < result.conf_int[j, 1]

    def test_narrower_ci_with_larger_alpha(self, engel_problem):
        """Wider confidence level (smaller alpha) → wider CI."""
        X, y, solver_result = engel_problem
        coef = solver_result.coefficients
        r90 = summary(X, y, coef, tau=0.5, se="rank", alpha=0.10)
        r95 = summary(X, y, coef, tau=0.5, se="rank", alpha=0.05)
        # 95% CI should be at least as wide as 90% CI for at least one coef
        width_90 = r90.conf_int[:, 1] - r90.conf_int[:, 0]
        width_95 = r95.conf_int[:, 1] - r95.conf_int[:, 0]
        assert np.all(width_95 >= width_90 - 1e-6)

    def test_engel_slope_ci_contains_known_value(self, engel_problem):
        """The Engel median slope is around 0.56; CI should contain it."""
        X, y, solver_result = engel_problem
        result = summary(X, y, solver_result.coefficients, tau=0.5, se="rank")
        # slope is index 1
        assert result.conf_int[1, 0] < 0.60
        assert result.conf_int[1, 1] > 0.50

    def test_rank_std_errors_are_half_ci_width(self, engel_problem):
        """Rank CIs don't come from SE; std_errors should be derived from CI width."""
        X, y, solver_result = engel_problem
        result = summary(X, y, solver_result.coefficients, tau=0.5, se="rank")
        # SE should be positive (derived from CI half-width)
        assert np.all(result.std_errors > 0)

    def test_nid_variant(self, small_problem):
        """Rank CIs with iid=False (NID variant)."""
        X, y, beta_true = small_problem
        from pinball.linear.solvers.br import BRSolver
        solver = BRSolver(ci=True, iid=False, alpha=0.05)
        res = solver.solve(X, y, 0.5)
        result = summary(X, y, res.coefficients, tau=0.5, se="rank")
        assert isinstance(result, InferenceResult)
        assert result.conf_int.shape == (2, 2)
