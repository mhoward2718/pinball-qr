"""Tests for the Powell kernel sandwich standard errors (se='ker')."""

import numpy as np
import pytest

from pinball._inference import InferenceResult, summary


def _has_native():
    try:
        from pinball._native import rqbr  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def engel_fitted():
    """Fit Engel data at median with FNB solver, return (X_aug, y, coef)."""
    from pinball.datasets import load_engel
    data = load_engel()
    n = data.data.shape[0]
    X = np.column_stack([np.ones(n), data.data[:, 0]])
    y = data.target
    from pinball.linear.solvers.fnb import FNBSolver
    solver = FNBSolver()
    result = solver.solve(X, y, 0.5)
    return X, y, result.coefficients


@pytest.fixture
def synthetic_fitted():
    """Synthetic well-conditioned problem."""
    rng = np.random.RandomState(99)
    n = 300
    X = np.column_stack([np.ones(n), rng.randn(n, 2)])
    beta_true = np.array([2.0, 1.0, -0.5])
    y = X @ beta_true + rng.randn(n) * 0.3
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return X, y, coef


@pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
class TestKernelSE:
    """Test the kernel (Powell) sandwich SE method."""

    def test_returns_inference_result(self, engel_fitted):
        X, y, coef = engel_fitted
        result = summary(X, y, coef, tau=0.5, se="ker")
        assert isinstance(result, InferenceResult)

    def test_se_method_is_ker(self, engel_fitted):
        X, y, coef = engel_fitted
        result = summary(X, y, coef, tau=0.5, se="ker")
        assert result.se_method == "ker"

    def test_se_positive(self, engel_fitted):
        X, y, coef = engel_fitted
        result = summary(X, y, coef, tau=0.5, se="ker")
        assert np.all(result.std_errors > 0)

    def test_conf_int_shape(self, engel_fitted):
        X, y, coef = engel_fitted
        result = summary(X, y, coef, tau=0.5, se="ker")
        assert result.conf_int.shape == (X.shape[1], 2)

    def test_conf_int_contains_estimate(self, engel_fitted):
        X, y, coef = engel_fitted
        result = summary(X, y, coef, tau=0.5, se="ker", alpha=0.05)
        for j in range(len(coef)):
            assert result.conf_int[j, 0] <= coef[j] <= result.conf_int[j, 1]

    def test_t_statistics_nonzero(self, engel_fitted):
        X, y, coef = engel_fitted
        result = summary(X, y, coef, tau=0.5, se="ker")
        assert np.all(np.abs(result.t_statistics) > 0)

    def test_p_values_in_01(self, engel_fitted):
        X, y, coef = engel_fitted
        result = summary(X, y, coef, tau=0.5, se="ker")
        assert np.all(result.p_values >= 0)
        assert np.all(result.p_values <= 1)


class TestKernelSESynthetic:
    """Test kernel SE on synthetic data (no Fortran needed for coef)."""

    def test_returns_inference_result(self, synthetic_fitted):
        X, y, coef = synthetic_fitted
        result = summary(X, y, coef, tau=0.5, se="ker")
        assert isinstance(result, InferenceResult)

    def test_se_same_order_as_iid(self, synthetic_fitted):
        """Kernel SE should be in the same ballpark as IID SE."""
        X, y, coef = synthetic_fitted
        r_ker = summary(X, y, coef, tau=0.5, se="ker")
        r_iid = summary(X, y, coef, tau=0.5, se="iid")
        ratio = r_ker.std_errors / r_iid.std_errors
        # Ratio should be within 0.1x to 10x — same order of magnitude
        assert np.all(ratio > 0.1) and np.all(ratio < 10)

    def test_different_tau(self, synthetic_fitted):
        """Kernel SE works for non-median quantiles."""
        X, y, coef = synthetic_fitted
        for tau in [0.1, 0.25, 0.75, 0.9]:
            result = summary(X, y, coef, tau=tau, se="ker")
            assert np.all(result.std_errors > 0)
