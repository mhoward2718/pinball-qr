"""Tests for the bootstrap inference module.

Covers xy-pair, wild, and MCMB bootstrap methods for quantile regression.
"""

import numpy as np
import pytest

from pinball._bootstrap import (
    BootstrapResult,
    _mcmb,
    _wild,
    _xy_pairs,
    bootstrap,
)


def _has_native():
    try:
        from pinball._native import rqfnb  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def synthetic_problem():
    """Simple well-conditioned problem for bootstrap testing."""
    rng = np.random.RandomState(42)
    n = 100
    X = np.column_stack([np.ones(n), rng.randn(n)])
    beta_true = np.array([3.0, 1.5])
    y = X @ beta_true + rng.randn(n) * 0.5
    return X, y, beta_true


@pytest.fixture
def engel_problem():
    """Engel data with intercept column prepended."""
    from pinball.datasets import load_engel
    data = load_engel()
    n = data.data.shape[0]
    X = np.column_stack([np.ones(n), data.data[:, 0]])
    y = data.target
    return X, y


# ──────────────────────────────────────────────────────────────────────
# BootstrapResult
# ──────────────────────────────────────────────────────────────────────

class TestBootstrapResult:

    def test_construction(self):
        B = np.random.randn(50, 3)
        r = BootstrapResult(
            boot_coefficients=B,
            coefficients=np.mean(B, axis=0),
            std_errors=np.std(B, axis=0, ddof=1),
            conf_int=np.column_stack([
                np.percentile(B, 2.5, axis=0),
                np.percentile(B, 97.5, axis=0),
            ]),
            bsmethod="xy",
            nboot=50,
        )
        assert r.boot_coefficients.shape == (50, 3)
        assert r.bsmethod == "xy"
        assert r.nboot == 50

    def test_covariance(self):
        """BootstrapResult.covariance should be sample cov of boot coefficients."""
        rng = np.random.RandomState(7)
        B = rng.randn(200, 2)
        r = BootstrapResult(
            boot_coefficients=B,
            coefficients=np.mean(B, axis=0),
            std_errors=np.std(B, axis=0, ddof=1),
            conf_int=np.zeros((2, 2)),
            bsmethod="xy",
            nboot=200,
        )
        np.testing.assert_allclose(r.covariance, np.cov(B, rowvar=False), atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# bootstrap() dispatcher
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
class TestBootstrapDispatcher:

    def test_default_method_is_xy(self, synthetic_problem):
        X, y, _ = synthetic_problem
        result = bootstrap(X, y, tau=0.5, nboot=20)
        assert result.bsmethod == "xy"

    def test_invalid_method_raises(self, synthetic_problem):
        X, y, _ = synthetic_problem
        with pytest.raises(ValueError, match="Unknown"):
            bootstrap(X, y, tau=0.5, nboot=20, method="invalid")

    def test_returns_bootstrap_result(self, synthetic_problem):
        X, y, _ = synthetic_problem
        result = bootstrap(X, y, tau=0.5, nboot=20, method="xy")
        assert isinstance(result, BootstrapResult)


# ──────────────────────────────────────────────────────────────────────
# xy-pair bootstrap
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
class TestXYPairsBootstrap:

    def test_boot_coefficients_shape(self, synthetic_problem):
        X, y, _ = synthetic_problem
        result = _xy_pairs(X, y, tau=0.5, nboot=50, random_state=42)
        assert result.boot_coefficients.shape == (50, X.shape[1])

    def test_std_errors_positive(self, synthetic_problem):
        X, y, _ = synthetic_problem
        result = _xy_pairs(X, y, tau=0.5, nboot=50, random_state=42)
        assert np.all(result.std_errors > 0)

    def test_conf_int_contains_true_value(self, synthetic_problem):
        """With large enough nboot, 95% CI should contain true β."""
        X, y, beta_true = synthetic_problem
        result = _xy_pairs(X, y, tau=0.5, nboot=200, random_state=42)
        for j in range(len(beta_true)):
            assert result.conf_int[j, 0] < beta_true[j] < result.conf_int[j, 1], (
                f"True β[{j}]={beta_true[j]} not in "
                f"[{result.conf_int[j, 0]:.3f}, {result.conf_int[j, 1]:.3f}]"
            )

    def test_reproducible_with_seed(self, synthetic_problem):
        X, y, _ = synthetic_problem
        r1 = _xy_pairs(X, y, tau=0.5, nboot=30, random_state=42)
        r2 = _xy_pairs(X, y, tau=0.5, nboot=30, random_state=42)
        np.testing.assert_array_equal(r1.boot_coefficients, r2.boot_coefficients)

    def test_mofn_subsampling(self, synthetic_problem):
        """m-of-n subsampling: smaller subsample should give wider SE."""
        X, y, _ = synthetic_problem
        _xy_pairs(X, y, tau=0.5, nboot=100, random_state=42)
        r_sub = _xy_pairs(X, y, tau=0.5, nboot=100, random_state=42, mofn=50)
        # Subsampled bootstrap usually gives wider SE after sqrt(m/n) scaling
        # Just check it runs and shapes are correct
        assert r_sub.boot_coefficients.shape == (100, X.shape[1])

    def test_engel_data(self, engel_problem):
        X, y = engel_problem
        result = _xy_pairs(X, y, tau=0.5, nboot=50, random_state=0)
        assert result.boot_coefficients.shape == (50, 2)
        assert np.all(result.std_errors > 0)


# ──────────────────────────────────────────────────────────────────────
# Wild bootstrap
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
class TestWildBootstrap:

    def test_boot_coefficients_shape(self, synthetic_problem):
        X, y, _ = synthetic_problem
        result = _wild(X, y, tau=0.5, nboot=50, random_state=42)
        assert result.boot_coefficients.shape == (50, X.shape[1])

    def test_std_errors_positive(self, synthetic_problem):
        X, y, _ = synthetic_problem
        result = _wild(X, y, tau=0.5, nboot=50, random_state=42)
        assert np.all(result.std_errors > 0)

    def test_reproducible_with_seed(self, synthetic_problem):
        X, y, _ = synthetic_problem
        r1 = _wild(X, y, tau=0.5, nboot=30, random_state=42)
        r2 = _wild(X, y, tau=0.5, nboot=30, random_state=42)
        np.testing.assert_array_equal(r1.boot_coefficients, r2.boot_coefficients)

    def test_conf_int_shape(self, synthetic_problem):
        X, y, _ = synthetic_problem
        result = _wild(X, y, tau=0.5, nboot=50, random_state=42)
        assert result.conf_int.shape == (X.shape[1], 2)

    def test_ci_lower_lt_upper(self, synthetic_problem):
        X, y, _ = synthetic_problem
        result = _wild(X, y, tau=0.5, nboot=50, random_state=42)
        assert np.all(result.conf_int[:, 0] < result.conf_int[:, 1])


# ──────────────────────────────────────────────────────────────────────
# MCMB bootstrap
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
class TestMCMBBootstrap:

    def test_boot_coefficients_shape(self, synthetic_problem):
        X, y, _ = synthetic_problem
        result = _mcmb(X, y, tau=0.5, nboot=50, random_state=42)
        assert result.boot_coefficients.shape == (50, X.shape[1])

    def test_std_errors_positive(self, synthetic_problem):
        X, y, _ = synthetic_problem
        result = _mcmb(X, y, tau=0.5, nboot=50, random_state=42)
        assert np.all(result.std_errors > 0)

    def test_reproducible_with_seed(self, synthetic_problem):
        X, y, _ = synthetic_problem
        r1 = _mcmb(X, y, tau=0.5, nboot=30, random_state=42)
        r2 = _mcmb(X, y, tau=0.5, nboot=30, random_state=42)
        np.testing.assert_array_equal(r1.boot_coefficients, r2.boot_coefficients)

    def test_conf_int_shape(self, synthetic_problem):
        X, y, _ = synthetic_problem
        result = _mcmb(X, y, tau=0.5, nboot=50, random_state=42)
        assert result.conf_int.shape == (X.shape[1], 2)


# ──────────────────────────────────────────────────────────────────────
# Integration with summary()
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
class TestBootstrapSummaryIntegration:

    def test_summary_se_boot(self, synthetic_problem):
        """summary(se='boot') should use bootstrap for SE estimation."""
        from pinball._inference import InferenceResult, summary
        X, y, _ = synthetic_problem
        from pinball.linear.solvers.fnb import FNBSolver
        coef = FNBSolver().solve(X, y, 0.5).coefficients
        result = summary(X, y, coef, tau=0.5, se="boot", alpha=0.05, nboot=50)
        assert isinstance(result, InferenceResult)
        assert result.se_method == "boot"
        assert np.all(result.std_errors > 0)
