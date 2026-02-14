"""Tests for the inference module (standard errors, summary tables)."""

import numpy as np
import pytest

from pinball._inference import InferenceResult, summary


def _has_native():
    try:
        import pinball._native  # noqa: F401
        return True
    except ImportError:
        return False


class TestInferenceResult:

    def test_repr(self):
        r = InferenceResult(
            coefficients=np.array([1.0, 2.0]),
            std_errors=np.array([0.1, 0.2]),
            t_statistics=np.array([10.0, 10.0]),
            p_values=np.array([0.0, 0.0]),
            conf_int=np.array([[0.8, 1.2], [1.6, 2.4]]),
            se_method="iid",
        )
        text = repr(r)
        assert "iid" in text
        assert "Coef" in text


class TestSummary:

    @pytest.fixture
    def problem(self):
        rng = np.random.RandomState(42)
        n, _p = 200, 3
        X = np.column_stack([np.ones(n), rng.randn(n, 2)])
        beta_true = np.array([5.0, 2.0, -1.0])
        y = X @ beta_true + rng.randn(n) * 0.5
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return X, y, coef

    def test_iid_returns_inference_result(self, problem):
        X, y, coef = problem
        result = summary(X, y, coef, tau=0.5, se="iid")
        assert isinstance(result, InferenceResult)
        assert result.se_method == "iid"

    def test_nid_returns_inference_result(self, problem):
        X, y, coef = problem
        result = summary(X, y, coef, tau=0.5, se="nid")
        assert isinstance(result, InferenceResult)
        assert result.se_method == "nid"

    @pytest.mark.skipif(not _has_native(), reason="Fortran extension not built")
    def test_auto_selects_rank_for_small_n(self, problem):
        X, y, coef = problem
        # n=200 < 1001 → auto should select "rank" (falls back to nid for now)
        result = summary(X, y, coef, tau=0.5, se="auto")
        # Currently rank falls through to nid (stub), but check it runs
        assert isinstance(result, InferenceResult)

    def test_coefficient_values_preserved(self, problem):
        X, y, coef = problem
        result = summary(X, y, coef, tau=0.5, se="iid")
        np.testing.assert_array_equal(result.coefficients, coef)

    def test_se_positive(self, problem):
        X, y, coef = problem
        result = summary(X, y, coef, tau=0.5, se="iid")
        assert np.all(result.std_errors > 0)

    def test_conf_int_shape(self, problem):
        X, y, coef = problem
        result = summary(X, y, coef, tau=0.5, se="iid")
        assert result.conf_int.shape == (3, 2)

    def test_conf_int_contains_estimate(self, problem):
        X, y, coef = problem
        result = summary(X, y, coef, tau=0.5, se="iid", alpha=0.05)
        # Each coefficient should be within its CI
        for j in range(len(coef)):
            assert result.conf_int[j, 0] <= coef[j] <= result.conf_int[j, 1]

    def test_feature_names(self, problem):
        X, y, coef = problem
        names = ["const", "x1", "x2"]
        result = summary(X, y, coef, tau=0.5, se="iid", feature_names=names)
        assert result.feature_names == names

    def test_p_values_in_01(self, problem):
        X, y, coef = problem
        result = summary(X, y, coef, tau=0.5, se="iid")
        assert np.all(result.p_values >= 0)
        assert np.all(result.p_values <= 1)


class TestLambdaSelection:

    def test_positive(self):
        from pinball.util.lambda_selection import lambda_hat_bcv
        X = np.random.randn(100, 5)
        lam = lambda_hat_bcv(X, tau=0.5)
        assert lam > 0

    def test_decreases_with_n(self):
        from pinball.util.lambda_selection import lambda_hat_bcv
        rng = np.random.RandomState(42)
        X100 = rng.randn(100, 5)
        X1000 = rng.randn(1000, 5)
        assert lambda_hat_bcv(X100, 0.5) > lambda_hat_bcv(X1000, 0.5)

    def test_increases_with_c(self):
        from pinball.util.lambda_selection import lambda_hat_bcv
        X = np.random.randn(100, 5)
        assert lambda_hat_bcv(X, 0.5, c=2) > lambda_hat_bcv(X, 0.5, c=1)
