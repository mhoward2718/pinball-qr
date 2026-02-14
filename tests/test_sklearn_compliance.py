"""Tests for sklearn check_estimator compliance.

Written TDD-style *before* fixing the estimator.

These tests cover the four specific failures found by
``sklearn.utils.estimator_checks.check_estimator``:

1. Mixin order: ``RegressorMixin`` before ``BaseEstimator``
2. ``validate_data`` instead of ``check_X_y`` / ``check_array``
3. Informative error when ``n_samples < n_features``
4. Informative error when ``n_samples == 1``
"""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from pinball import QuantileRegressor


def _has_native():
    try:
        from pinball._native import rqfnb  # noqa: F401
        return True
    except ImportError:
        return False


_native_required = pytest.mark.skipif(
    not _has_native(), reason="Fortran extension not built"
)


# ── 1. Mixin ordering ───────────────────────────────────────────────

class TestMixinOrder:
    """RegressorMixin must appear before BaseEstimator in MRO."""

    def test_regressor_mixin_before_base_estimator(self):
        from sklearn.base import BaseEstimator, RegressorMixin
        mro = QuantileRegressor.__mro__
        idx_reg = mro.index(RegressorMixin)
        idx_base = mro.index(BaseEstimator)
        assert idx_reg < idx_base, (
            f"RegressorMixin at {idx_reg} should come before "
            f"BaseEstimator at {idx_base} in MRO"
        )


# ── 2. n_features_in_ consistency via validate_data ─────────────────

@_native_required
class TestNFeaturesConsistency:
    """predict() must raise when input has wrong number of features."""

    def test_predict_wrong_n_features_raises(self):
        rng = np.random.RandomState(0)
        X_train = rng.randn(50, 3)
        y_train = X_train @ [1, 2, 3] + rng.randn(50)
        model = QuantileRegressor().fit(X_train, y_train)
        X_bad = rng.randn(10, 5)  # wrong n_features
        with pytest.raises(ValueError, match="feature"):
            model.predict(X_bad)

    def test_n_features_in_set_after_fit(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 4)
        y = rng.randn(50)
        model = QuantileRegressor().fit(X, y)
        assert model.n_features_in_ == 4


# ── 3. Informative error for n_samples edge cases ───────────────────

class TestEdgeCaseErrors:
    """Fit with very small n_samples should produce informative errors."""

    def test_fit_1_sample_mentions_n_samples(self):
        """Error message for 1-sample fit must contain '1 sample'."""
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        model = QuantileRegressor()
        with pytest.raises(ValueError, match=r"1 sample"):
            model.fit(X, y)

    def test_fit_n_lt_p_with_br_mentions_samples(self):
        """When n < p the BR solver should mention 'n_samples'."""
        rng = np.random.RandomState(42)
        X = rng.randn(5, 20)
        y = rng.randn(5)
        model = QuantileRegressor(method="br")
        with pytest.raises(ValueError, match=r"n_samples"):
            model.fit(X, y)

    @_native_required
    def test_fit_n_lt_p_with_fn_succeeds(self):
        """The FN solver should handle n < p (interior point method)."""
        rng = np.random.RandomState(42)
        n, p = 10, 15
        X = rng.randn(n, p)
        y = rng.randn(n)
        # fit_intercept=False to avoid adding the intercept column
        model = QuantileRegressor(method="fn", fit_intercept=False)
        model.fit(X, y)
        assert hasattr(model, "coef_")


# ── 4. sample_weight with wide data shouldn't blow up ───────────────

@_native_required
class TestSampleWeightWideData:
    """sample_weight path must handle n <= p gracefully."""

    def test_sample_weight_with_wide_data_fn(self):
        """Fitting with sample_weight when n < p using FN should succeed."""
        rng = np.random.RandomState(0)
        X = rng.randn(10, 20)
        y = rng.randn(10)
        sw = np.ones(10)
        model = QuantileRegressor(method="fn", fit_intercept=False)
        model.fit(X, y, sample_weight=sw)
        assert hasattr(model, "coef_")

    def test_sample_weight_with_wide_data_br_raises(self):
        """Fitting with sample_weight when n < p using BR should raise."""
        rng = np.random.RandomState(0)
        X = rng.randn(10, 20)
        y = rng.randn(10)
        sw = np.ones(10)
        model = QuantileRegressor(method="br", fit_intercept=False)
        with pytest.raises(ValueError, match=r"n_samples"):
            model.fit(X, y, sample_weight=sw)


# ── 5. Full check_estimator pass ────────────────────────────────────

@_native_required
class TestSklearnCompliance:
    """The estimator should pass sklearn's full check_estimator suite."""

    def test_check_estimator_passes(self):
        """Run all sklearn checks — no failures expected."""
        # check_estimator raises on the first failure
        check_estimator(QuantileRegressor())
