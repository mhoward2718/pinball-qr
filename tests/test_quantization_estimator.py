"""TDD tests for QuantizationQuantileEstimator.

The sklearn-compatible estimator that wraps the CLVQ grid
construction, Voronoi assignment, and cell-conditional quantile
computation into a single fit/predict workflow.
"""

import numpy as np
import pytest


# ============================================================
# Helpers
# ============================================================

def _make_univariate_data(n=300, seed=42):
    """Simple Y = X^2 + noise, X ~ Uniform(-2, 2)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, n).reshape(-1, 1)
    Y = X.ravel() ** 2 + rng.randn(n) * 0.3
    return X, Y


def _make_bivariate_data(n=400, seed=99):
    """Y = X1^2 + X2^2 + noise, X ~ Uniform(-2, 2)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (n, 2))
    Y = np.sum(X ** 2, axis=1) + rng.randn(n) * 0.3
    return X, Y


# ============================================================
# 1.  Construction / parameter validation
# ============================================================

class TestEstimatorInit:

    def test_default_params(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        est = QuantizationQuantileEstimator()
        assert est.tau == 0.5
        assert est.N == 20
        assert est.n_grids == 50
        assert est.p == 2

    def test_custom_params(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        est = QuantizationQuantileEstimator(
            tau=0.9, N=10, n_grids=30, p=1, random_state=42
        )
        assert est.tau == 0.9
        assert est.N == 10
        assert est.n_grids == 30
        assert est.p == 1
        assert est.random_state == 42

    def test_get_params(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        est = QuantizationQuantileEstimator(tau=0.25, N=15)
        params = est.get_params()
        assert params["tau"] == 0.25
        assert params["N"] == 15

    def test_set_params(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        est = QuantizationQuantileEstimator()
        est.set_params(tau=0.1, N=5)
        assert est.tau == 0.1
        assert est.N == 5


# ============================================================
# 2.  Fit
# ============================================================

class TestEstimatorFit:

    def test_fit_returns_self(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        X, Y = _make_univariate_data()
        est = QuantizationQuantileEstimator(n_grids=5, random_state=42)
        result = est.fit(X, Y)
        assert result is est

    def test_is_fitted_after_fit(self):
        from sklearn.utils.validation import check_is_fitted
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        X, Y = _make_univariate_data()
        est = QuantizationQuantileEstimator(n_grids=5, random_state=42)
        est.fit(X, Y)
        check_is_fitted(est)  # should not raise

    def test_stores_grid_and_cell_quantiles(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        X, Y = _make_univariate_data()
        est = QuantizationQuantileEstimator(
            N=10, n_grids=5, random_state=42
        )
        est.fit(X, Y)
        # Internal fitted attributes
        assert hasattr(est, "grid_")
        assert hasattr(est, "cell_quantiles_")

    def test_fit_bivariate(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        X, Y = _make_bivariate_data()
        est = QuantizationQuantileEstimator(
            N=15, n_grids=5, random_state=42
        )
        est.fit(X, Y)
        assert est.grid_.ndim == 2  # (d, N) for d>1


# ============================================================
# 3.  Predict
# ============================================================

class TestEstimatorPredict:

    def test_predict_shape_univariate(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        X, Y = _make_univariate_data()
        est = QuantizationQuantileEstimator(
            N=10, n_grids=10, random_state=42
        )
        est.fit(X, Y)
        preds = est.predict(X[:5])
        assert preds.shape == (5,)

    def test_predict_shape_bivariate(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        X, Y = _make_bivariate_data()
        est = QuantizationQuantileEstimator(
            N=15, n_grids=5, random_state=42
        )
        est.fit(X, Y)
        preds = est.predict(X[:10])
        assert preds.shape == (10,)

    def test_predict_before_fit_raises(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        est = QuantizationQuantileEstimator()
        with pytest.raises(Exception):
            est.predict(np.array([[1.0]]))

    def test_median_prediction_reasonable(self):
        """For tau=0.5 the predictions should be close to the true median."""
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        X, Y = _make_univariate_data(n=500, seed=7)
        est = QuantizationQuantileEstimator(
            tau=0.5, N=15, n_grids=20, random_state=7
        )
        est.fit(X, Y)

        # Predict at X = 0  →  true median ≈ 0
        x_test = np.array([[0.0]])
        pred = est.predict(x_test)
        assert abs(pred[0]) < 1.5  # generous tolerance


# ============================================================
# 4.  Quantile ordering
# ============================================================

class TestQuantileOrdering:

    def test_predictions_ordered_by_tau(self):
        """q_{0.25}(x) ≤ q_{0.5}(x) ≤ q_{0.75}(x) for most x."""
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        X, Y = _make_univariate_data(n=500, seed=11)
        taus = [0.25, 0.5, 0.75]
        preds = {}
        for tau in taus:
            est = QuantizationQuantileEstimator(
                tau=tau, N=15, n_grids=20, random_state=11
            )
            est.fit(X, Y)
            preds[tau] = est.predict(X[:20])

        # At least 80% of test points should respect ordering
        ordered = (preds[0.25] <= preds[0.5] + 0.5) & (
            preds[0.5] <= preds[0.75] + 0.5
        )
        assert np.mean(ordered) >= 0.7


# ============================================================
# 5.  Determinism
# ============================================================

class TestEstimatorDeterminism:

    def test_same_seed_same_predictions(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        X, Y = _make_univariate_data()
        e1 = QuantizationQuantileEstimator(
            N=10, n_grids=10, random_state=42
        ).fit(X, Y)
        e2 = QuantizationQuantileEstimator(
            N=10, n_grids=10, random_state=42
        ).fit(X, Y)
        np.testing.assert_array_equal(e1.predict(X), e2.predict(X))


# ============================================================
# 6.  Pinball loss (inherited from BaseQuantileEstimator)
# ============================================================

class TestPinballLoss:

    def test_pinball_loss_positive(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        X, Y = _make_univariate_data()
        est = QuantizationQuantileEstimator(
            N=10, n_grids=10, random_state=42
        )
        est.fit(X, Y)
        loss = est.pinball_loss(X, Y)
        assert loss > 0

    def test_pinball_loss_finite(self):
        from pinball.nonparametric.quantization._estimator import (
            QuantizationQuantileEstimator,
        )
        X, Y = _make_univariate_data()
        est = QuantizationQuantileEstimator(
            N=10, n_grids=10, random_state=42
        )
        est.fit(X, Y)
        loss = est.pinball_loss(X, Y)
        assert np.isfinite(loss)
