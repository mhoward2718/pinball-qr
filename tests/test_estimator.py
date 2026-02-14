"""Tests for the QuantileRegressor sklearn-compatible estimator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pinball._estimator import QuantileRegressor
from pinball.linear.solvers.base import SolverResult

# ──────────────────────────────────────────────────────────────────────
# Helper: mock solver that returns known coefficients
# ──────────────────────────────────────────────────────────────────────

def _make_mock_solver(coef):
    """Return a mock solver whose solve() returns the given coefficients."""
    mock = MagicMock()
    mock.solve.return_value = SolverResult(
        coefficients=np.asarray(coef, dtype=np.float64),
        residuals=np.zeros(10),
        objective_value=0.0,
        status=0,
        iterations=1,
    )
    return mock


class TestQuantileRegressorInit:

    def test_default_params(self):
        model = QuantileRegressor()
        assert model.tau == 0.5
        assert model.method == "fn"
        assert model.fit_intercept is True
        assert model.solver_options is None

    def test_custom_params(self):
        model = QuantileRegressor(tau=0.9, method="br", fit_intercept=False)
        assert model.tau == 0.9
        assert model.method == "br"
        assert model.fit_intercept is False


class TestQuantileRegressorFit:

    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n, 2)
        y = X @ [3, 5] + 10 + rng.randn(n) * 0.5
        return X, y

    def test_fit_returns_self(self, data):
        X, y = data
        mock_solver = _make_mock_solver([10.0, 3.0, 5.0])

        with patch("pinball.linear._estimator.get_solver", return_value=mock_solver):
            model = QuantileRegressor(tau=0.5, method="fn")
            result = model.fit(X, y)
            assert result is model

    def test_fit_sets_attributes(self, data):
        X, y = data
        mock_solver = _make_mock_solver([10.0, 3.0, 5.0])

        with patch("pinball.linear._estimator.get_solver", return_value=mock_solver):
            model = QuantileRegressor(tau=0.5).fit(X, y)

        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")
        assert hasattr(model, "residuals_")
        assert hasattr(model, "solver_result_")
        assert hasattr(model, "n_features_in_")
        assert model.n_features_in_ == 2

    def test_fit_intercept_true(self, data):
        X, y = data
        mock_solver = _make_mock_solver([10.0, 3.0, 5.0])

        with patch("pinball.linear._estimator.get_solver", return_value=mock_solver):
            model = QuantileRegressor(tau=0.5, fit_intercept=True).fit(X, y)

        assert model.intercept_ == 10.0
        np.testing.assert_array_equal(model.coef_, [3.0, 5.0])

    def test_fit_intercept_false(self, data):
        X, y = data
        mock_solver = _make_mock_solver([3.0, 5.0])

        with patch("pinball.linear._estimator.get_solver", return_value=mock_solver):
            model = QuantileRegressor(tau=0.5, fit_intercept=False).fit(X, y)

        assert model.intercept_ == 0.0
        np.testing.assert_array_equal(model.coef_, [3.0, 5.0])

    def test_multi_quantile(self, data):
        X, y = data
        mock_solver = MagicMock()
        mock_solver.solve.side_effect = [
            SolverResult(
                coefficients=np.array([5.0, 2.0, 4.0]),
                residuals=np.zeros(100),
                objective_value=0.0, status=0, iterations=1,
            ),
            SolverResult(
                coefficients=np.array([10.0, 3.0, 5.0]),
                residuals=np.zeros(100),
                objective_value=0.0, status=0, iterations=1,
            ),
            SolverResult(
                coefficients=np.array([15.0, 4.0, 6.0]),
                residuals=np.zeros(100),
                objective_value=0.0, status=0, iterations=1,
            ),
        ]

        with patch("pinball.linear._estimator.get_solver", return_value=mock_solver):
            model = QuantileRegressor(tau=[0.1, 0.5, 0.9]).fit(X, y)

        assert model.coef_.shape == (2, 3)
        assert model.intercept_.shape == (3,)
        np.testing.assert_array_equal(model.intercept_, [5.0, 10.0, 15.0])

    def test_sample_weight(self, data):
        X, y = data
        mock_solver = _make_mock_solver([10.0, 3.0, 5.0])
        weights = np.ones(len(y))

        with patch("pinball.linear._estimator.get_solver", return_value=mock_solver):
            QuantileRegressor().fit(X, y, sample_weight=weights)

        # Solver should have been called once
        mock_solver.solve.assert_called_once()


class TestQuantileRegressorPredict:

    def test_predict_shape(self):
        mock_solver = _make_mock_solver([10.0, 3.0, 5.0])
        with patch("pinball.linear._estimator.get_solver", return_value=mock_solver):
            model = QuantileRegressor().fit(np.random.randn(50, 2), np.random.randn(50))
        pred = model.predict(np.random.randn(10, 2))
        assert pred.shape == (10,)

    def test_predict_not_fitted_raises(self):
        from sklearn.exceptions import NotFittedError
        model = QuantileRegressor()
        with pytest.raises(NotFittedError):
            model.predict(np.random.randn(5, 2))

    def test_predict_values(self):
        mock_solver = _make_mock_solver([1.0, 2.0, 3.0])
        X_train = np.random.randn(50, 2)
        y_train = np.random.randn(50)
        with patch("pinball.linear._estimator.get_solver", return_value=mock_solver):
            model = QuantileRegressor().fit(X_train, y_train)

        X_test = np.array([[1.0, 0.0], [0.0, 1.0]])
        pred = model.predict(X_test)
        # intercept=1, coef=[2, 3]
        np.testing.assert_allclose(pred, [1 + 2 * 1 + 3 * 0, 1 + 2 * 0 + 3 * 1])


class TestQuantileRegressorScore:

    def test_score_returns_float(self):
        mock_solver = _make_mock_solver([1.0])
        X = np.random.randn(50, 1)
        y = np.random.randn(50)
        with patch("pinball.linear._estimator.get_solver", return_value=mock_solver):
            model = QuantileRegressor(fit_intercept=False).fit(X, y)
        score = model.score(X, y)
        assert isinstance(score, float)

    def test_score_negative(self):
        """Score (R²) is negative for a very bad fit."""
        mock_solver = _make_mock_solver([0.0])  # bad fit: predicts zero
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        y = rng.randn(100) + 5  # clearly non-zero mean
        with patch("pinball.linear._estimator.get_solver", return_value=mock_solver):
            model = QuantileRegressor(fit_intercept=False).fit(X, y)
        score = model.score(X, y)
        assert score < 0
