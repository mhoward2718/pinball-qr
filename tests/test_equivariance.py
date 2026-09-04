"""Koenker's equivariance identities, checked as exact properties.

These hold for the quantile regression estimator by construction (Koenker,
*Quantile Regression*, 2005, Thm 2.2), so they need no reference implementation
and no stored numbers -- if a solver breaks one, the solver is wrong.

What they catch that the optimality certificate does not: every identity
compares *two different fits*, and the reparameterisation one is the only check
in the suite that runs a solver on a transformed design.  A solver that lands on
a different vertex depending on how the problem is scaled certifies as optimal
both times and still fails here.  The reflection identity covers tau <-> 1-tau
symmetry, which is precisely the failure mode of the most recently fixed bug in
this repo (a LASSO penalty that was only symmetric at tau=0.5).
"""

import numpy as np
import pytest


def _has_native():
    try:
        from pinball._native import rqfnb  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_native(), reason="Fortran extension not built"
)

# `pfn` subsamples, so its accuracy is set by the conditioning of the globbed
# reduced problem rather than by the solver tolerance -- see the note in
# tests/test_optimality.py.  Direct solvers are held an order tighter.
ATOL = {"br": 1e-9, "fn": 1e-9, "pfn": 1e-6}
METHODS = ["br", "fn", "pfn"]
TAUS = [0.05, 0.5, 0.95]


def _fit(method, X, y, tau):
    from pinball.linear.solvers import get_solver

    # pfn draws a random subsample; fix the global state so failures are
    # reproducible.  (Exactness means the seed changes the path, not the answer.)
    np.random.seed(0)
    return get_solver(method).solve(X, y, tau).coefficients


@pytest.fixture
def data():
    rng = np.random.RandomState(42)
    n, p = 400, 3
    X = np.column_stack([np.ones(n), rng.randn(n, p - 1)])
    y = X @ np.array([1.0, 2.0, 3.0]) + rng.randn(n)
    return X, y


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("c", [0.5, 2.0, 100.0])
def test_scale_equivariance(data, method, tau, c):
    """beta(tau; c*y, X) == c * beta(tau; y, X) for c > 0."""
    X, y = data
    np.testing.assert_allclose(
        _fit(method, X, c * y, tau), c * _fit(method, X, y, tau),
        atol=ATOL[method] * max(c, 1.0),
        err_msg=f"{method} tau={tau} c={c}: scale equivariance violated",
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("tau", TAUS)
def test_negative_scale_flips_the_quantile(data, method, tau):
    """beta(tau; -y, X) == -beta(1-tau; y, X).  Reflection: the mirror of the
    tau-th quantile of y is the (1-tau)-th quantile of -y."""
    X, y = data
    np.testing.assert_allclose(
        _fit(method, X, -y, tau), -_fit(method, X, y, 1.0 - tau),
        atol=ATOL[method],
        err_msg=f"{method} tau={tau}: reflection equivariance violated",
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("tau", TAUS)
def test_regression_equivariance(data, method, tau):
    """beta(tau; y + X gamma, X) == beta(tau; y, X) + gamma."""
    X, y = data
    gamma = np.array([0.7, -1.3, 2.5])
    np.testing.assert_allclose(
        _fit(method, X, y + X @ gamma, tau), _fit(method, X, y, tau) + gamma,
        atol=ATOL[method],
        err_msg=f"{method} tau={tau}: regression equivariance violated",
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("tau", TAUS)
def test_reparameterization_equivariance(data, method, tau):
    """beta(tau; y, X A) == A^-1 beta(tau; y, X) for nonsingular A.

    The only check in the suite that hands a solver a *different* design matrix,
    so it is the one that exercises the conditioning-sensitive paths (leverage
    bands, the normal-equations Gram matrix, stepy's accumulation).
    """
    X, y = data
    A = np.array([[1.0, 0.4, 0.0], [0.0, 2.0, -0.5], [0.0, 0.0, 1.5]])
    np.testing.assert_allclose(
        _fit(method, X @ A, y, tau), np.linalg.solve(A, _fit(method, X, y, tau)),
        atol=ATOL[method] * 10,
        err_msg=f"{method} tau={tau}: reparameterization equivariance violated",
    )


@pytest.mark.parametrize("tau", TAUS)
def test_estimator_level_equivariance(data, tau):
    """The same identity through the public estimator, intercept handling and
    all -- solver-level correctness does not by itself prove the wrapper is."""
    from pinball.linear import QuantileRegressor

    X, y = data
    Xf = X[:, 1:]                       # let the estimator add its own intercept
    c = 3.0
    a = QuantileRegressor(tau=tau, method="fn").fit(Xf, c * y)
    b = QuantileRegressor(tau=tau, method="fn").fit(Xf, y)
    np.testing.assert_allclose(a.coef_, c * b.coef_, atol=1e-8)
    np.testing.assert_allclose(a.intercept_, c * b.intercept_, atol=1e-8)
