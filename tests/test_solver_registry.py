"""Tests for the solver registry."""

import pytest

from pinball.linear.solvers import get_solver, list_solvers, register_solver
from pinball.linear.solvers.base import BaseSolver, SolverResult


class _DummySolver(BaseSolver):
    def _solve_impl(self, X, y, tau, **kwargs):
        import numpy as np
        return SolverResult(
            coefficients=np.zeros(X.shape[1]),
            residuals=y.copy(),
        )


class TestRegistry:

    def test_list_solvers_not_empty(self):
        solvers = list_solvers()
        assert len(solvers) > 0

    def test_builtin_br_registered(self):
        assert "br" in list_solvers()

    def test_builtin_fn_registered(self):
        assert "fn" in list_solvers()

    def test_builtin_fnb_registered(self):
        assert "fnb" in list_solvers()

    def test_get_solver_returns_instance(self):
        solver = get_solver("br")
        assert isinstance(solver, BaseSolver)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown solver"):
            get_solver("nonexistent_solver_xyz")

    def test_register_custom_solver(self):
        register_solver("_test_dummy", _DummySolver)
        assert "_test_dummy" in list_solvers()
        solver = get_solver("_test_dummy")
        assert isinstance(solver, _DummySolver)

    def test_register_non_solver_raises(self):
        with pytest.raises(TypeError, match="not a BaseSolver"):
            register_solver("bad", str)
