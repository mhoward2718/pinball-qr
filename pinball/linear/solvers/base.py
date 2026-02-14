"""Abstract base class and result dataclass for quantile regression solvers.

All solvers implement the ``BaseSolver`` interface, producing a standardised
``SolverResult``.  This follows the **Strategy** pattern: the estimator
delegates to an interchangeable solver selected at runtime.

Design principles
-----------------
* **Single Responsibility** — each solver translates inputs to its Fortran /
  NumPy implementation and converts outputs to ``SolverResult``.
* **Open / Closed** — new solvers are added by subclassing; existing code is
  never modified.
* **Liskov Substitution** — every ``BaseSolver`` subclass can be used wherever
  ``BaseSolver`` is expected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinball._typing import FloatArray


@dataclass(frozen=True)
class SolverResult:
    """Standardised output produced by every solver.

    Parameters
    ----------
    coefficients : FloatArray
        Estimated regression coefficients, shape ``(p,)`` for a single
        quantile or ``(p, n_quantiles)`` for multiple quantiles.
    residuals : FloatArray
        Residuals ``y - X @ coefficients``, shape ``(n,)`` or
        ``(n, n_quantiles)``.
    dual_solution : FloatArray or None
        Dual (simplex) solution when available (e.g. from BR).
    objective_value : float
        Value of the quantile-regression objective (weighted pinball loss)
        at the solution.
    status : int
        Solver exit flag.  ``0`` indicates success; other values are
        solver-specific.
    iterations : int
        Number of iterations taken by the solver.
    solver_info : dict
        Solver-specific extra information (e.g. ``sol``, ``dsol``, ``ci``,
        ``tnmat`` from the BR solver).
    """

    coefficients: FloatArray
    residuals: FloatArray
    dual_solution: FloatArray | None = None
    objective_value: float = 0.0
    status: int = 0
    iterations: int = 0
    solver_info: dict[str, Any] = field(default_factory=dict)


class BaseSolver(ABC):
    """Abstract base class that every quantile-regression solver must implement.

    Subclasses **must** override :meth:`solve`.  They **may** override
    :meth:`supports_multiple_quantiles` to advertise batch-τ capability and
    :meth:`validate_inputs` to add solver-specific input checks.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
        **kwargs: Any,
    ) -> SolverResult:
        """Fit a quantile-regression model and return the result.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix (including intercept column if desired).
        y : np.ndarray, shape (n,)
            Response vector.
        tau : float
            Quantile level in (0, 1).
        **kwargs
            Solver-specific options.

        Returns
        -------
        SolverResult

        Raises
        ------
        ValueError
            If the inputs fail validation.
        """
        X, y, tau = self._validate_and_prepare(X, y, tau)
        return self._solve_impl(X, y, tau, **kwargs)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _solve_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
        **kwargs: Any,
    ) -> SolverResult:
        """Core solving logic — implemented by every concrete solver."""

    def validate_inputs(  # noqa: B027
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
    ) -> None:
        """Solver-specific validation, called after generic checks.

        Override this to add extra checks (e.g. BR requires ``n > p``).
        The default implementation does nothing.
        """

    @staticmethod
    def supports_multiple_quantiles() -> bool:
        """Return ``True`` if the solver can fit all τ in one call."""
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_and_prepare(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
    ) -> tuple:
        """Run generic + solver-specific validation and ensure dtypes."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim}-D array.")
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError(
                f"X and y have incompatible shapes: X is ({n}, {p}), "
                f"y has {y.shape[0]} elements."
            )
        if not (0 < tau < 1):
            raise ValueError(f"tau must be in (0, 1), got {tau}.")
        if n < 2:
            raise ValueError(
                "Got 1 sample, need at least 2 n_samples for "
                "quantile regression."
            )

        # Solver-specific checks
        self.validate_inputs(X, y, tau)

        return X, y, tau
