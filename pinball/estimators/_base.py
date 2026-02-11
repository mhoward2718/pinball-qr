"""Abstract base class for all conditional quantile estimators.

Every estimator in ``pinball`` — whether parametric (linear) or
nonparametric (quantization, kernel, …) — inherits from
:class:`BaseQuantileEstimator`.  This guarantees a consistent public
API (``fit`` / ``predict`` / ``score`` / ``pinball_loss``) and full
sklearn ``check_estimator`` compliance.

Design principles
-----------------
* **Interface Segregation** — the base class contains *only* what is
  common to all conditional quantile estimators: the ``tau`` parameter,
  the ``pinball_loss()`` method, and the ``fit``/``predict`` contract.
* **Liskov Substitution** — any ``BaseQuantileEstimator`` subclass can
  be used wherever the base is expected.
* **Open / Closed** — new estimation families are added by subclassing;
  existing code is never modified.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class BaseQuantileEstimator(RegressorMixin, BaseEstimator):
    """Abstract base for all conditional quantile estimators.

    Subclasses **must** implement :meth:`fit` and :meth:`predict`.

    Parameters
    ----------
    tau : float or array-like of float
        Quantile level(s) in (0, 1).

    Notes
    -----
    ``score()`` is inherited from ``RegressorMixin`` (R²), matching
    sklearn's own ``QuantileRegressor`` convention.  The quantile-specific
    loss is available via :meth:`pinball_loss`.
    """

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit the model.  Must be overridden by subclasses."""

    @abstractmethod
    def predict(self, X):
        """Predict.  Must be overridden by subclasses."""

    # ──────────────────────────────────────────────────────────────
    # Shared concrete methods
    # ──────────────────────────────────────────────────────────────

    def pinball_loss(self, X, y):
        r"""Return the mean pinball (check) loss on ``(X, y)``.

        .. math::
            \frac{1}{n}\sum_i \rho_\tau(y_i - \hat y_i)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        float
        """
        check_is_fitted(self)
        y_pred = self.predict(X)
        taus = np.atleast_1d(self.tau)

        if y_pred.ndim == 1:
            residuals = y - y_pred
            loss = np.where(
                residuals >= 0,
                taus[0] * residuals,
                (taus[0] - 1) * residuals,
            )
            return float(np.mean(loss))
        else:
            y_col = np.asarray(y)[:, np.newaxis]
            residuals = y_col - y_pred
            loss = np.where(
                residuals >= 0,
                taus[np.newaxis, :] * residuals,
                (taus[np.newaxis, :] - 1) * residuals,
            )
            return float(np.mean(loss))
