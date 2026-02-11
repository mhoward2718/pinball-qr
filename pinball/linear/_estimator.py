"""Sklearn-compatible quantile regression estimator.

This is the main public API for the ``pinball`` package.

.. code-block:: python

    from pinball import QuantileRegressor
    model = QuantileRegressor(tau=0.5, method="fn")
    model.fit(X, y)
    y_hat = model.predict(X_new)

sklearn ``check_estimator`` compliance notes
---------------------------------------------
This class passes all 46 checks in ``sklearn.utils.estimator_checks
.check_estimator`` (sklearn 1.8).  Several design choices were made to
satisfy those checks while staying faithful to the R ``quantreg``
package we ported from.  They are documented here so the rationale is
not lost.

1. **Mixin order** – ``RegressorMixin`` before ``BaseEstimator``.
   ``check_mixin_order`` requires specialised mixins first in the MRO.

2. **``validate_data`` instead of ``check_X_y`` / ``check_array``** –
   ``check_n_features_in_after_fitting`` requires that ``predict()``
   raises when the number of input features does not match ``fit()``.
   ``validate_data(self, X, reset=False)`` handles this automatically
   and sets ``n_features_in_`` on ``fit``.

3. **``score()`` returns R² (inherited from ``RegressorMixin``)** –
   ``check_regressors_train`` asserts ``score > 0.5`` on easy data.
   Our previous custom ``score()`` returned negative mean pinball loss,
   which is always ≤ 0.  sklearn's own ``QuantileRegressor`` also
   inherits the default R² scorer.  The pinball loss is still available
   via :meth:`pinball_loss`.

4. **Error message for n_samples = 1** – ``check_fit2d_1sample``
   verifies the error message contains one of ``"1 sample"``,
   ``"n_samples = 1"``, etc.  The base solver's ``_validate_and_prepare``
   raises ``"Got 1 sample …"`` to satisfy this.

5. **``n ≤ p`` check lives in BRSolver, not BaseSolver** –
   The interior-point solver (FNB) handles underdetermined systems,
   so the ``n > p`` constraint is BR-specific (Single Responsibility).
   ``check_sample_weight_equivalence_on_dense_data`` generates data
   with n=15, p=30; if BaseSolver rejected that, the FN path would
   fail unnecessarily.

6. **sample_weight handling** – Three aspects, all matching R/sklearn:

   a. *Weighting formula*: ``w * X``, ``w * y`` — exactly R's
      ``rq.wfit()``.  The pinball loss is positively homogeneous of
      degree 1, so ``ρ_τ(w·u) = w·ρ_τ(u)`` for w > 0.

   b. *Zero-weight filtering*: rows with ``sample_weight == 0`` are
      dropped before fitting.  sklearn's own ``QuantileRegressor``
      does the same (see its source: ``"Filtering out zero sample
      weights from the beginning …"``).  R's ``rq.wfit`` keeps zeros
      (they become all-zero rows), but dropping them is mathematically
      equivalent and avoids artificial rank deficiency.

   c. *Integer-weight row expansion*: when all weights are integers,
      rows are repeated instead of scaled.  This guarantees numerical
      equivalence with ``X.repeat(w)`` / ``y.repeat(w)``, which is
      what ``check_sample_weight_equivalence_on_dense_data`` tests.
      For non-integer weights we fall back to the R-style ``w*X, w*y``.

   d. *Intercept column is added before weighting*: R's ``rq.wfit``
      receives a model matrix that already contains the intercept
      column, then does ``wx <- x * weights``.  We replicate that
      order so the intercept is weighted identically.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from sklearn.utils.validation import check_is_fitted, validate_data

from pinball.estimators._base import BaseQuantileEstimator
from pinball.linear.solvers import get_solver
from pinball.linear.solvers.base import SolverResult


class QuantileRegressor(BaseQuantileEstimator):
    """Quantile regression with an sklearn-compatible interface.

    Wraps the high-performance Fortran solvers from the ``quantreg`` tradition
    behind a familiar ``fit`` / ``predict`` / ``score`` API.

    Parameters
    ----------
    tau : float or array-like of float, default=0.5
        Quantile(s) to estimate.  A single float gives the usual
        ``(n_features_in_,)`` coefficient vector.  A sequence of floats
        (e.g. ``[0.1, 0.5, 0.9]``) fits all quantiles and stores
        ``coef_`` with shape ``(n_features_in_, n_quantiles)``.
    method : str, default="fn"
        Solver back-end.  One of ``"br"``, ``"fn"`` / ``"fnb"``,
        ``"lasso"``, ``"pfn"``.  See :func:`pinball.solvers.list_solvers`.
    fit_intercept : bool, default=True
        Whether to add an intercept column to *X* automatically.
    solver_options : dict or None
        Extra keyword arguments forwarded to the solver's ``solve()``
        method (e.g. ``{"ci": True, "alpha": 0.05}`` for
        ``method="br"``).

    Attributes
    ----------
    coef_ : ndarray, shape (n_features,) or (n_features, n_quantiles)
        Estimated coefficients (excluding intercept).
    intercept_ : float or ndarray
        Intercept term(s).  Zero when ``fit_intercept=False``.
    residuals_ : ndarray
        Residuals from the fit.
    solver_result_ : SolverResult or list[SolverResult]
        Full solver output(s) for advanced users.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_iter_ : int or list[int]
        Number of solver iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from pinball import QuantileRegressor
    >>> X = np.random.randn(200, 3)
    >>> y = X @ [1, 2, 3] + np.random.randn(200)
    >>> model = QuantileRegressor(tau=0.5, method="fn")
    >>> model.fit(X, y)
    QuantileRegressor(...)
    >>> model.predict(X[:5])  # doctest: +SKIP
    array([...])
    """

    def __init__(
        self,
        tau: Union[float, Sequence[float]] = 0.5,
        method: str = "fn",
        fit_intercept: bool = True,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tau = tau
        self.method = method
        self.fit_intercept = fit_intercept
        self.solver_options = solver_options

    # ──────────────────────────────────────────────────────────────────
    # fit / predict / score
    # ──────────────────────────────────────────────────────────────────

    def fit(self, X, y, sample_weight=None):
        """Fit the quantile regression model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,), optional
            Individual weights.  When provided, each observation's
            contribution to the pinball loss is scaled by the
            corresponding weight.  Equivalent to repeating the row
            ``w`` times for integer weights.

        Returns
        -------
        self
        """
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)

        # Prepend intercept column *before* weighting, matching R's
        # rq.wfit() which receives a model matrix that already includes
        # the intercept, then does  wx <- x * weights; wy <- y * weights.
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0], dtype=np.float64), X])

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_weight length mismatch")
            # Filtering out zero sample weights from the beginning
            # makes life easier for the solver (avoids rank deficiency
            # from all-zero rows).  This is mathematically equivalent —
            # zero-weight rows contribute nothing to the pinball loss.
            # sklearn.linear_model.QuantileRegressor does the same.
            nz = sample_weight > 0
            if not np.all(nz):
                X = X[nz]
                y = y[nz]
                sample_weight = sample_weight[nz]
            # For integer weights, expand rows — this is numerically
            # identical to fitting the data with duplicated observations
            # and avoids LP-vertex ambiguity that arises when scaling
            # rows by w (the simplex may reach a different optimum in
            # degenerate problems).  For non-integer weights, fall back
            # to R's rq.wfit() approach: wx = w*X, wy = w*y.
            int_weights = np.allclose(sample_weight,
                                      np.round(sample_weight), atol=1e-12)
            if int_weights:
                repeats = np.round(sample_weight).astype(int)
                X = np.repeat(X, repeats, axis=0)
                y = np.repeat(y, repeats)
            else:
                X = X * sample_weight[:, np.newaxis]
                y = y * sample_weight

        taus = np.atleast_1d(self.tau).ravel()
        solver = get_solver(self.method)
        opts = self.solver_options or {}

        results: list[SolverResult] = []
        for t in taus:
            results.append(solver.solve(X, y, float(t), **opts))

        # Unpack
        if len(taus) == 1:
            res = results[0]
            all_coef = res.coefficients
            if self.fit_intercept:
                self.intercept_ = all_coef[0]
                self.coef_ = all_coef[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = all_coef
            self.residuals_ = res.residuals
            self.solver_result_ = res
            self.n_iter_ = res.iterations
        else:
            coefs = np.column_stack([r.coefficients for r in results])
            if self.fit_intercept:
                self.intercept_ = coefs[0, :]
                self.coef_ = coefs[1:, :]
            else:
                self.intercept_ = np.zeros(len(taus))
                self.coef_ = coefs
            self.residuals_ = np.column_stack([r.residuals for r in results])
            self.solver_result_ = results
            self.n_iter_ = [r.iterations for r in results]

        return self

    def predict(self, X):
        """Predict using the fitted quantile regression model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples,) or (n_samples, n_quantiles)
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return X @ self.coef_ + self.intercept_

    # score() is inherited from RegressorMixin (returns R²), matching
    # sklearn.linear_model.QuantileRegressor's convention.
    # pinball_loss() is inherited from BaseQuantileEstimator.

    # ──────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────

    def summary(self, X, y, se="auto", alpha=0.05, feature_names=None, **kwargs):
        """Produce a summary table with standard errors and CIs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The **design matrix** used in fitting (before ``fit_intercept``
            augmentation).
        y : array-like, shape (n_samples,)
        se : str
            SE method: ``"iid"``, ``"nid"``, ``"ker"``, ``"rank"``,
            ``"boot"``, or ``"auto"``.
        alpha : float
        feature_names : list of str, optional
        **kwargs
            Extra arguments forwarded to the SE estimator.
            For ``se="rank"``: ``iid`` (bool), ``interp`` (bool).
            For ``se="boot"``: ``nboot`` (int), ``method`` (str),
            ``random_state``.

        Returns
        -------
        InferenceResult
        """
        from pinball.linear._inference import summary as _summary

        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)

        if self.fit_intercept:
            X_aug = np.column_stack([np.ones(X.shape[0]), X])
            all_coef = np.concatenate([[self.intercept_], self.coef_])
            if feature_names is not None:
                feature_names = ["const"] + list(feature_names)
        else:
            X_aug = X
            all_coef = self.coef_

        tau = float(np.atleast_1d(self.tau)[0])
        return _summary(X_aug, y, all_coef, tau, se=se, alpha=alpha,
                        feature_names=feature_names, **kwargs)