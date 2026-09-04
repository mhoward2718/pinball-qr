## Purpose

Provide linear (parametric) conditional quantile regression through an
sklearn-compatible estimator, so that quantile models can be fit, predicted
from, and scored using the same API and tooling as any other scikit-learn
estimator.

## Requirements

### Requirement: sklearn-compatible estimator interface

`QuantileRegressor` SHALL implement the scikit-learn estimator contract:
constructor parameters stored unmodified on `self`, a `fit(X, y)` returning
`self`, and `predict(X)` / `score(X, y)` available after fitting. It SHALL be
usable inside scikit-learn pipelines and cross-validation utilities.

#### Scenario: Fitting and predicting

- **WHEN** a user calls `fit(X, y)` on a `QuantileRegressor` with valid data
- **THEN** the estimator returns `self`, sets `coef_`, `intercept_`,
  `residuals_`, `n_features_in_` and `n_iter_`, and `predict(X)` returns one
  prediction per row of `X`

#### Scenario: Used inside a scikit-learn pipeline

- **WHEN** a `QuantileRegressor` is placed in a `Pipeline` and passed to
  `cross_val_score`
- **THEN** it clones, fits and scores without error

#### Scenario: Predicting before fitting

- **WHEN** `predict` is called on an unfitted estimator
- **THEN** scikit-learn's `NotFittedError` is raised

### Requirement: Single and multiple quantile levels

The estimator SHALL accept `tau` as either a single float in (0, 1) or a
sequence of floats, and SHALL shape its fitted attributes accordingly.

#### Scenario: Single quantile

- **WHEN** `tau=0.5` is fitted on data with `p` features
- **THEN** `coef_` has shape `(p,)` and `intercept_` is a scalar

#### Scenario: Multiple quantiles

- **WHEN** `tau=[0.1, 0.5, 0.9]` is fitted on data with `p` features
- **THEN** `coef_` has shape `(p, 3)`, `intercept_` has shape `(3,)`, and
  `predict(X)` returns shape `(n, 3)`

#### Scenario: Extreme quantile levels

- **WHEN** the estimator is fitted at `tau=0.01` and at `tau=0.99` on data
  with a genuine location shift across the conditional distribution
- **THEN** both fits succeed and the `tau=0.99` fitted line lies above the
  `tau=0.01` fitted line over the observed range of `X`

#### Scenario: Invalid quantile level

- **WHEN** `tau` is outside (0, 1) — for example `0`, `1`, or `1.5`
- **THEN** a `ValueError` is raised at fit time

### Requirement: Solver selection

The estimator SHALL select its solver at runtime by the `method` name from the
solver registry, and SHALL forward `solver_options` to the solver unchanged.

#### Scenario: Choosing a solver by name

- **WHEN** `method="br"` is requested
- **THEN** the Barrodale-Roberts solver is used and `solver_result_` holds
  that solver's `SolverResult`

#### Scenario: Unknown solver name

- **WHEN** `method` names a solver that is not registered
- **THEN** a `ValueError` naming the unknown method is raised

### Requirement: Intercept handling

The estimator SHALL add an intercept column automatically when
`fit_intercept=True` and SHALL report `intercept_` as zero when
`fit_intercept=False`, with `coef_` excluding the intercept in both cases.

#### Scenario: Intercept fitted

- **WHEN** `fit_intercept=True` and the data has a nonzero conditional median
  at `X = 0`
- **THEN** `intercept_` is nonzero and `coef_` has length equal to the number
  of input features

#### Scenario: Intercept suppressed

- **WHEN** `fit_intercept=False`
- **THEN** `intercept_` is `0.0` and predictions equal `X @ coef_`
