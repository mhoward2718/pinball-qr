## MODIFIED Requirements

### Requirement: Single and multiple quantile levels

The estimator SHALL accept `tau` as either a single float in (0, 1) or a
sequence of floats, and SHALL shape its fitted attributes accordingly. A
sequence SHALL be fitted through the solver's batch interface, so that solvers
able to share work across levels do so, while results remain identical to
fitting each level separately.

#### Scenario: Single quantile

- **WHEN** `tau=0.5` is fitted on data with `p` features
- **THEN** `coef_` has shape `(p,)` and `intercept_` is a scalar

#### Scenario: Multiple quantiles

- **WHEN** `tau=[0.1, 0.5, 0.9]` is fitted on data with `p` features
- **THEN** `coef_` has shape `(p, 3)`, `intercept_` has shape `(3,)`, and
  `predict(X)` returns shape `(n, 3)`

#### Scenario: A grid fit matches individual fits

- **WHEN** the same level is fitted alone and as part of a multi-level fit
- **THEN** the coefficients agree to solver tolerance

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
`solver_options` SHALL be the route by which solver-specific settings, including
a random seed for solvers that sample, reach the solver.

#### Scenario: Choosing a solver by name

- **WHEN** `method="br"` is requested
- **THEN** the Barrodale-Roberts solver is used and `solver_result_` holds
  that solver's `SolverResult`

#### Scenario: Unknown solver name

- **WHEN** `method` names a solver that is not registered
- **THEN** a `ValueError` naming the unknown method is raised

#### Scenario: Seeding a sampling solver

- **WHEN** a preprocessing fit is run twice with the same seed passed through
  `solver_options`
- **THEN** both fits produce identical coefficients

## ADDED Requirements

### Requirement: Degenerate sample weights are reported in terms of weights

Zero-weight observations are dropped before fitting, so weights that are
everywhere zero leave nothing to fit. The estimator SHALL reject that case with
an error naming the weights, rather than letting an empty design reach the
solver and surface as a generic "not enough samples".

#### Scenario: All weights are zero

- **WHEN** `fit` is called with a `sample_weight` vector that is entirely zero
- **THEN** a `ValueError` mentioning zero weights is raised

#### Scenario: Some weights are zero

- **WHEN** only some weights are zero
- **THEN** those observations are dropped and the fit proceeds normally
