## ADDED Requirements

### Requirement: Smoothed (approximate) solvers are distinguishable from exact ones

The package MAY provide solvers that minimise a smoothed surrogate of the
pinball loss rather than the loss itself. Such a solver SHALL make its
approximate nature discoverable at runtime rather than only in documentation,
and SHALL NOT be held to the agreement requirement that binds exact solvers.

#### Scenario: A smoothed solver reports what it did

- **WHEN** a smoothed solver returns a result
- **THEN** `solver_info` records the smoothing bandwidth and kernel used, so the
  estimand is recoverable from the result alone

#### Scenario: A smoothed fit is close to, but not equal to, the exact fit

- **WHEN** the same problem is solved by a smoothed solver and by an exact one
  at the same quantile level
- **THEN** the two differ by an amount consistent with the smoothing bias, and
  the smoothed fit is not required to reproduce the exact one

#### Scenario: Shrinking the bandwidth recovers the exact estimator

- **WHEN** a smoothed solver is run over a sequence of decreasing bandwidths
- **THEN** its fit moves toward the exact quantile regression fit

**Rationale:** this is the property that ties an approximate estimator back to
the exact ones that are independently verified; without it a self-consistent but
wrong objective would pass every other check.

### Requirement: Smoothed solvers certify their own optimality

Because a smoothed objective is differentiable, a smoothed solver's result
SHALL satisfy the first-order condition for that objective, and this SHALL be
checkable without reference to any other implementation.

#### Scenario: The gradient vanishes at the returned coefficients

- **WHEN** a smoothed solver returns a converged fit
- **THEN** the gradient of its smoothed objective at those coefficients is zero
  to solver tolerance

#### Scenario: Extreme quantile levels

- **WHEN** a smoothed solver is run at levels as extreme as 0.05 and 0.95
- **THEN** the first-order condition still holds at the returned coefficients

### Requirement: Interval estimates for smoothed fits

A smoothed solver offering confidence intervals SHALL provide both an
asymptotic and a resampling route, SHALL bracket its own point estimate, and
SHALL be reproducible when a seed is supplied.

#### Scenario: Intervals contain the estimate

- **WHEN** confidence intervals are requested at level alpha
- **THEN** every interval's lower bound is at most, and upper bound at least,
  the corresponding fitted coefficient

#### Scenario: The two routes agree

- **WHEN** asymptotic and resampling standard errors are computed for the same
  fit
- **THEN** they agree to within sampling error, since both estimate the same
  quantity

#### Scenario: Reproducible resampling

- **WHEN** the resampling intervals are computed twice with the same seed
- **THEN** the replicates are identical
