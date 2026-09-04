## Purpose

Provide standard errors, confidence intervals and bootstrap distributions for
fitted linear quantile regression models, so that coefficients can be reported
with defensible uncertainty rather than as point estimates alone.

## Requirements

### Requirement: Standard error methods

`summary()` SHALL support the standard-error families inherited from the R
reference — independent-and-identically-distributed (`iid`), non-identically
distributed (`nid`), kernel (`ker`) and rank-inversion (`rank`) — and SHALL
report which method produced the result.

#### Scenario: Requesting a standard error method

- **WHEN** `summary()` is called on a fitted model with a supported method name
- **THEN** an `InferenceResult` is returned carrying one standard error per
  coefficient and the method name used

#### Scenario: Unsupported method name

- **WHEN** `summary()` is called with an unrecognized method name
- **THEN** a `ValueError` listing the supported methods is raised

#### Scenario: Inference at an extreme quantile

- **WHEN** `summary()` is called on a model fitted at `tau=0.95`
- **THEN** it returns finite, positive standard errors for every coefficient

### Requirement: Rank-inversion confidence intervals

Rank-inversion confidence intervals SHALL be produced by the simplex solver's
confidence-interval path and SHALL bracket the corresponding point estimate.

#### Scenario: Interval contains the point estimate

- **WHEN** rank-inversion intervals are computed at level `alpha` for a fitted
  model
- **THEN** each coefficient's lower bound is at most, and its upper bound at
  least, the fitted coefficient

### Requirement: Bootstrap methods

The bootstrap SHALL support the xy-pair, wild, and MCMB resampling schemes,
SHALL return the full replicate distribution alongside summary statistics, and
SHALL be reproducible when a random seed or generator is supplied.

#### Scenario: Reproducibility under a fixed seed

- **WHEN** `bootstrap()` is run twice with the same data and the same seed
- **THEN** both runs produce identical replicate coefficients

#### Scenario: Replicate distribution returned

- **WHEN** `bootstrap()` is run with `n_replicates = B`
- **THEN** the `BootstrapResult` exposes `B` replicate coefficient vectors and
  a standard error derived from them
