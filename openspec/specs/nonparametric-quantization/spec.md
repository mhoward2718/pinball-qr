## Purpose

Estimate conditional quantiles without assuming a linear model, by compressing
the covariate space onto an optimal quantization grid and reading quantiles off
the resulting Voronoi cells.

## Requirements

### Requirement: Quantization-based conditional quantile estimation

`QuantizationQuantileEstimator` SHALL fit an `N`-point quantization grid over
the covariate space using Competitive Learning Vector Quantization, and SHALL
predict the conditional quantile of a new point from the cell its covariates
fall into.

#### Scenario: Fitting and predicting

- **WHEN** the estimator is fitted on `(X, y)` and asked to predict for new `X`
- **THEN** it returns one conditional quantile estimate per input row

#### Scenario: Recovering a known conditional quantile

- **WHEN** the estimator is fitted at `tau=0.5` on data generated from a
  monotone conditional distribution with a known median function
- **THEN** predictions track the known conditional median across the covariate
  range

#### Scenario: Estimation at an extreme quantile

- **WHEN** the estimator is fitted at `tau=0.9` on the same data
- **THEN** its predictions lie above its `tau=0.5` predictions across the
  covariate range

### Requirement: Bagging over multiple grids

When more than one grid is requested, the estimator SHALL aggregate across
grids **in prediction space** — averaging the per-grid predicted quantiles —
and SHALL NOT average grid point positions, because grid points from
independently initialized grids have no correspondence to one another.

#### Scenario: Aggregating several grids

- **WHEN** the estimator is fitted with `n_grids > 1`
- **THEN** each grid retains its own cell-quantile table and predictions are
  the average of the per-grid predicted values

#### Scenario: Grid coverage is preserved

- **WHEN** the estimator is fitted with `n_grids > 1` on data whose support is
  far from the origin
- **THEN** predictions remain within the observed range of `y` and do not
  collapse toward the data centroid

### Requirement: Reproducibility

Grid initialization and the stochastic CLVQ updates SHALL be reproducible when
a random seed or generator is supplied.

#### Scenario: Fixed seed gives a fixed fit

- **WHEN** the estimator is fitted twice on the same data with the same seed
- **THEN** both fits produce identical predictions
