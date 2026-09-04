## Purpose

Hold the package to numerical agreement with the R reference implementations it
is ported from, so that "fast quantile regression in Python" means the same
answers as `quantreg`, not merely similar ones.

## Requirements

### Requirement: Estimation matches the R reference

Fitted coefficients SHALL agree numerically with the corresponding R reference
function — `quantreg` for the linear solvers and inference, `QuantifQuantile`
for the quantization estimator — to documented tolerance, on the reference
datasets shipped with the package.

#### Scenario: Linear fit matches quantreg

- **WHEN** `QuantileRegressor` is fitted on a shipped reference dataset at a
  given `tau`
- **THEN** its coefficients match those of the corresponding `quantreg::rq`
  fit to documented tolerance

#### Scenario: Parity holds away from the median

- **WHEN** the same comparison is run at an extreme `tau` such as 0.05 or 0.95
- **THEN** parity holds to the same documented tolerance

**Rationale:** several past defects were invisible at `tau=0.5` and appeared
only in the tails, so median-only comparison is not sufficient evidence of
parity.

### Requirement: Deliberate divergence from R must be recorded

Where this package intentionally departs from the R reference, the divergence
SHALL be documented with its justification, rather than left as an unexplained
numerical difference.

#### Scenario: A justified departure

- **WHEN** a solver deliberately behaves differently from its R counterpart
  (for example, warning on a condition R passes over silently)
- **THEN** the difference and its reason are recorded in the capability's spec
  or in the solver's own documentation

### Requirement: Provenance and attribution are preserved

Ported code SHALL retain attribution to its upstream authors and license.

#### Scenario: Modifying ported code

- **WHEN** a change is made to code ported from `quantreg`, `QuantifQuantile`
  or POGS
- **THEN** the existing attribution and license notices remain intact
