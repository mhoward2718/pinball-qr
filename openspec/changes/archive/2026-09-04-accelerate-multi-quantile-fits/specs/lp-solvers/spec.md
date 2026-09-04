## ADDED Requirements

### Requirement: Batch fitting of a quantile grid

Solvers SHALL accept a whole grid of quantile levels through `solve_multi`,
returning one result per level **in the order the caller supplied**. The default
implementation solves each level independently, so a solver that gains nothing
from the grid keeps its existing behaviour exactly. Solvers that can exploit the
grid SHALL still return results identical, to solver tolerance, to solving each
level on its own — the grid is a performance affordance, never a change of answer.

#### Scenario: Results come back in the requested order

- **WHEN** `solve_multi` is called with levels in a non-monotonic order such as
  `[0.9, 0.1, 0.5]`
- **THEN** the i-th returned result is the fit for the i-th requested level,
  whatever internal order the solver used

#### Scenario: Grid fitting agrees with individual fits

- **WHEN** a solver fits `[0.1, 0.5, 0.9]` through `solve_multi`
- **THEN** each result matches that level fitted on its own to solver tolerance

#### Scenario: An invalid level is rejected wherever it appears

- **WHEN** the grid contains a level outside (0, 1) at any position
- **THEN** a `ValueError` is raised before any fitting is done

#### Scenario: Grid fitting at extreme levels

- **WHEN** a grid spanning `0.01` to `0.99` is fitted
- **THEN** every level returns a fit that satisfies the quantile regression
  optimality condition

### Requirement: Preprocessing shares work across a quantile grid

The preprocessing solver SHALL, when given a grid, draw one subsample and
compute one confidence band for the whole grid and seed each level with the
previous level's residuals. Correctness SHALL NOT depend on how finely the grid
is spaced — only speed may.

#### Scenario: A coarse grid is as correct as a fine one

- **WHEN** the same level is fitted alone, inside a 3-point grid, and inside a
  33-point grid
- **THEN** all three agree to solver tolerance

#### Scenario: Sharing work actually reduces work

- **WHEN** a grid of levels is fitted through the grid path and, separately, one
  level at a time
- **THEN** the grid path performs strictly fewer inner solver iterations

**Rationale:** preprocessing is exact regardless of how good the seed is — the
sign-verification loop repairs a bad one — so no comparison of coefficients can
show that sharing is wired up correctly. Only a count can.

### Requirement: Preprocessing terminates

The preprocessing solver SHALL always terminate. When it cannot verify its
aggregation within the allowed number of fixup rounds it SHALL enlarge the
subsample rather than retry at the same size, and SHALL never return a fit whose
aggregation was not verified.

#### Scenario: The fixup budget is exhausted

- **WHEN** the solver is configured to allow no fixup rounds at all
- **THEN** it enlarges the subsample, warns, and returns an optimal fit rather
  than retrying indefinitely

### Requirement: Reproducible subsampling

Solvers that sample SHALL accept a random seed following the package convention,
and SHALL produce identical fits for identical seeds.

#### Scenario: Same seed, same fit

- **WHEN** the preprocessing solver is run twice with the same seed and data
- **THEN** the two fits are bitwise identical

#### Scenario: Different seeds, same answer

- **WHEN** it is run with two different seeds
- **THEN** both fits agree to solver tolerance, because the seed changes which
  observations are aggregated but not the optimum

## MODIFIED Requirements

### Requirement: Solvers report rather than hide numerical failure

A solver SHALL NOT silently return an untrustworthy fit. When the underlying
routine reports a non-success status, fails to reach its requested tolerance,
exhausts its iteration budget, **or returns non-finite coefficients**, the
solver SHALL surface that condition to the caller via a warning and via
`SolverResult.status`. Parameters that would drive the underlying routine into
such a state SHALL be rejected at construction time.

#### Scenario: Underlying routine reports a failure status

- **WHEN** a solver's native routine returns a non-zero/non-success status
  (for example a singular design)
- **THEN** a warning is raised and the non-success status is recorded in
  `SolverResult.status`

#### Scenario: Iteration budget exhausted before convergence

- **WHEN** an iterative solver reaches its maximum iteration count without
  meeting its convergence tolerance
- **THEN** the caller is warned that the result did not converge, and the
  condition is recorded in `SolverResult.status`

#### Scenario: Non-finite coefficients

- **WHEN** the underlying routine returns coefficients containing NaN or
  infinity, even with a success status
- **THEN** a warning is raised and the failure is recorded in
  `SolverResult.status`

#### Scenario: A tolerance that cannot be satisfied is rejected

- **WHEN** an interior-point solver is constructed with a non-positive
  convergence tolerance
- **THEN** a `ValueError` is raised, rather than the solve running on to produce
  non-finite output

### Requirement: Uniform solver interface

Every solver SHALL subclass `BaseSolver` and implement
`solve(X, y, tau, **kwargs)` returning a `SolverResult` carrying
`coefficients`, `residuals`, `objective_value`, `status`, `iterations` and
`solver_info`. Where the underlying method produces a dual solution, the solver
SHALL expose it in `SolverResult.dual_solution` so callers can verify optimality
without re-solving.

#### Scenario: Any registered solver satisfies the contract

- **WHEN** any solver returned by `get_solver(name)` is asked to solve a
  well-conditioned problem
- **THEN** it returns a `SolverResult` whose `coefficients` has length `p` and
  whose `residuals` has length `n`

#### Scenario: Objective value is the pinball loss

- **WHEN** a solver returns a `SolverResult` for level `tau`
- **THEN** `objective_value` equals the pinball loss
  `tau * sum(max(r, 0)) + (1 - tau) * sum(max(-r, 0))` of its own residuals

#### Scenario: The dual certifies the fit

- **WHEN** the interior-point solver returns a fit at level `tau`
- **THEN** its `dual_solution` `a` satisfies `X' a = 0` with every entry inside
  `[tau - 1, tau]`, which is exactly the optimality condition for that level
