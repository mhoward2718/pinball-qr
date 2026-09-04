## Purpose

Provide the interchangeable linear-programming solver back-ends that actually
compute quantile regression fits, behind a single uniform interface, so that
solvers can be added or swapped without changing estimator code.

## Requirements

### Requirement: Uniform solver interface

Every solver SHALL subclass `BaseSolver` and implement
`solve(X, y, tau, **kwargs)` returning a `SolverResult` carrying
`coefficients`, `residuals`, `objective_value`, `status`, `iterations` and
`solver_info`.

#### Scenario: Any registered solver satisfies the contract

- **WHEN** any solver returned by `get_solver(name)` is asked to solve a
  well-conditioned problem
- **THEN** it returns a `SolverResult` whose `coefficients` has length `p` and
  whose `residuals` has length `n`

#### Scenario: Objective value is the pinball loss

- **WHEN** a solver returns a `SolverResult` for level `tau`
- **THEN** `objective_value` equals the pinball loss
  `tau * sum(max(r, 0)) + (1 - tau) * sum(max(-r, 0))` of its own residuals

### Requirement: Solver registry is open for extension

The registry SHALL allow new solvers to be added via `register_solver(name, cls)`
without modifying existing solver or estimator code, SHALL reject classes that
are not `BaseSolver` subclasses, and SHALL expose the registered names through
`list_solvers()`.

#### Scenario: Registering a new solver

- **WHEN** a user registers a valid `BaseSolver` subclass under a new name
- **THEN** `get_solver(name)` returns an instance of it and the name appears in
  `list_solvers()`

#### Scenario: Registering a non-solver

- **WHEN** `register_solver` is called with a class that does not subclass
  `BaseSolver`
- **THEN** a `TypeError` is raised and the registry is unchanged

### Requirement: Solvers agree with one another

All exact solvers SHALL produce the same fit, to numerical tolerance, on
problems with a unique solution — differences between solvers are performance
and applicability, not answers.

#### Scenario: Simplex and interior point agree at the median

- **WHEN** the same well-conditioned problem is solved with `method="br"` and
  `method="fn"` at `tau=0.5`
- **THEN** the two coefficient vectors agree to within solver tolerance

#### Scenario: Solvers agree at an extreme quantile

- **WHEN** the same problem is solved with `method="br"` and `method="fn"` at
  `tau=0.99`
- **THEN** the two coefficient vectors agree to within solver tolerance

### Requirement: Solvers report rather than hide numerical failure

A solver SHALL NOT silently return an untrustworthy fit. When the underlying
routine reports a non-success status, fails to reach its requested tolerance,
or exhausts its iteration budget, the solver SHALL surface that condition to
the caller via a warning and via `SolverResult.status`.

#### Scenario: Underlying routine reports a failure status

- **WHEN** a solver's native routine returns a non-zero/non-success status
  (for example a singular design)
- **THEN** a warning is raised and the non-success status is recorded in
  `SolverResult.status`

#### Scenario: Iteration budget exhausted before convergence

- **WHEN** an iterative solver reaches its maximum iteration count without
  meeting its convergence tolerance
- **THEN** the caller is warned that the result did not converge, rather than
  receiving an apparently normal result

### Requirement: Input validation

Solvers SHALL validate their inputs and raise informative errors rather than
producing undefined results.

#### Scenario: Mismatched shapes

- **WHEN** `X` has `n` rows but `y` has a different length
- **THEN** a `ValueError` describing the mismatch is raised

#### Scenario: Solver-specific domain limits

- **WHEN** a solver is called at a `tau` outside the range it supports
- **THEN** a `ValueError` naming the supported range is raised
