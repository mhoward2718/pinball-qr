# Solvers

The solver layer converts the quantile regression optimisation problem
into a solution using one of several algorithms.  All solvers share a
common interface (`BaseSolver`) and are accessed through a registry.

## Registry Functions

```python
from pinball.solvers import get_solver, list_solvers, register_solver
```

### `get_solver(name, **kwargs)`

Instantiate a solver by name.

```python
solver = get_solver("fn")          # FNBSolver with defaults
solver = get_solver("br")          # BRSolver
solver = get_solver("lasso", lambda_=0.1)
```

### `list_solvers()`

Return a list of registered solver names.

```python
>>> list_solvers()
['br', 'fn', 'fnb', 'lasso', 'pfn']
```

### `register_solver(name, cls)`

Register a custom solver class.

```python
register_solver("my_solver", MySolverClass)
```

---

## BaseSolver

```python
from pinball.solvers import BaseSolver
```

Abstract base class for all solvers.

### Interface

| Method | Description |
|--------|-------------|
| `solve(X, y, tau, **kwargs)` | Validate inputs, solve, return `SolverResult` |
| `validate_inputs(X, y, tau)` | Input validation hook (override in subclasses) |
| `_solve_impl(X, y, tau, **kwargs)` | *Abstract* — the actual solve logic |

### `solve(X, y, tau, **kwargs) → SolverResult`

The public entry point.  Calls `validate_inputs()` followed by `_solve_impl()`.

---

## SolverResult

```python
from pinball.solvers import SolverResult
```

A dataclass returned by every solver.

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `ndarray (p,)` | Estimated coefficient vector |
| `residuals` | `ndarray (n,)` or `None` | Residuals \\(y - X\beta\\) |
| `dual_solution` | `ndarray (n,)` or `None` | Dual variables (BR solver only) |
| `objective_value` | `float` or `None` | Optimal pinball loss |
| `status` | `int` | Solver exit status (0 = success) |
| `iterations` | `int` | Number of iterations |
| `solver_info` | `dict` | Additional solver-specific output |

---

## BRSolver

```python
from pinball.linear.solvers.br import BRSolver
```

Barrodale-Roberts simplex solver.  Wraps the Fortran `rqbr` subroutine.

### Constructor

```python
BRSolver()  # no parameters
```

### Solver-specific options (via `**kwargs` in `solve()`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ci` | `bool` | `False` | Compute rank-inversion confidence intervals |
| `alpha` | `float` | `0.05` | Significance level for CI |
| `iid` | `bool` | `True` | IID assumption for bandwidth in CI |

### Solver info keys

When `ci=True`, `solver_info` contains:

- `"ci"` — confidence interval array, shape `(p, 2)`
- `"ci_dual"` — dual solutions at CI bounds

---

## FNBSolver

```python
from pinball.linear.solvers.fnb import FNBSolver
```

Frisch-Newton interior-point solver (bounded variables formulation).

### Constructor

```python
FNBSolver(
    beta=0.99995,  # step-size damping in (0, 1)
    eps=1e-6,      # convergence tolerance
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `0.99995` | Step-size damping. Larger = more aggressive. |
| `eps` | `float` | `1e-6` | Convergence tolerance. \\(\tau\\) must be in \\((\varepsilon, 1-\varepsilon)\\). |

---

## PreprocessingSolver

```python
from pinball.linear.solvers.pfn import PreprocessingSolver
```

Preprocessing + interior-point solver for large-\\(n\\) problems.
See [Preprocessing Theory](../theory/preprocessing.md) for details.

### Constructor

```python
PreprocessingSolver(
    inner_solver=None,     # BaseSolver or None (defaults to FNBSolver)
    mm_factor=0.8,         # middle-band fraction
    max_bad_fixups=3,      # fixup budget
    eps=1e-6,              # bandwidth floor
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inner_solver` | `BaseSolver` or `None` | `FNBSolver()` | Solver for the reduced subproblem |
| `mm_factor` | `float` | `0.8` | Fraction of subsample kept in the middle band |
| `max_bad_fixups` | `int` | `3` | Fixup iterations before doubling \\(m\\) |
| `eps` | `float` | `1e-6` | Floor for leverage-adjusted bandwidth |

---

## LassoSolver

```python
from pinball.linear.solvers.lasso import LassoSolver
```

\\(\ell_1\\)-penalised quantile regression via augmented LP.

### Constructor

```python
LassoSolver(
    lambda_=None,             # penalty (None = auto)
    penalize_intercept=False, # penalise first column?
    beta=0.99995,             # FNB damping
    eps=1e-6,                 # FNB tolerance
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lambda_` | `float` or `None` | `None` | Penalty parameter. `None` uses the Belloni-Chernozhukov default. |
| `penalize_intercept` | `bool` | `False` | Whether the intercept column is penalised |
| `beta` | `float` | `0.99995` | Interior-point damping |
| `eps` | `float` | `1e-6` | Convergence tolerance |

### Automatic penalty

When `lambda_=None`:

\[
\hat\lambda = 2 \, \Phi^{-1}\!\left(1 - \frac{0.05}{2p}\right)
\sqrt{\frac{\tau(1-\tau)}{n}}
\]

---

## Direct Solver Usage

All solvers can be used independently of the `QuantileRegressor` estimator:

```python
import numpy as np
from pinball.solvers import get_solver

X = np.column_stack([np.ones(100), np.random.randn(100, 2)])
y = np.random.randn(100)

solver = get_solver("fn")
result = solver.solve(X, y, tau=0.5)

print(result.coefficients)
print(result.objective_value)
print(result.iterations)
```

!!! warning
    When using solvers directly, you must add the intercept column
    yourself if needed.  The `QuantileRegressor` adds it automatically
    when `fit_intercept=True`.
