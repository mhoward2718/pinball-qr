# Linear Solvers

Pinball ships four linear quantile-regression solvers.  Each targets
a different regime of problem size and structure.  They all solve the
same linear program but differ in computational complexity, auxiliary
output, and numerical behaviour.

## Solver Selection Guide

| Solver | Method key | Best for | Complexity | CI built-in? |
|--------|-----------|----------|------------|--------------|
| Barrodale-Roberts | `"br"` | \\(n \lesssim 5\,000\\) | \\(O(n \cdot p^2)\\) worst case | **Yes** (rank-inversion) |
| Frisch-Newton | `"fn"` / `"fnb"` | \\(5\,000 < n < 100\,000\\) | \\(O(n \cdot p^{1.5})\\) typical | No |
| Preprocessing + FN | `"pfn"` | \\(n > 100\,000\\) | \\(O(\sqrt{p} \cdot n^{2/3} \cdot p^{1.5})\\) | No |
| Lasso | `"lasso"` | Sparse / high-dimensional | same as FN on augmented problem | No |

```python
from pinball import QuantileRegressor

# Choose the solver via the `method` parameter
model = QuantileRegressor(tau=0.5, method="fn")
model.fit(X, y)
```

---

## Barrodale-Roberts (`"br"`)

The simplex-based solver of Barrodale & Roberts (1974), as specialised
for quantile regression by Koenker & d'Orey (1987, 1994).  This is the
workhorse of small-sample quantile regression and remains the default
in R's `quantreg::rq()`.

### How it works

The check-function minimisation is reformulated as a linear program.
The Barrodale-Roberts algorithm is a modified simplex method that
exploits the special structure of the constraint matrix (every row of
\\(X\\) enters at most one basis) to achieve very efficient pivoting.

Each simplex iteration corresponds to an **interpolation**: the
quantile hyperplane passes through \\(p\\) data points.  The algorithm
moves from one set of interpolation points to the next, always
decreasing the objective.

### Unique features

- **Dual solution** — the solver returns the full dual vector \\(d_i\\),
  which classifies every observation as above, below, or on the
  hyperplane.  This is needed for rank-inversion confidence intervals.

- **Rank-inversion CI** — when called with `ci=True`, the solver
  computes Koenker (1994) confidence intervals by inverting a rank
  test.  These intervals are exact (up to discretisation) and do not
  require density estimation.

- **Full quantile process** — with `tau=None`, the solver computes
  the solution path for *all* quantile levels where the solution
  changes, which is a piecewise-linear function of \\(\tau\\).

### When to use

- Small datasets (\\(n \lesssim 5\,000\\)); exact CI are needed; or you
  want the full quantile process.

### Example

```python
from pinball import QuantileRegressor

model = QuantileRegressor(tau=0.5, method="br",
                          solver_options={"ci": True, "alpha": 0.05})
model.fit(X, y)
# solver_result_ contains dual solution and CI
ci = model.solver_result_.solver_info["ci"]
```

### References

1. Barrodale, I. and Roberts, F.D.K. (1974). "Solution of an
   overdetermined system of equations in the \\(\ell_1\\) norm."
2. Koenker, R. and d'Orey, V. (1987, 1994). "Computing regression
   quantiles." *Applied Statistics*.
3. Koenker, R. (1994). "Confidence intervals for regression quantiles."

---

## Frisch-Newton Interior Point (`"fn"` / `"fnb"`)

An interior-point (barrier) method that solves the quantile regression
LP by following a central path through the interior of the feasible
region.  This is the algorithm described in Portnoy & Koenker (1997)
as the "Laplacian tortoise" — slower per iteration than simplex, but
with far better scaling.

### How it works

The primal-dual LP is solved by Newton steps on the KKT system with a
log-barrier term.  Each iteration solves a \\(p \times p\\) linear
system (not \\(n \times n\\)), so the cost per iteration is
\\(O(n p^2)\\) for a dense Cholesky factorisation.  Convergence is
typically achieved in \\(O(\sqrt{n})\\) iterations, giving an overall
complexity of \\(O(n^{3/2} p^2)\\) — much better than simplex for
large \\(n\\).

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 0.99995 | Step-size damping; larger values take more aggressive steps |
| `eps` | 1e-6 | Convergence tolerance; \\(\tau\\) must be in \\((\varepsilon, 1-\varepsilon)\\) |

### When to use

- Medium-to-large datasets (\\(5\,000 < n < 100\,000\\))
- When CI can be computed separately (via `summary()` or `bootstrap()`)
- As the default for most practical work

### Example

```python
from pinball import QuantileRegressor

model = QuantileRegressor(tau=0.5, method="fn")
model.fit(X, y)
```

### References

1. Portnoy, S. and Koenker, R. (1997). "The Gaussian hare and the
   Laplacian tortoise." *Statistical Science* 12(4): 279–300.

---

## Preprocessing + Frisch-Newton (`"pfn"`)

The key to scaling quantile regression to **really large** datasets.
This solver wraps the Frisch-Newton method with the Portnoy-Koenker
preprocessing strategy, reducing the effective problem size from \\(n\\)
to \\(O(\sqrt{p} \cdot n^{2/3})\\).

???+ tip "The key insight"
    Most observations are far from the quantile hyperplane and can be
    **aggregated into two summary points** (one for observations above,
    one for below) without changing the optimal solution.  Only the
    "middle band" of observations near the hyperplane participates in
    each LP iteration.

This is the solver for datasets with \\(n > 100{,}000\\) — or even
millions of rows.  See the dedicated
[preprocessing section](preprocessing.md) for the full theory.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inner_solver` | `FNBSolver()` | Any `BaseSolver` used on the reduced subproblem |
| `mm_factor` | 0.8 | Fraction of subsample \\(m\\) kept in the "middle band" |
| `max_bad_fixups` | 3 | Fixup iterations before doubling \\(m\\) |
| `eps` | 1e-6 | Bandwidth floor for detecting extreme residuals |

### When to use

- Large datasets (\\(n > 100{,}000\\))
- Any problem where FNB is too slow

### Example

```python
from pinball import QuantileRegressor

model = QuantileRegressor(tau=0.5, method="pfn")
model.fit(X_large, y_large)
```

### References

1. Portnoy, S. and Koenker, R. (1997). "The Gaussian hare and the
   Laplacian tortoise." *Statistical Science* 12(4): 279–300.

---

## \\(\ell_1\\)-Penalised Lasso (`"lasso"`)

Sparse quantile regression via \\(\ell_1\\) penalisation.  Solves:

\[
\hat\beta(\tau) = \arg\min_{\beta}
\sum_{i=1}^{n} \rho_\tau(y_i - x_i^\top \beta)
+ \lambda \sum_{j=1}^{p} |\beta_j|
\]

The penalised problem is converted to an augmented LP and solved by
the Frisch-Newton interior-point method.

### How it works

The penalty is incorporated by **augmenting the design matrix**:

\[
X_{\text{aug}} = \begin{bmatrix} X \\ \text{diag}(\lambda) \end{bmatrix},
\quad
y_{\text{aug}} = \begin{bmatrix} y \\ 0 \end{bmatrix}
\]

This converts the penalised problem into an unpenalised quantile
regression on a larger dataset, which is then solved by FNB.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_` | `None` (auto) | Penalty parameter; if `None`, Belloni-Chernozhukov default is used |
| `penalize_intercept` | `False` | Whether the intercept column is penalised |

### Automatic penalty selection

When `lambda_=None`, the solver uses the Belloni & Chernozhukov (2011)
pivot-based default:

\[
\hat\lambda = 2 \cdot \Phi^{-1}\!\left(1 - \frac{\alpha}{2p}\right)
\cdot \sqrt{\frac{\tau(1-\tau)}{n}}
\]

### When to use

- High-dimensional settings where \\(p\\) is large relative to \\(n\\)
- When many coefficients are expected to be zero (sparsity)
- Variable-selection applications

### Example

```python
from pinball import QuantileRegressor

# Automatic lambda
model = QuantileRegressor(tau=0.5, method="lasso")
model.fit(X_sparse, y)

# Custom lambda
model = QuantileRegressor(tau=0.5, method="lasso",
                          solver_options={"lambda_": 0.1})
model.fit(X_sparse, y)
```

### References

1. Belloni, A. and Chernozhukov, V. (2011). "\\(\ell_1\\)-penalized quantile
   regression in high-dimensional sparse models." *Annals of Statistics*.

---

## Solver Registry

All solvers are registered in a global dictionary and can be listed
programmatically:

```python
from pinball.linear.solvers import list_solvers, get_solver

# See available methods
print(list_solvers())
# ['br', 'fn', 'fnb', 'lasso', 'pfn']

# Instantiate directly
solver = get_solver("fn")
result = solver.solve(X, y, tau=0.5)
```

Custom solvers can be registered with `register_solver()`:

```python
from pinball.linear.solvers import register_solver, BaseSolver

class MySolver(BaseSolver):
    def _solve_impl(self, X, y, tau, **kwargs):
        ...

register_solver("my_method", MySolver)
```
