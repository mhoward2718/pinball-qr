# Estimators

## BaseQuantileEstimator

::: pinball.estimators.BaseQuantileEstimator

Abstract base class for all conditional quantile estimators in pinball.
Inherits from `sklearn.base.RegressorMixin` and `sklearn.base.BaseEstimator`.

### Interface

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Fit the model. *Abstract — must be implemented by subclasses.* |
| `predict(X)` | Predict conditional quantiles. *Abstract.* |
| `score(X, y)` | R² score (inherited from `RegressorMixin`) |
| `pinball_loss(X, y)` | Mean pinball (check) loss: \\(\frac{1}{n}\sum_i \rho_\tau(y_i - \hat y_i)\\) |

---

## QuantileRegressor

```python
from pinball import QuantileRegressor
```

The primary estimator for **linear quantile regression**.  Wraps the
high-performance Fortran solvers behind a familiar sklearn interface.

### Constructor

```python
QuantileRegressor(
    tau=0.5,              # float or list of float
    method="fn",          # "br", "fn"/"fnb", "pfn", "lasso"
    fit_intercept=True,   # bool
    solver_options=None,  # dict or None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | `float` or `list[float]` | `0.5` | Quantile level(s) in (0, 1). A list fits all quantiles simultaneously. |
| `method` | `str` | `"fn"` | Solver backend. See [Linear Solvers](../theory/linear-methods.md). |
| `fit_intercept` | `bool` | `True` | Whether to add an intercept column. |
| `solver_options` | `dict` or `None` | `None` | Extra arguments forwarded to the solver (e.g. `{"ci": True}` for BR). |

### Attributes (after fitting)

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `coef_` | `(n_features,)` or `(n_features, n_quantiles)` | Estimated coefficients (excluding intercept) |
| `intercept_` | `float` or `ndarray` | Intercept term(s). Zero when `fit_intercept=False`. |
| `residuals_` | `(n_samples,)` or `(n_samples, n_quantiles)` | Residuals from the fit |
| `solver_result_` | `SolverResult` or `list[SolverResult]` | Full solver output for advanced use |
| `n_features_in_` | `int` | Number of features seen during `fit` |
| `n_iter_` | `int` or `list[int]` | Solver iteration count |

### Methods

#### `fit(X, y, sample_weight=None)`

Fit the quantile regression model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like, (n, p) | Training data |
| `y` | array-like, (n,) | Target values |
| `sample_weight` | array-like, (n,) or None | Sample weights. Integer weights expand rows; float weights scale rows. |

Returns `self`.

#### `predict(X)`

Predict quantile(s) for new data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like, (n_new, p) | New data |

Returns `ndarray` of shape `(n_new,)` or `(n_new, n_quantiles)`.

#### `pinball_loss(X, y)`

Mean pinball loss on the given data.

#### `score(X, y)`

R² score (inherited from `RegressorMixin`).

### Example

```python
import numpy as np
from pinball import QuantileRegressor

rng = np.random.default_rng(42)
X = rng.normal(size=(500, 3))
y = X @ [1, 2, 3] + rng.normal(size=500)

# Single quantile
model = QuantileRegressor(tau=0.5, method="fn")
model.fit(X, y)
print(model.coef_)       # ≈ [1, 2, 3]
print(model.intercept_)  # ≈ 0

# Multiple quantiles
model = QuantileRegressor(tau=[0.1, 0.5, 0.9])
model.fit(X, y)
print(model.coef_.shape)  # (3, 3) — one column per quantile
```

---

## QuantizationQuantileEstimator

```python
from pinball.nonparametric import QuantizationQuantileEstimator
```

Nonparametric conditional quantile estimator via optimal quantization
(Charlier, Paindaveine & Saracco, 2015).

### Constructor

```python
QuantizationQuantileEstimator(
    tau=0.5,          # float
    N=20,             # int — number of grid points
    n_grids=50,       # int — bootstrap grids
    p=2,              # float — L_p norm
    random_state=None # int or None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | `float` | `0.5` | Quantile level in (0, 1) |
| `N` | `int` | `20` | Number of points in the quantization grid |
| `n_grids` | `int` | `50` | Number of independent bootstrap grids (higher = more stable) |
| `p` | `float` | `2` | L_p norm exponent for CLVQ. `p=2` is Euclidean. |
| `random_state` | `int` or `None` | `None` | Random seed for reproducibility |

### Attributes (after fitting)

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `grid_` | `(N,)` or `(d, N)` | Averaged optimal quantization grid |
| `cell_quantiles_` | `(N, 1)` | Averaged conditional quantile per Voronoi cell |
| `n_features_in_` | `int` | Number of features seen during `fit` |

### Methods

#### `fit(X, y)`

Fit via CLVQ grid construction + Voronoi cell quantiles.

!!! note
    This estimator does not accept `sample_weight` — the CLVQ
    algorithm's internal bootstrap depends on \\(n\\) in ways
    that make weighted and unweighted paths non-equivalent.

#### `predict(X)`

Assign each row of `X` to its nearest Voronoi cell and return the
cell's conditional quantile estimate.

### Example

```python
import numpy as np
from pinball.nonparametric import QuantizationQuantileEstimator

rng = np.random.default_rng(42)
X = rng.uniform(0, 10, (500, 1))
y = np.sin(X.ravel()) + 0.3 * rng.normal(size=500)

model = QuantizationQuantileEstimator(tau=0.5, N=20, random_state=42)
model.fit(X, y)
y_hat = model.predict(X)
```
