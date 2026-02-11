# Inference

Post-estimation inference for linear quantile regression:
standard errors, confidence intervals, and bootstrap.

## Summary / Standard Errors

```python
from pinball import QuantileRegressor, summary
```

### `summary(model, X, y, se="nid", alpha=0.05)`

Compute standard errors, t-statistics, p-values, and confidence
intervals for a fitted `QuantileRegressor`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `QuantileRegressor` | — | A fitted model |
| `X` | array-like, (n, p) | — | Training data (same as used in `fit`) |
| `y` | array-like, (n,) | — | Training targets |
| `se` | `str` | `"nid"` | Standard error method (see below) |
| `alpha` | `float` | `0.05` | Significance level |

Returns an `InferenceResult`.

### SE Methods

| Method | Name | Description | Default for |
|--------|------|-------------|-------------|
| `"rank"` | Rank inversion | Koenker (1994) — exact CI based on dual | BR solver only |
| `"iid"` | IID sandwich | Koenker-Bassett (1978) — assumes i.i.d. errors | — |
| `"nid"` | Huber sandwich | Local sparsity / Hendricks-Koenker bandwidth | \\(n \geq 1001\\) |
| `"ker"` | Kernel sandwich | Powell (1991) kernel estimate of the density | — |
| `"boot"` | Bootstrap | Delegates to `bootstrap()` | \\(n < 1001\\) |

### Example

```python
from pinball import QuantileRegressor, summary

model = QuantileRegressor(tau=0.5, method="fn")
model.fit(X, y)

result = summary(model, X, y, se="nid")
print(result)
# InferenceResult(se_method='nid')
#                Coef    Std Err          t      P>|t|     [0.025     0.975]
#    intercept   81.48     14.63       5.57     0.0000      52.80    110.16
#       income    0.56      0.01      38.41     0.0000       0.53      0.59
```

---

## InferenceResult

```python
from pinball.linear._inference import InferenceResult
```

A dataclass holding the summary table.

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients` | `ndarray (p,)` | Point estimates |
| `std_errors` | `ndarray (p,)` | Standard errors |
| `t_statistics` | `ndarray (p,)` | \\(t = \text{coef} / \text{se}\\) |
| `p_values` | `ndarray (p,)` | Two-sided p-values |
| `conf_int` | `ndarray (p, 2)` | Confidence interval bounds |
| `se_method` | `str` | Which SE method was used |
| `feature_names` | `list[str]` or `None` | Optional column names |

The `__repr__` method prints a formatted table similar to R's
`summary.rq()` output.

---

## Bootstrap

```python
from pinball import bootstrap
```

### `bootstrap(model, X, y, method="xy-pair", nboot=200, alpha=0.05, random_state=None)`

Bootstrap inference using one of three strategies.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `QuantileRegressor` | — | A fitted model |
| `X` | array-like, (n, p) | — | Training data |
| `y` | array-like, (n,) | — | Training targets |
| `method` | `str` | `"xy-pair"` | Bootstrap method (see below) |
| `nboot` | `int` | `200` | Number of bootstrap replicates |
| `alpha` | `float` | `0.05` | Significance level |
| `random_state` | `int` or `None` | `None` | Seed for reproducibility |

Returns a `BootstrapResult`.

### Bootstrap Methods

| Method | Name | Description |
|--------|------|-------------|
| `"xy-pair"` | XY-pair | Classical nonparametric resampling of \\((x_i, y_i)\\) pairs (Efron, 1979) |
| `"wild"` | Wild bootstrap | Perturbation-based bootstrap (Feng, He & Hu, 2011) — better for heteroscedastic errors |
| `"mcmb"` | MCMB | Markov chain marginal bootstrap (He & Hu, 2002) — fastest, best for moderate \\(n\\) |

### Example

```python
from pinball import QuantileRegressor, bootstrap

model = QuantileRegressor(tau=0.5)
model.fit(X, y)

result = bootstrap(model, X, y, method="xy-pair", nboot=500, random_state=42)
print(result.std_errors)
print(result.conf_int)
```

---

## BootstrapResult

```python
from pinball.linear._bootstrap import BootstrapResult
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `boot_coefficients` | `ndarray (nboot, p)` | All bootstrap replicate coefficients |
| `coefficients` | `ndarray (p,)` | Point estimate (original fit or mean of replicates) |
| `std_errors` | `ndarray (p,)` | Column-wise std of `boot_coefficients` |
| `conf_int` | `ndarray (p, 2)` | Percentile confidence intervals |
| `bsmethod` | `str` | Bootstrap method used |
| `nboot` | `int` | Number of replicates |

### Properties

| Property | Description |
|----------|-------------|
| `boot_coefficients` | The full R × p matrix of replicate draws — useful for plotting bootstrap distributions |

### Example: Visualising Bootstrap Distribution

```python
import matplotlib.pyplot as plt

result = bootstrap(model, X, y, nboot=1000, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for j, ax in enumerate(axes):
    ax.hist(result.boot_coefficients[:, j], bins=30, alpha=0.7)
    ax.axvline(result.coefficients[j], color='red', linestyle='--')
    ax.set_title(f"β_{j}")
plt.tight_layout()
plt.show()
```

---

## References

1. Koenker, R. and Bassett, G. (1978). "Regression quantiles."
   *Econometrica* 46(1): 33–50.
2. Koenker, R. (1994). "Confidence intervals for regression quantiles."
3. Powell, J.L. (1991). "Estimation of monotonic regression models
   under quantile restrictions."
4. Efron, B. (1979). "Bootstrap methods: another look at the jackknife."
5. Feng, X., He, X. and Hu, J. (2011). "Wild bootstrap for quantile
   regression." *Biometrika* 98(4): 995–999.
6. He, X. and Hu, F. (2002). "Markov chain marginal bootstrap."
   *JASA* 97(459): 783–795.
