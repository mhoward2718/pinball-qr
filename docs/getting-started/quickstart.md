# Quick Start

## Fitting Your First Quantile Regression

```python
import numpy as np
from pinball import QuantileRegressor

# Generate some data
rng = np.random.default_rng(42)
n = 500
X = rng.uniform(-2, 2, (n, 1))
y = 2 * X.ravel() + 1 + rng.standard_t(df=3, size=n)

# Fit the median (tau=0.5)
model = QuantileRegressor(tau=0.5, method="fn")
model.fit(X, y)
print(f"Intercept: {model.intercept_:.3f}")
print(f"Slope:     {model.coef_[0]:.3f}")
```

## Multiple Quantiles at Once

Pinball can fit several quantile levels in a single call:

```python
model = QuantileRegressor(tau=[0.1, 0.25, 0.5, 0.75, 0.9])
model.fit(X, y)

# coef_ has shape (n_features, n_quantiles)
for i, tau in enumerate([0.1, 0.25, 0.5, 0.75, 0.9]):
    print(f"  tau={tau:.2f}  intercept={model.intercept_[i]:.3f}  "
          f"slope={model.coef_[0, i]:.3f}")
```

## Inference: Standard Errors and Confidence Intervals

```python
from pinball import summary

result = summary(model, X, y, se_method="nid")
print(result)
```

## Bootstrap Confidence Intervals

```python
from pinball import bootstrap

boot = bootstrap(model, X, y, nboot=500, method="xy")
print(boot)
```

## Choosing a Solver

| Method | Solver | Best for | Complexity |
|--------|--------|----------|------------|
| `"br"` | Barrodale-Roberts simplex | Small data (n < 5,000), exact CI | \\(O(n \cdot p^2)\\) |
| `"fn"` | Frisch-Newton interior point | Medium data (default) | \\(O(n \cdot p^{1.5})\\) |
| `"pfn"` | Preprocessing + Frisch-Newton | Large data (n > 50,000) | \\(\tilde{O}(\sqrt{p} \cdot n^{2/3})\\) |
| `"lasso"` | L1-penalised interior point | High-dimensional sparse | \\(O(n \cdot p^{1.5})\\) |

```python
# For a dataset with 1 million rows
model = QuantileRegressor(tau=0.5, method="pfn")
model.fit(X_large, y_large)
```

## Nonparametric Estimation

When the conditional quantile function is non-linear and you don't want to
specify a parametric form:

```python
from pinball.nonparametric.quantization import QuantizationQuantileEstimator

est = QuantizationQuantileEstimator(tau=0.5, N=20, n_grids=50, random_state=42)
est.fit(X, y)
y_pred = est.predict(X_new)
```

## Using with scikit-learn Pipelines

Both estimators are fully sklearn-compatible:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("qr", QuantileRegressor(tau=0.5, method="fn")),
])
pipe.fit(X, y)
```
