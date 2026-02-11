# Pinball

**Fast, accurate quantile regression for Python.**

<p align="center">
  <img src="assets/pinball_logo.png" alt="Pinball logo" width="200"/>
</p>

[![PyPI](https://img.shields.io/pypi/v/pinball.svg)](https://pypi.org/project/pinball/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![sklearn](https://img.shields.io/badge/sklearn-compatible-blue.svg)](https://scikit-learn.org/)

---

Pinball brings the speed and statistical rigor of R's
[quantreg](https://cran.r-project.org/package=quantreg) package to Python,
wrapped in a familiar scikit-learn interface.  It provides:

- **High-performance Fortran solvers** — the Barrodale-Roberts simplex and
  Frisch-Newton interior-point algorithms, compiled from the original
  quantreg Fortran code
- **Preprocessing for massive datasets** — the Portnoy-Koenker "globs"
  technique that reduces million-row problems to manageable size
- **Full statistical inference** — standard errors, confidence intervals,
  and three bootstrap methods (xy-pair, wild, MCMB)
- **L1-penalised (lasso) quantile regression** — for high-dimensional
  sparse models using the Belloni-Chernozhukov approach
- **Nonparametric conditional quantiles** — the Charlier-Paindaveine-Saracco
  quantization-based estimator for when linearity doesn't hold
- **100% scikit-learn compatible** — passes `check_estimator` for both
  linear and nonparametric estimators

## Quick Example

```python
from pinball import QuantileRegressor, load_engel

X, y = load_engel(return_X_y=True)
model = QuantileRegressor(tau=[0.1, 0.25, 0.5, 0.75, 0.9], method="fn")
model.fit(X, y)
print(model.coef_)
```

## Why "Pinball"?

The **pinball loss** (also called the check function or asymmetric absolute
loss) is the objective function at the heart of quantile regression:

\[
\rho_\tau(u) = u \cdot (\tau - \mathbf{1}_{u < 0})
\]

It penalises underestimates by \\(\tau\\) and overestimates by \\(1 - \tau\\),
making it the natural loss for estimating the \\(\tau\\)-th conditional
quantile.

## Project Lineage

Pinball is a faithful port of two R packages:

| Component | R Source | Reference |
|-----------|----------|-----------|
| Linear solvers & inference | [quantreg](https://cran.r-project.org/package=quantreg) | Koenker (2005), *Quantile Regression* |
| Preprocessing | quantreg (`rq.fit.pfn`) | Portnoy & Koenker (1997), *Statistical Science* |
| Nonparametric estimator | [QuantifQuantile](https://cran.r-project.org/package=QuantifQuantile) | Charlier, Paindaveine & Saracco (2015), *JSPI* |
