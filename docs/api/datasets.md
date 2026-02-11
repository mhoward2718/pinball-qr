# Datasets

Pinball bundles two classic datasets used throughout the quantile
regression literature.  Both are returned as `sklearn.utils.Bunch`
objects for compatibility with sklearn workflows.

---

## load_engel

```python
from pinball.datasets import load_engel
```

### Description

Ernst Engel's 1857 food expenditure data.  235 Belgian working-class
households with a single predictor (income) predicting food
expenditure.  This is the canonical dataset for illustrating quantile
regression — it appears in virtually every tutorial and textbook.

### Returns

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `data` | `ndarray` | `(235, 1)` | Household income |
| `target` | `ndarray` | `(235,)` | Food expenditure |
| `feature_names` | `list[str]` | — | `["income"]` |
| `DESCR` | `str` | — | Human-readable description |

### Example

```python
from pinball.datasets import load_engel
from pinball import QuantileRegressor

engel = load_engel()
X, y = engel.data, engel.target

model = QuantileRegressor(tau=[0.1, 0.25, 0.5 , 0.75, 0.9])
model.fit(X, y)

# The slopes diverge: wealthier households show more variability
# in food spending — a textbook example of heteroscedasticity.
print(model.coef_)
```

### Why this dataset?

Engel's law states that the proportion of income spent on food
decreases as income rises.  Quantile regression reveals an additional
insight: the *variability* of food spending also increases with
income.  The slope at the 90th percentile is much steeper than at the
10th — richer households have more diverse food budgets.

---

## load_barro

```python
from pinball.datasets import load_barro
```

### Description

Cross-country economic growth data from Barro (1991) and Barro & Lee
(1994).  161 countries with 13 predictor variables and net GDP growth
as the target.  This multi-predictor dataset is useful for
demonstrating quantile regression with several covariates.

### Returns

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `data` | `ndarray` | `(161, 13)` | Predictor variables |
| `target` | `ndarray` | `(161,)` | Net GDP growth rate |
| `feature_names` | `list[str]` | — | See table below |
| `DESCR` | `str` | — | Human-readable description |

### Feature Names

| Column | Name | Description |
|--------|------|-------------|
| 0 | `lgdp2` | Log initial GDP per capita |
| 1 | `mse2` | Male secondary school enrolment |
| 2 | `fse2` | Female secondary school enrolment |
| 3 | `fhe2` | Female higher education |
| 4 | `mhe2` | Male higher education |
| 5 | `lexp2` | Life expectancy |
| 6 | `lintr2` | Log investment rate |
| 7 | `gedy2` | Government education / GDP |
| 8 | `Iy2` | Investment / GDP |
| 9 | `gcony2` | Government consumption / GDP |
| 10 | `lblakp2` | Log black-market premium |
| 11 | `pol2` | Political instability |
| 12 | `ttrad2` | Terms of trade |

### Example

```python
from pinball.datasets import load_barro
from pinball import QuantileRegressor

barro = load_barro()
X, y = barro.data, barro.target

# Median regression
model = QuantileRegressor(tau=0.5, method="fn")
model.fit(X, y)

# Which predictors matter at the lower tail?
model_low = QuantileRegressor(tau=0.1)
model_low.fit(X, y)
print(dict(zip(barro.feature_names, model_low.coef_)))
```

### References

1. Barro, R. (1991). "Economic growth in a cross section of countries."
   *Quarterly Journal of Economics* 106(2): 407–443.
2. Barro, R. and Lee, J.-W. (1994). "Sources of economic growth."
   *Carnegie-Rochester Conference Series on Public Policy* 40: 1–46.
