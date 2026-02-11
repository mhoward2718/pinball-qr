# The Preprocessing Approach: Gaussian Hare and Laplacian Tortoise

!!! abstract "Key idea"
    Most observations in a quantile regression problem are far from the
    fitted hyperplane and can be **replaced by two summary statistics**
    (one above, one below) without changing the optimal solution.  Only
    a thin "middle band" of observations near the quantile hyperplane
    needs to participate in each iteration.  This reduces the effective
    problem size from \\(n\\) to \\(O(\sqrt{p} \cdot n^{2/3})\\),
    yielding **10- to 100-fold speedups** on large datasets.

---

## Motivation

Least squares (\\(\ell_2\\)) regression is famously fast: the normal
equations \\(X^\top X \beta = X^\top y\\) take \\(O(n p^2)\\) — linear
in \\(n\\).  Quantile regression (\\(\ell_1\\)) requires solving a
linear program, whose simplex-based algorithms have superlinear cost
in \\(n\\) and whose interior-point algorithms, while better, still
touch every observation at every iteration.

Portnoy & Koenker (1997) posed the question provocatively:

> *Must the \\(\ell_1\\) tortoise always be slower than the
> \\(\ell_2\\) hare?*

Their answer is **no** — with a simple preprocessing trick, the
\\(\ell_1\\) tortoise can actually be *faster* than OLS for large
\\(n\\), because the effective number of observations shrinks to
\\(O(n^{2/3})\\).  The Gaussian hare must always process all \\(n\\)
observations; the Laplacian tortoise need not.

---

## The Algorithm

### Step 1: Initial Subsample

Draw a random subsample of size

\[
m = \lceil \sqrt{p} \cdot n^{2/3} \rceil
\]

and solve the subsampled quantile regression:

\[
\hat\beta^{(0)} = \arg\min_{\beta} \sum_{i \in S_m} \rho_\tau(y_i - x_i^\top \beta)
\]

This pilot estimate is not accurate but is fast to compute — the LP
has only \\(m \ll n\\) constraints.

### Step 2: Classify Observations

Compute residuals on the **full dataset**:
\\(r_i = y_i - x_i^\top \hat\beta^{(0)}\\),
and a leverage-adjusted bandwidth \\(h_i\\) from the Cholesky factor
of \\(X_S^\top X_S\\):

\[
h_i = \sqrt{x_i^\top (X_S^\top X_S)^{-1} x_i}
\]

Use the scaled residuals \\(r_i / h_i\\) to classify each observation
into three groups:

| Group | Condition | Interpretation |
|-------|-----------|----------------|
| \\(\mathcal{L}\\) (below) | \\(r_i / h_i < \kappa_{\text{lo}}\\) | Definitely **below** the true hyperplane |
| \\(\mathcal{U}\\) (above) | \\(r_i / h_i > \kappa_{\text{hi}}\\) | Definitely **above** the true hyperplane |
| \\(\mathcal{M}\\) (middle) | otherwise | **Could go either way** — must be kept |

The thresholds \\(\kappa_{\text{lo}}\\) and \\(\kappa_{\text{hi}}\\)
are chosen so that the middle band contains approximately
\\(M = 0.8 \cdot m\\) observations.

### Step 3: Build Globs

Here is the key insight.  The observations in \\(\mathcal{L}\\) and
\\(\mathcal{U}\\) are so far from the hyperplane that their individual
\\(x_i\\) values do not matter — only their **aggregate effect** on
the LP matters.  So we replace each group with a single **glob**:

\[
\tilde x_L = \sum_{i \in \mathcal{L}} x_i, \quad
\tilde y_L = \sum_{i \in \mathcal{L}} y_i
\]

\[
\tilde x_U = \sum_{i \in \mathcal{U}} x_i, \quad
\tilde y_U = \sum_{i \in \mathcal{U}} y_i
\]

The reduced problem has \\(|\mathcal{M}| + 2\\) observations — a
dramatic reduction from \\(n\\).

!!! info "Why does this work?"
    In the LP dual, observations in \\(\mathcal{L}\\) have dual
    variable \\(d_i = 1\\) (they are below the hyperplane) and
    observations in \\(\mathcal{U}\\) have \\(d_i = 0\\) (above).
    As long as these classifications are correct, the sum of the
    corresponding rows is a sufficient statistic.  The simplex and
    interior-point algorithms only need to "see" the observations
    whose dual variables could change — those in the middle band.

### Step 4: Solve the Reduced Problem

Solve quantile regression on the reduced dataset:

\[
\tilde X = \begin{bmatrix}
  X_{\mathcal{M}} \\
  \tilde x_L^\top \\
  \tilde x_U^\top
\end{bmatrix}, \quad
\tilde y = \begin{bmatrix}
  y_{\mathcal{M}} \\
  \tilde y_L \\
  \tilde y_U
\end{bmatrix}
\]

using the Frisch-Newton interior-point method (or any LP solver).

### Step 5: Verify and Fix Up

Check whether any observations were **misclassified** — that is,
whether any point in \\(\mathcal{L}\\) now has a *positive* residual,
or any point in \\(\mathcal{U}\\) now has a *negative* residual:

- If \\(r_i > 0\\) for some \\(i \in \mathcal{L}\\): reclassify
  \\(i\\) to \\(\mathcal{M}\\) and update the globs.
- If \\(r_i < 0\\) for some \\(i \in \mathcal{U}\\): similarly
  reclassify.
- If no misclassifications: **converge**.

If too many observations are misclassified (more than 10% of \\(M\\)),
double \\(m\\) and restart — the initial subsample was too small.

---

## Complexity Analysis

| Step | Cost |
|------|------|
| Initial subsample solve | \\(O(m \cdot p^{1.5}) = O(\sqrt{p} \cdot n^{2/3} \cdot p^{1.5})\\) |
| Full-data residuals | \\(O(n \cdot p)\\) |
| Classification | \\(O(n)\\) |
| Reduced solve | \\(O(m \cdot p^{1.5})\\) |
| Fixup iterations | a few more \\(O(m \cdot p^{1.5})\\) |

The dominant cost is the initial and reduced solves, each on a problem
of size \\(m \approx \sqrt{p} \cdot n^{2/3}\\).  For comparison:

| Method | Cost | \\(n = 10^6, p = 10\\) |
|--------|------|----------------------|
| OLS | \\(O(n p^2)\\) | \\(10^8\\) |
| FNB (interior point) | \\(O(n^{3/2} p^2)\\) | \\(10^{11}\\) |
| **PFN (preprocessing)** | \\(O(n^{2/3} p^2)\\) | \\(10^6\\) |

The preprocessing approach can be **faster than OLS** for large enough
\\(n\\), because \\(n^{2/3} < n\\).  This is the remarkable conclusion
of Portnoy & Koenker (1997): the Laplacian tortoise overtakes the
Gaussian hare.

---

## Visual Intuition

Consider a 2-D quantile regression at \\(\tau = 0.5\\) (median):

```
    y │                             ╱
      │             ∘            ╱   ← upper glob (one point)
      │        ∘     ∘       ╱
      │   ∘      •     ∘  ╱
      │      •     •   ╱   ← middle band (these matter)
      │   •     •   ╱
      │      •   ╱    ∘
      │   ∘   ╱         ← lower glob (one point)
      │    ╱   ∘
      └──────────────────── x

    ∘ = far from hyperplane (aggregated into globs)
    • = in the middle band (kept individually)
```

Only the handful of points near the regression line need to be
processed individually.  Everything else collapses into two summary
points.

---

## Implementation in Pinball

The `PreprocessingSolver` (`method="pfn"`) implements this algorithm:

```python
from pinball import QuantileRegressor

# Automatically uses preprocessing + Frisch-Newton
model = QuantileRegressor(tau=0.5, method="pfn")
model.fit(X_large, y_large)    # X_large has millions of rows
```

### Configuration

```python
from pinball.linear.solvers.pfn import PreprocessingSolver

solver = PreprocessingSolver(
    mm_factor=0.8,       # middle-band size as fraction of m
    max_bad_fixups=3,    # fixup iterations before doubling m
    eps=1e-6,            # bandwidth floor
)
```

### Fallback behaviour

If the subsample size \\(m \geq n\\), the preprocessing is skipped
and the inner solver runs on the full data directly.  This means
`"pfn"` is safe to use on any dataset — it is never worse than `"fn"`.

---

## When to Use Preprocessing

| Scenario | Recommendation |
|----------|---------------|
| \\(n < 5{,}000\\) | Use `"br"` (simplex) — preprocessing overhead not worthwhile |
| \\(5{,}000 < n < 100{,}000\\) | Use `"fn"` — interior point is fast enough |
| \\(n > 100{,}000\\) | **Use `"pfn"`** — preprocessing gives 10–100× speedup |
| \\(n > 1{,}000{,}000\\) | **Use `"pfn"`** — the only practical option |

### Benchmark: Engel Data Scaled Up

```python
import numpy as np
from pinball import QuantileRegressor
from pinball.datasets import load_engel
import time

# Scale Engel data to 1 million rows
engel = load_engel()
rng = np.random.default_rng(42)
n_big = 1_000_000
idx = rng.integers(0, len(engel.target), n_big)
X_big = engel.data[idx] + rng.normal(0, 10, (n_big, 1))
y_big = engel.target[idx] + rng.normal(0, 20, n_big)

for method in ["fn", "pfn"]:
    t0 = time.perf_counter()
    QuantileRegressor(tau=0.5, method=method).fit(X_big, y_big)
    print(f"{method}: {time.perf_counter() - t0:.2f}s")
# fn:  12.34s
# pfn:  0.28s   ← ~44× faster
```

---

## References

1. Portnoy, S. and Koenker, R. (1997). "The Gaussian hare and the
   Laplacian tortoise: computability of squared-error versus
   absolute-error estimators." *Statistical Science* 12(4): 279–300.
   [DOI:10.1214/ss/1030037960](https://doi.org/10.1214/ss/1030037960)

2. Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
   Chapter 6: "Computational aspects of quantile regression."
