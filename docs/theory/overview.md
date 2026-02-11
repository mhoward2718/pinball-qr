# Quantile Regression: An Overview

## Beyond the Mean

Ordinary least squares (OLS) estimates the **conditional mean**
\\(E[Y \mid X = x]\\).  This is a single summary of how \\(Y\\) relates
to \\(X\\) — useful, but incomplete.  It tells us nothing about:

- How the **spread** of \\(Y\\) changes with \\(X\\) (heteroscedasticity)
- How the **tails** of the distribution shift
- Whether the effect of \\(X\\) on \\(Y\\) is **symmetric**

**Quantile regression** (Koenker & Bassett, 1978) estimates the entire family
of **conditional quantile functions**:

\[
Q_\tau(Y \mid X = x) = x^\top \beta(\tau), \quad \tau \in (0, 1)
\]

Each value of \\(\tau\\) gives a different regression surface.  Together they
paint a complete picture of how the conditional distribution of \\(Y\\) depends
on \\(X\\).

## The Pinball Loss

Where OLS minimises the sum of squared residuals, quantile regression
minimises the **pinball (check) loss**:

\[
\hat\beta(\tau) = \arg\min_{\beta} \sum_{i=1}^{n} \rho_\tau(y_i - x_i^\top \beta)
\]

where the check function is:

\[
\rho_\tau(u) =
\begin{cases}
\tau \cdot u & \text{if } u \geq 0 \\
(\tau - 1) \cdot u & \text{if } u < 0
\end{cases}
\]

For \\(\tau = 0.5\\) this reduces to the **median regression** (minimising
the sum of absolute deviations).  For other values of \\(\tau\\) the loss
is asymmetric — it penalises positive residuals by \\(\tau\\) and negative
residuals by \\(1 - \tau\\).

## Why Quantile Regression?

### 1. Robustness

The median regression (\\(\tau = 0.5\\)) is far more robust to outliers
than OLS.  The breakdown point of the median is 50%, while for OLS it
is \\(1/n\\).  This was already understood by Boscovich and Laplace in
the 18th century (Koenker, 2005).

### 2. Heteroscedasticity

In many applications the variance of \\(Y\\) is not constant across
values of \\(X\\).  For example, in Engel's data on food expenditure:
wealthier households show much more *variability* in food spending.
Quantile regression reveals this naturally — the slopes of the upper
and lower quantiles diverge.

### 3. Distributional Effects

Policy questions often concern the tails.  "Does a training programme
help the *least skilled* workers?" is a question about the lower
quantiles, not the mean.  Quantile regression provides a direct answer.

### 4. No Distributional Assumptions

Unlike maximum likelihood methods, quantile regression makes no
assumption about the error distribution.  The pinball loss is
*distribution-free* — it works whether errors are Gaussian,
heavy-tailed, skewed, or heteroscedastic.

## Connection to Linear Programming

Minimising the pinball loss is a **linear program** (LP).  Writing the
problem in the usual LP form is the key insight that enables efficient
computation.  With \\(u_i = \max(0, r_i)\\) and \\(v_i = \max(0, -r_i)\\):

\[
\min_{\beta, u, v} \; \tau \, \mathbf{1}^\top u + (1 - \tau) \, \mathbf{1}^\top v
\quad \text{s.t.} \quad X\beta + u - v = y, \quad u, v \geq 0
\]

This LP can be solved by the **simplex method** (Barrodale & Roberts, 1974)
or by **interior-point methods** (Portnoy & Koenker, 1997).

## The Duality Principle

The dual of the quantile regression LP is elegant and important for
inference.  It takes the form:

\[
\max_{d} \; y^\top d
\quad \text{s.t.} \quad X^\top d = (1 - \tau) X^\top \mathbf{1}, \quad
0 \leq d_i \leq 1
\]

The dual solution \\(d\\) classifies observations:

| \\(d_i\\) | Interpretation |
|-----------|----------------|
| \\(d_i = 0\\) | Observation is *above* the fitted hyperplane |
| \\(d_i = 1\\) | Observation is *below* the fitted hyperplane |
| \\(0 < d_i < 1\\) | Observation lies *on* the hyperplane (an interpolation point) |

This dual structure is exploited by the Barrodale-Roberts solver and
underlies the rank-inversion confidence intervals.

## Historical Context

The history of \\(\ell_1\\) estimation is surprisingly rich:

- **1757** — Boscovich proposes minimising absolute deviations for fitting
  a line to astronomical data
- **1789** — Laplace develops computational methods for \\(\ell_1\\) regression
- **1809** — Gauss publishes the method of least squares; \\(\ell_1\\) methods
  fall out of favour due to computational difficulty
- **1978** — Koenker & Bassett introduce the full quantile regression
  framework
- **1997** — Portnoy & Koenker show that interior-point methods with
  preprocessing can make \\(\ell_1\\) regression *faster* than \\(\ell_2\\)
  for large problems

## Further Reading

- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Koenker, R. (2024). `quantreg`: Quantile Regression. R package.
- Portnoy, S. and Koenker, R. (1997). "The Gaussian hare and the
  Laplacian tortoise." *Statistical Science* 12(4): 279–300.
