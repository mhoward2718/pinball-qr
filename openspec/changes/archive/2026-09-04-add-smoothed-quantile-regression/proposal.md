# Add convolution-smoothed quantile regression (conquer)

## Why

Every solver in this package minimises the pinball loss exactly, by linear
programming. Convolution smoothing takes the other route: replace the
non-differentiable check function with a kernel-smoothed surrogate and use
gradient descent. It is the main post-1997 development in quantile regression
computation, and `quantreg` itself offers it via `rq.fit.conquer`.

It is worth having for what it *is*, not for speed: smoothing strictly lowers
the asymptotic variance and relaxes the dimension-growth condition for
asymptotic normality, at the cost of an `O(h^(s+1))` bias.

## What changes

- A `conquer` solver: five kernels (Gaussian, logistic, uniform, parabolic,
  triangular), R's bandwidth default `h = ((log n + p)/n)^0.4`, Barzilai-Borwein
  descent from a Newton-scaled first step.
- Inference: the sandwich asymptotic covariance and a multiplier bootstrap,
  giving all four interval types R reports (asymptotic, percentile, pivotal,
  normal). Reachable through `solver_options={"ci": "both"}`.
- README credit for the method's authors and the reference package.

## This estimator is different, and the specs must say so

conquer targets a pseudo-parameter, not the exact quantile. Two existing
requirements would be wrong if applied to it:

- **"Solvers agree with one another"** is already scoped to *exact* solvers, so
  conquer sits outside it by construction. Measured: it differs from the simplex
  solution by ~7e-3 on a coefficient scale of ~3.
- **The optimality certificate** in the test suite certifies minimisers of the
  pinball loss. conquer minimises something else and will not certify. It gets
  its own certificate instead: `L_h` is convex and differentiable, so a zero
  gradient is necessary and sufficient.

It is also **not scale equivariant in y** at the default bandwidth, because `h`
depends on `(n, p)` alone and does not follow the data. Scaling `h` alongside
`y` restores the identity to ~3e-10. Both directions are pinned by tests.

## Implementation is independent, not a port

The R `conquer` package is **GPL-3**; this project is MIT. Copying its C++
would relicense the project, so it was not consulted. The algorithm comes from
the published papers; the bandwidth default and the exact form of the sandwich
variance were determined by probing R's *outputs*, and both are pinned against
R in the test suite.

NumPy rather than Fortran, on measurement: at n=1e6 the work is 45% BLAS `gemv`
(identical under any language), 44% the `erf` transcendental (also already
compiled), and 11% temporaries. Fusing the temporaries and using Fortran-ordered
data gives 1.15x, which is about the entire gap to the C++ reference (1.2x at
n=1e6). A new Fortran source would add four wheel-matrix surfaces and break the
convention that `fortran/ratfor/*.r` is the readable source of truth, for
roughly nothing.

## R reference

`conquer::conquer` from R package conquer 1.3.3. Measured agreement: coefficients
≤3.7e-9 across five kernels × three quantiles; asymptotic CI 2.3e-10; bootstrap
intervals within 1% on width, which is Monte Carlo noise since the draws cannot
be matched across RNGs.

## Non-goals

- Replacing any exact solver. conquer is additive and opt-in.
- `conquer.process` / `conquer.reg` (penalised and process variants).
- Bandwidth *selection* (cross-validated or plug-in); only the fixed default and
  a user-supplied value are supported.
- Matching R's bootstrap draw-for-draw, which is impossible across RNGs.
