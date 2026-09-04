# Design

## Does this touch `fortran/` or the `_native` extension?

**No.** Pure NumPy/SciPy, no compiled code, no change to the wheel build beyond
one new filename. See the profiling argument in the proposal for why.

One build detail that is easy to miss: `pinball/linear/solvers/meson.build`
lists Python sources **explicitly**, so a new module must be added there or it
silently will not ship in the wheel while still working from a source checkout.

## Public API surface

`pinball/__init__.py` is unchanged. Additions are the `conquer` registry entry
and, in `pinball.linear.solvers.conquer`, the functions `smoothed_loss`,
`smoothed_gradient`, `smoothed_hessian`, `asymptotic_cov`,
`multiplier_bootstrap`, `confidence_intervals` and `default_bandwidth`.

Inference is reached the way `br`'s confidence intervals already are, through
`solver_options`, so no new estimator parameter is introduced:

    QuantileRegressor(method="conquer",
                      solver_options={"ci": "both", "random_state": 0})

## Why the gradient is the whole algorithm

With `rho_tau(u) = (tau - 1/2)u + |u|/2`, the convolution splits into a linear
part and `(h/2) G(u/h)` where `G(t) = int |t - v| K(v) dv`. Differentiating,
`grad L_h(b) = (1/n) X'[Kbar(-r/h) - tau]`, which tends to the ordinary
subgradient as `h -> 0`. The minimiser is unique, so the descent scheme affects
speed only, never the answer — which is why matching R does not require
matching R's iteration internals.

## The bug this design made easy to hit

The compactly supported kernels have polynomial CDFs that are valid **only**
inside `[-1, 1]` and are non-monotone outside. Clipping them with
`np.clip(poly, 0, 1)` sends `Kbar(3)` to 0 for the parabolic kernel when the
answer is 1 — inverting the gradient for every residual larger than `h`. Three
of five kernels still matched R exactly, and the symptom presented as an
optimiser divergence. Only a finite-difference check of the gradient against
the loss found it. The tails are now handled explicitly rather than by clipping.

A second, milder trap: with a compact kernel the gradient is locally *constant*,
so the Barzilai-Borwein ratio can carry no curvature information. Falling back
to the maximum step in that case throws the iterate away and it never recovers;
the fallback is now the last usable step, seeded from the curvature at the
starting point.

## Verification strategy

The two oracles used elsewhere in this repo do not apply, so they are replaced:

1. **Internal consistency** — analytic gradient vs finite differences of the
   loss, Hessian vs finite differences of the gradient, CDF vs its own density
   *including outside the support*, and `G(t)` against quadrature. This layer
   found the only real bug.
2. **A smooth optimality certificate** — zero gradient, positive-definite
   Hessian.
3. **The `h -> 0` limit** — the smoothed fit must approach the exact LP fit as
   the bandwidth shrinks. This is what ties the new estimator to the already
   verified ones, and it is the check that would catch a wrong-but-
   self-consistent loss. Asserted as a trend rather than a monotone sequence:
   on one dataset the bias path need not be monotone, and at very small `h` few
   observations fall in the band so the problem is poorly conditioned.
4. **R parity**, frozen in the test file.
5. **Equivariance**, corrected: regression, reparameterisation and reflection
   hold exactly because they leave residuals unchanged; scale holds only with
   `h` scaled too. A further wart is pinned — the *default* bandwidth is not
   reparameterisation invariant, because it counts covariates by looking for a
   constant column, and mixing the design destroys it.
