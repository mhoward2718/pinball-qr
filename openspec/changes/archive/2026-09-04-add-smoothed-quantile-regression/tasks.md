# Tasks

- [x] Kernels: CDF, density, and the `G(t)` closed form for all five, with the
      tails handled explicitly rather than by clipping.
- [x] Verify `G(t)` against numerical quadrature (45 kernel/t pairs, 2.3e-14).
- [x] Smoothed loss, gradient and Hessian; cross-check by finite differences.
- [x] Barzilai-Borwein descent with a curvature-seeded fallback step.
- [x] Bandwidth default matching R, verified across n and p.
- [x] Register as `method="conquer"`; add to `meson.build` so it ships.
- [x] Asymptotic sandwich covariance; reverse-engineered and matched to R's
      `asyCI` at a ratio of 1.00000.
- [x] Multiplier bootstrap with Exp(1) weights; percentile, pivotal and normal
      intervals; seeded via the package's `_ensure_rng` convention.
- [x] Wire `ci` through `solver_options`.
- [x] Test suite: internal consistency, smooth optimality certificate, `h -> 0`
      limit, R parity, equivariance, failure reporting.
- [x] README credit, stating plainly that no GPL-3 code was used.
- [ ] Bandwidth selection (cross-validated or plug-in) — deliberately out of
      scope, its own change.
- [ ] Coverage simulation for the intervals — the one layer parity cannot
      check, since R and pinball could agree and both be wrong. Local only.
