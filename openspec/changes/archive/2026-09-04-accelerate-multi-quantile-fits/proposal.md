# Accelerate multi-quantile fits, and stop failure being silent

## Why

Three findings from `research/lp-solvers-and-boosting.md`, all verified against
live R `quantreg` 6.1 and measured on this machine.

1. **`rqfnb` could return a wrong answer with no warning.** `lpfnb` leaves its
   loop on either a small duality gap *or* an exhausted iteration budget, and
   sets `info` in neither case — `info` is written only by `stepy`'s `dposv`.
   A second, worse path was found while testing: with a non-positive `eps` the
   routine returns **NaN coefficients with `info = 0`**. Both are upstream
   behaviours, so the fix belongs in the Python layer where it costs no parity.

2. **Every quantile in a grid was solved from scratch.** `quantreg` has shipped
   the Chernozhukov, Fernández-Val & Melly (2022) algorithm for years as
   `rq.fit.pfnb`: share one subsample across the grid and seed each τ with the
   previous fit's residuals. Measured here: **3.2x at n=50,000, p=20, 99 τ** and
   **4.6x at n=200,000**, with total fixups collapsing from 7141 to 2.

3. **`stepy` used BLAS-2 where BLAS-3 applies.** It accumulated
   `A·diag(d)·Aᵀ` with `n` rank-1 `dsyr` calls. Blocked `dsyrk` is **5.7–6.4x**
   faster on the kernel (measured in compiled C against the same OpenBLAS), and
   `stepy` is 25–32% of a Frisch-Newton iteration.

A fourth defect was already failing the suite before this work started:
`check_estimator` fails on scikit-learn 1.9, which added
`check_all_zero_sample_weights_error`. All-zero weights filtered every row away
and the empty design surfaced as a generic "need at least 2 samples" from inside
the solver. Since CI installs scikit-learn unpinned, CI was red on this too.

Two further defects surfaced while implementing: `pfn` formed the
normal-equations Gram matrix and then explicitly inverted its Cholesky factor
(squaring the condition number), and it could **loop forever** — confirmed by
running the old code to a timeout.

## What changes

- `FNBSolver` detects an exhausted iteration budget and non-finite coefficients,
  warns, and reports both through `SolverResult.status`. It also fills in
  `dual_solution`, which was hardcoded `None`.
- `BaseSolver` gains `solve_multi`; the default implementation loops, so every
  existing solver behaves exactly as before. `QuantileRegressor` routes
  multi-τ fits through it.
- `PreprocessingSolver` implements the grid path (CFM Algorithm 2), gains
  `random_state`, computes its leverage band by QR, and escalates instead of
  looping forever.
- `stepy` accumulates with blocked `dsyrk`, keeping a rank-1 fallback for `d < 0`.
- `QuantileRegressor.fit` rejects all-zero `sample_weight` with an error that
  names the weights, restoring `check_estimator` on scikit-learn 1.9.

## R reference

`quantreg::rq.fit.pfnb` for the multi-τ path, `quantreg::rq.fit.fnb` and
`rq.fit.br` for the single-τ solvers, `quantreg::bandwidth.rq` for the
bandwidth constants. Measured agreement after all changes: `br` 2.3e-14,
`fn` 2.8e-12, `pfn` 1.5e-7, multi-τ vs `rq.fit.pfnb` 1.6e-7 — against the
harness bar of `TOL_ABS=1e-4`, `TOL_REL=1e-3`.

## Deliberate divergences from quantreg

- **Non-termination fix.** R's `rq.fit.pfn` has the same structure and can loop
  forever; we double the subsample instead. Changes no returned value, only
  whether the call returns at all.
- **QR leverage band.** R forms `chol(crossprod(xs))` and inverts it. Same
  leverages, but at `cond(Xs)` rather than `cond(Xs)²`.
- **Blocked `dsyrk`.** Results are not bitwise identical to the `dsyr` loop
  (different summation order, ~1e-14); `fn`'s distance from R moved from
  1.9e-13 to 2.8e-12 accordingly.

## Non-goals

- Interior-point warm starting across τ. Measured as a **negative result**: a
  primal-only warm start needs 441–569 iterations against 429 cold, and an
  aggressive one diverges. CFM warm-starts the *preprocessing*, not the barrier;
  every reduced LP here is still solved cold.
- Gondzio multiple centrality correctors (`lpfnb` already does Mehrotra).
- The `O(n²)` `dsol` allocation that caps `rqbr`'s full-quantile-process mode.
- Shrinking the retained subsample to `O(n^{1/2})`. That is the paper's theory;
  `rq.fit.pfnb` resets `m0` per τ and we mirror it.
