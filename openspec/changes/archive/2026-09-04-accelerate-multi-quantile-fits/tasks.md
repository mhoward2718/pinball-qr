# Tasks

- [x] Repair the meson editable build in the `pinball-dev` env; confirm the
      suite runs with **zero** `_has_native` skips (a green run with everything
      skipped proves nothing). Baseline recorded: `1 failed, 230 passed`.
- [x] Detect the exhausted iteration budget and non-finite coefficients in
      `FNBSolver`; reject non-positive `eps`; populate `dual_solution`.
- [x] Tests for the above, **verified to fail against `HEAD`** (5/5 did).
- [x] Add `BaseSolver.solve_multi` with a per-τ default; route multi-τ fits
      through it in `QuantileRegressor.fit`.
- [x] Implement the CFM grid path in `PreprocessingSolver`, mirroring
      `rq.fit.pfnb` (one subsample, one band, `m0` reset per τ, carried residuals).
- [x] Replace the Gram-matrix/inverse leverage band with a QR-based one.
- [x] Add `random_state`, reusing `_ensure_rng` from `_bootstrap.py`.
- [x] Fix the non-termination path; **confirmed the old code hangs** under a
      25 s timeout and the new code returns.
- [x] Build the optimality certificate and equivariance suites.
- [x] Benchmark blocked `dsyrk` as a go/no-go gate before writing Fortran
      (5.7–6.4x on the kernel → go).
- [x] Implement blocked `dsyrk` in `stepy`; keep `ratfor/rqfnb.r` in sync;
      retain a rank-1 fallback for `d < 0`.
- [x] `stepy` unit tests across block boundaries (`nb-1`, `nb`, `nb+1`, `3nb+7`)
      and `p` in 1..64.
- [x] Re-run live R parity after the Fortran change.
- [x] `make lint` clean.
- [ ] Confirm the cibuildwheel matrix is green on all four runners — the only
      evidence that the Fortran change holds cross-platform.
- [x] Fix `test_sklearn_compliance`: reject all-zero `sample_weight` with an
      error naming the weights. **Verified to fail against `HEAD`** and to pass
      with the fix. Suite is now fully green (378 passed, 0 failed).
