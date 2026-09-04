# Design

## Does this touch `fortran/` or the `_native` extension?

**Yes** — `fortran/rqfnb.f`, with `fortran/ratfor/rqfnb.r` updated in lockstep
as `openspec/config.yaml` requires. No signature changes, so the f2py interface,
`meson.build` and `.f2py_f2cmap` are untouched, and `rqfn.f` — which calls the
shared `stepy` — needs no edit. Cross-platform evidence is the cibuildwheel
matrix; the one platform-specific hazard is the block buffer, capped near
128 KiB to stay inside the **1 MiB Windows thread stack**.

## Public API surface

`pinball/__init__.py` is unchanged. Additions are `BaseSolver.solve_multi`,
`PreprocessingSolver(random_state=...)`, two `FNBSolver` status constants, and
`SolverResult.dual_solution` now being populated by `FNBSolver`.

`get_solver(name)` takes no constructor kwargs, so `random_state` reaches the
solver through the existing `solver_options` route:
`QuantileRegressor(method="pfn", solver_options={"random_state": 0})`. No new
estimator parameter — one would be meaningless for `br`/`fn`/`pogs`.

## Why the grid path stays exact regardless of τ spacing

The pinball loss is convex and positively homogeneous, hence subadditive:
`ρ(Σuᵢ) ≤ Σρ(uᵢ)`. For *any* glob sets the reduced objective is therefore a
pointwise lower bound on the full one. If at the reduced optimum every globbed
observation has the sign its glob assumed, the loss is linear on each glob and
the two objectives agree there — and a minimiser of a lower bound that touches
the true objective minimises the true objective. Nothing in that argument
mentions τ spacing, subsample size, the band, or the seed: those affect only how
many fixup rounds are needed. A coarse grid is as exact as a fine one, just
slower. The single way to break it is returning before the sign check passes,
which is why the loop escalates rather than giving up.

## Verification, and its one real gap

Because exactness is unconditional, **no assertion on coefficients, objectives,
or optimality can tell a correct warm start from a broken one** — the fixup loop
silently repairs a bad one. Cold-vs-warm agreement, grid independence and the
optimality certificate all pass either way. The discriminating test is therefore
on **counters** (`tests/test_solver_pfn_multi.py::test_warm_start_reduces_work`):
the grid path must do strictly fewer inner iterations than a cold run. It is a
correctness test, not a benchmark, and asserts on deterministic counts rather
than wall-clock.

Supporting layers: an exact optimality certificate (`tests/_optimality.py`),
Koenker's equivariance identities, cross-solver agreement, and live R parity.
The certificate needs two conditions, not one — dual feasibility alone accepted
a β perturbed by 1e-6 at four of five τ, because the dual depends only on
residual signs.
