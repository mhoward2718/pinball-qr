# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pinball-qr` — quantile regression for Python. The estimation engine is a set
of Fortran 77 linear-programming solvers ported from Roger Koenker's R
`quantreg`, wrapped in an sklearn-compatible API. The nonparametric estimator
is ported from the R `QuantifQuantile` package, and the ADMM solver from POGS.

## Spec-driven development (OpenSpec)

This repo uses OpenSpec. **Behavior changes start with a spec, not with code.**

- `openspec/specs/<capability>/spec.md` — what the system does *today*. Five
  capabilities: `linear-quantile-regression`, `lp-solvers`,
  `statistical-inference`, `nonparametric-quantization`, `r-reference-parity`.
- `openspec/changes/<name>/` — in-flight proposals (proposal, delta specs,
  design, tasks). Deltas use `## ADDED/MODIFIED/REMOVED Requirements`; main
  specs use plain `## Requirements`.
- `openspec/config.yaml` — project context and per-artifact rules that OpenSpec
  feeds to the model. Read it before proposing; it encodes the conventions
  below.

Workflow commands: `/opsx:propose`, `/opsx:apply`, `/opsx:archive`,
`/opsx:sync`, `/opsx:update`, `/opsx:explore`. Planning and implementation are
separate steps — `propose` writes artifacts only and stops.

```bash
openspec list --specs          # capabilities and requirement counts
openspec show <spec-id>        # read a spec
openspec validate --all        # must pass before committing spec changes
```

## Commands

```bash
make dev          # pip install -e ".[dev,docs]" --no-build-isolation
make test         # pytest
make lint         # ruff check pinball tests
make format       # ruff format + ruff check --fix
make coverage     # pytest --cov, writes htmlcov/
make docs-serve   # mkdocs live reload
```

Single test / single case:

```bash
pytest tests/test_solver_fnb.py
pytest tests/test_solver_fnb.py::TestFNBSolverIntegration::test_engel_median
pytest tests/test_solver_lasso.py::TestLassoSolverSymmetricPenalty
pytest -k "lambda"
```

### Distribution: users get wheels, not a source build

`pip install pinball-qr` installs a **prebuilt wheel** — end users never need
gfortran or a BLAS toolchain. `.github/workflows/build-publish.yml` drives
cibuildwheel over cp310–cp313 for Linux x86_64 (manylinux_2_28), macOS arm64
and x86_64, and Windows AMD64, plus an sdist; all cibuildwheel config lives in
`[tool.cibuildwheel]` in `pyproject.toml`. Wheels build on every push to main
and every PR — not only on release — so a broken build surfaces before a tag.
Publishing to PyPI happens via trusted publishing on `v*` tags only.

Each wheel is tested as built (`test-command = pytest {project}/tests`), so the
compiled Fortran path *is* genuinely exercised there. If you change the build
(meson.build, `.f2py_f2cmap`, dependencies, or anything in `fortran/`), the
wheel matrix is what proves it still works across platforms — say so in
design.md, since it affects all four runners.

### Local development from a checkout

The above is the packaged path; working from a source checkout is where the
sharp edges are:

1. **After changing anything in `fortran/`, rebuild before testing** — a stale
   `.so` means you are testing the old solver. `make install` (or `make dev`)
   rebuilds; `pip install -e . --no-build-isolation` is the underlying command.
2. **Running `pytest` from the repo root can import the source tree without
   compiled extensions.** Tests needing the extension are guarded by a local
   `_has_native()` helper and **skip silently** when it is absent, so a green
   local run does not by itself prove the Fortran path ran — check the skip
   count. CI's pure-Python job relies on exactly this skipping behavior; its
   compiled job installs the package and runs from a temp directory so the
   *installed* build is imported instead.

## Architecture

The call path for a linear fit is:

```
QuantileRegressor.fit()          pinball/linear/_estimator.py
  └─ get_solver(method)          pinball/linear/solvers/__init__.py   (registry)
       └─ BaseSolver.solve()     pinball/linear/solvers/base.py       (template method)
            └─ _solve_impl()     br.py / fnb.py / pfn.py / lasso.py / pogs.py
                 └─ pinball._native.rqbr / rqfnb                      (Fortran)
```

Three patterns hold this together, and they are load-bearing:

- **Strategy + registry.** Solvers are selected at runtime by name. Adding a
  solver means calling `register_solver(name, cls)` — no existing code changes.
  Registered: `br`, `fn`/`fnb`, `lasso`, `pfn`, `pogs`.
- **Template method.** `BaseSolver.solve()` is concrete and does validation and
  preparation; subclasses implement `_solve_impl()` only. Put shared invariants
  in `solve()`, never duplicate them per solver.
- **Uniform result.** Every solver returns a `SolverResult` (coefficients,
  residuals, objective_value, status, iterations, solver_info). Estimator code
  depends on that shape, not on which solver ran.

`QuantileRegressor` accepts `tau` as a scalar or a sequence; multi-tau fits
loop over levels and stack into `coef_` of shape `(p, n_quantiles)`.

Other subsystems: `pinball/linear/_inference.py` (iid / nid / ker / rank
standard errors), `_bootstrap.py` (xy, wild, mcmb), and
`pinball/nonparametric/quantization/` (CLVQ grid + Voronoi cells).

## Conventions specific to this codebase

**Numerical parity with R is the acceptance bar.** A change to estimation
behavior is not done until it has been checked against the R reference
(`quantreg`, or `QuantifQuantile` for the quantization estimator). Name the R
function you compared against in the proposal or spec. Deliberate divergences
must be documented, not left as unexplained numeric differences.

**Test at extreme tau, not just the median.** Multiple real defects in this
repo's history were mathematically invisible at `tau=0.5` and only appeared in
the tails — an asymmetric LASSO penalty, and slow POGS convergence. A test
suite that only exercises `tau=0.5` cannot catch this class of bug. Every
estimation spec requires at least one extreme-tau scenario.

**Treat `fortran/*.f` as ported code.** The `.f` files are Ratfor output; the
originals live in `fortran/ratfor/*.r`. Prefer fixing things in the Python
layer. If a Fortran change is genuinely necessary, keep the Ratfor source in
sync, justify the divergence from `quantreg`, and note in the design that the
wheel build matrix is affected.

**Preserve attribution.** Ported code carries upstream authorship and license
notices (Koenker et al.; Charlier/Paindaveine/Saracco; Fougner). Keep them.

**Do not silently return an untrustworthy fit.** Solvers warn on non-success
status and on failure to reach tolerance. When adding a solver, surface
numerical failure through both a warning and `SolverResult.status`.

## Local scratch

`local_testing/` is gitignored and holds ad hoc R-comparison scripts and dev
experiments used to validate solver fixes. `research/` holds exploratory
research notes, clearly separated into verified literature findings versus
unverified proposals — respect that separation when adding to it.
