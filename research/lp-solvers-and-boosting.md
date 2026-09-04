# Can Koenker's LP Solvers Be Improved, and Can Globbing Meet Gradient Boosting?

**Status:** research note. Three strictly separated parts.
**Part 0** is measured on this repo and this machine — reproducible, and the
numbers are mine. **Part 1** is literature, multi-source and adversarially
verified (101 agents, 5 angles, 3-vote verification). **Part 2** is my own
proposal — no literature source, not implemented, not proven.
Do not cite Part 2 as if it came from a paper.

Companion to [`globbing-beyond-linear-qr.md`](globbing-beyond-linear-qr.md),
whose Part 1 negative results (no published globbing→boosting or
globbing→quantization extension) were treated as established here and used to
scope the search toward what is *new* relative to them.

---

## First, a correction to the premise

The question was posed as "this has been the case since like the 80s." That
isn't quite right, and the difference matters for what's left to improve:

| solver | algorithm | vintage |
|---|---|---|
| `rqbr.f` | Barrodale–Roberts simplex, Koenker & d'Orey | ~1987 |
| `rqfnb.f` | Frisch–Newton primal-dual log-barrier IPM | 1997 |

More specifically, `rqfnb` is **not** a naive primal-dual step. Koenker's own
header (`fortran/ratfor/rqfnb.r:11-14`) says it is

> "a projected Newton primal-dual logarithmic barrier method which uses
> **the predictor-corrector approach of Mehrotra** for the mu steps."

and the compiled code confirms it: the `mu * ((g/mu)**3)` adaptive centering
and the `dr` term carrying `dx(i)*dz(i)/x(i)` in `lpfnb` are exactly Mehrotra's
second-order corrector. So "swap in Mehrotra" is not an available win — that
box was ticked in 1997. The next rung up is Gondzio's *multiple* centrality
correctors, not Mehrotra.

The honest summary of question (1): the *algorithm* is 1997-vintage and still
close to state-of-the-art for its target regime (dense design, large n, small
p). The *implementation* has a measurable inefficiency, and the *multi-quantile
usage pattern* has a large one. Details below.

---

## Part 0 — What I measured in this repo

All numbers from the built `_native` extension in `build/cp312`, this machine.
Benchmark sources are in the session scratchpad (`bench.c`, `bench_rqfnb.py`,
`bench_warm2.py`).

### 0.1 `stepy` uses BLAS-2 where BLAS-3 would do (~15–20% end-to-end)

`fortran/rqfnb.f`, subroutine `stepy`, forms the normal-equations matrix
`A·D·Aᵀ` with `n` separate rank-1 BLAS-2 updates:

```fortran
do23042 i=1,n
call dsyr('U',p,d(i),a(1,i),1,ada,p)
23042 continue
```

The same matrix is one BLAS-3 `dsyrk` on `B = A·diag(√d)`. Measured in
compiled C against the same OpenBLAS (`bench.c`, scaling copy included in the
`dsyrk` time, so it is apples-to-apples):

```
        n    p    dsyr loop      dsyrk   speedup   rel.err
    10000    5        0.8ms      0.2ms      3.5x   3.8e-15
   100000    5        5.4ms      2.0ms      2.7x   1.7e-14
   100000   20       11.1ms      4.0ms      2.8x   1.6e-14
  1000000    5       25.0ms      9.5ms      2.6x   4.4e-14
  1000000   20       92.0ms     40.4ms      2.3x   6.7e-14
```

Agreement to machine precision, 2.3–3.5× on the kernel. **But** `stepy` is only
part of an iteration. Timing the repo's own `_native.stepy` against the measured
per-iteration cost of a full solve:

```
  n=100,000  p=5 :  stepy  4.07 ms | full IPM iter ~ 16.5 ms | share ~25%
  n=500,000  p=20:  stepy 52.19 ms | full IPM iter ~161.3 ms | share ~32%
```

Amdahl on those shares gives **~15–20% end-to-end** on `rqfnb`. Real, worth
doing, not dramatic. The other ~70% is the ~6 `O(np)` `dgemv` calls per
iteration, which are already optimal for this shape.

> A caution on method: a first version of this benchmark written at the NumPy
> level reported 10–45×. That was almost entirely Python per-call overhead.
> The compiled figure is the real one.

### 0.2 Every τ is solved cold — the largest structural inefficiency

`pinball/linear/_estimator.py:215` is `for t in taus: solver.solve(X, y, t)`.
In `fnb.py`, τ reaches the Fortran through **only** `rhs = (1-τ)·colSums(X)` and
the initial dual `wn[:,0] = 1-τ`. Everything else is τ-independent. A 19-point
τ grid, cold:

```
  n=100,000  p=5 :  6.89 s  (363 ms/tau) | IPM iters/tau min 14, med 22, max 34, TOTAL 429
  n=500,000  p=20: 79.67 s (4193 ms/tau) | IPM iters/tau min 15, med 26, max 38, TOTAL 497
```

Per-τ iteration counts at n=100,000, p=5 (this is the actual vector, not a
summary — an earlier draft of this note over-claimed monotonicity from the
min/med/max alone):

```
  tau  :  0.05  0.10  0.15  0.20  0.25  0.30  0.35  0.40  0.45  0.50  0.55  0.60  0.65  0.70  0.75  0.80  0.85  0.90  0.95
  iters:    27    28    31    28    24    22    19    17    17    14    14    19    21    22    24    24    24    34    20
```

The shape is a clear **U with its minimum at the median** (14 at τ=0.50–0.55)
and roughly 1.5–2× that in the tails — but it is **not monotone**: τ=0.95 costs
20 iterations, fewer than τ=0.90's 34. So "the tails are more expensive on
average" is supported; "cost increases monotonically toward the extremes" is
**not**, on this design.

### 0.3 Naive IPM warm-starting across τ does NOT work (measured)

`wn[:,0]` is the settable initial primal point and `wn[:,2]` returns the optimal
primal weights, so a τ-homotopy is directly testable. 19 τ's, n=100k, p=5,
reference = cold solve:

```
      mode      time s  iters  max/tau  hit maxit  max coef reldiff
      cold       10.72    429       34          0          0.00e+00
      clip1e-3   44.55   1929      500          1          1.24e-02   <-- DIVERGED
      clip0.05   10.52    569       71          0          2.49e-09
      blend 0.9  13.00    549       63          0          1.58e-09
      blend 0.5  15.42    441       30          0          2.72e-09
```

Two results. (a) The aggressive start **diverges** — 4.5× the iterations, one τ
pinned at `maxit`, coefficients wrong in the 2nd decimal. This is the textbook
IPM warm-start failure: at the optimum most weights sit *at* 0 or 1, so
clipping lands the start on the boundary with tiny, badly-balanced
complementarity products. (b) The well-behaved variants converge correctly
(~1e-9) but are **neutral-to-worse** on iterations (441–569 vs 429 cold).

**Diagnosis:** the dual is never warm-started. In `lpfnb`, `y` is rebuilt by
`dgemv` on the τ-independent `a`,`c` and `stepy` with `d=1` on *every* call, and
`z`,`w` are re-derived from it. A "warm" start therefore hands the solver a
point whose primal and dual halves are mutually inconsistent — worse than a
consistent cold start. This is a measured result at one (n,p) and one design,
not a proof; but it is a clean explanation of why the published literature
warm-starts at a different level (Finding 5).

### 0.4 Full-process mode exists but is capped by an O(n²) allocation

`br.py:61` implements `tau=None` → the entire quantile process in one
parametric pass. But it allocates `dsol` as `(n, 3n)`: at n=100,000 that is
**240 TB**. The algorithm itself only ever touches columns `lsol` and `lsol+1`
of `dsol` (`rqbr.f:336-368, 594`) during the solve — the full `n × 3n` array is
needed only to *return* the whole dual path. So the memory allocation, not the
algorithm, is what makes whole-path solving infeasible; `sol` alone (the
coefficient path) is only `(p+3, nsol)`.

### 0.5 Engineering defects (not research — but real)

- **Silent non-convergence.** `lpfnb` hardcodes `maxit=500` and exits the loop
  with `gap > eps` leaving `info` untouched. `info` is only ever set by
  `stepy`'s `dposv`. `fnb.py` warns only on `info != 0`, so an
  iteration-limit failure is **completely invisible**. §0.3 above hit exactly
  this: a wrong answer returned with no warning. This should be fixed
  regardless of anything else in this note.
- **`pfn.py:76-80`** forms `Xs.T @ Xs` (squaring the condition number), Cholesky-
  factors it, then takes an explicit `np.linalg.inv(L)`. A QR of `Xs` plus
  `solve_triangular` gets the same leverage band at the true condition number.
- **`pfn.py`** uses the unseeded global `np.random.choice`, so `pfn` fits are
  not reproducible and the "double m and retry" loop can't be replayed.
- **`fnb.py`** re-does `np.asfortranarray(X.T)` per τ; `rqfnb` destroys `a`, so a
  copy is required, but a k-quantile fit could memcpy from one cached buffer.

---

## Part 1 — The five best-supported published findings

Verified across 5 search angles with 3-vote adversarial checking. Confidence
and vote splits are the harness's.

### Finding 1 — Convolution smoothing ("conquer") is the main post-1997 development, and it is a *different estimator*, not just a faster solver
*Confidence: high (3-0).*

He, Pan, Tan & Zhou, *Smoothed Quantile Regression with Large-Scale Inference*,
J. Econometrics 2023 ([arXiv:2012.05187](https://arxiv.org/abs/2012.05187));
Fernandes, Guerre & Horta, *Smoothing Quantile Regressions*, JBES 2021
([arXiv:1905.08535](https://arxiv.org/abs/1905.08535));
[`conquer` on CRAN](https://cran.r-project.org/web/packages/conquer/conquer.pdf).

Smoothing "turns the non-differentiable quantile loss function into a
twice-differentiable, convex and locally strongly convex surrogate, which
admits a fast and scalable Barzilai-Borwein gradient-based algorithm" — no LP,
no Newton system. Critically it is *statistically* different: FGH Theorem 3
gives `Σ_h(τ) = Σ(τ) − c_k·h·D⁻¹(τ) + o(h)` with `c_k > 0`, i.e. the asymptotic
variance is strictly **smaller**; and asymptotic normality holds under a weaker
dimension condition (`p^{8/3}/n → 0` vs `p⁴/n → 0`). `conquer` now ships inside
`quantreg` itself — the QR community treats it as a substitute for the Fortran
routines.

### Finding 2 — But the wall-clock claim for smoothing is *not* established by the founding paper
*Confidence: high (3-0).*

Same sources. A full-text grep of FGH for `cpu|wall|timing|simplex|interior-point|Frisch|Newton|faster|speed|linear program` returned **zero** substantive hits; there is no empirical application section and the largest simulation is n=1000. Evaluation is RMSE ratio and CI coverage only. The costs are real and verbatim: the estimator targets a **pseudo-parameter**, `β_h(τ) = β(τ) − h^{s+1}B(τ) + o(h^{s+1})`, described as "the price we pay"; and with higher-order kernels `R_h` "is not necessarily convex," with convexity proven only asymptotically in a neighbourhood (conquer avoids this by using non-negative kernels).

**Reading for this repo:** smoothing is a genuine and probably better estimator for large-scale work, but adopting it is a *statistical* decision (bias, bandwidth, a different target), not a drop-in speedup. Any "conquer is faster" claim needs He et al.'s own timing tables, which this pass did **not** verify.

### Finding 3 — GPU/first-order LP is real, beats the ADMM family badly, but does not transfer to this solver
*Confidence: high (3-0 on 8/10 claims; 2-1 on the SCS benchmark and the structure claim).*

PDLP ([arXiv:2106.04756](https://arxiv.org/abs/2106.04756));
cuPDLP.jl ([arXiv:2311.12180](https://arxiv.org/abs/2311.12180), Operations Research 2024);
cuPDLPx ([arXiv:2507.14051](https://arxiv.org/html/2507.14051));
GPU-FOM survey ([arXiv:2506.02174](https://arxiv.org/pdf/2506.02174)).

Three things, and the third is the one that matters here:
1. **It works.** On 383 MIPLIB-derived LPs at 1e-8, PDLP gives a "6.3x reduction in geometric mean solve time" over SCS and cuts unsolved from 227 to 49. cuPDLP.jl on an H100 is 4×/10×/20× over CPU at 1e-4 and "comparable ... to Gurobi."
2. **Nobody has pointed it at quantile regression.** Full-text grep of the PDLP paper: `quantile` = 0 across 13,019 words.
3. **The structural win doesn't apply.** PDHG's advantage is *avoiding a large factorization* — the survey measures 0.971 MB extra GPU memory for SpMV vs 481 MB for Cholesky. But `rqfnb` declares `ada(p,p)` and calls `dposv`: a **dense p×p** Cholesky, trivial for the tens-of-columns p typical in QR. Factorization was never the bottleneck, so removing it buys nothing. Also, PDLP's edge is confined to moderate (1e-4) accuracy; at 1e-8 it is *inferior to Gurobi barrier*.

**Note:** PDLP's own taxonomy places **POGS** — this repo's `pogs.py` — in the same ADMM family as SCS. The 6.3× result bears directly on that solver.

### Finding 4 — Sparse Frisch-Newton changes the complexity *exponent*, but only in the sparse/large-p regime
*Confidence: high (3-0).*

Koenker & Ng, *A Frisch-Newton Algorithm for Sparse Quantile Regression*, Acta Math. Appl. Sinica 2005 ([PDF](http://www.econ.uiuc.edu/~roger/research/sparse/fn3.pdf)).

Verbatim: `rqfn: log(Time) = -11.49 + 2.53 log(n)` vs `srqfn: log(Time) = -11.16 + 1.51 log(n)`; improvement "roughly 36 at n=64 to approximately 850 at n=1024." The intercepts slightly favour the dense code, so the **entire** gain is in the slope. And the key structural trick: "the sparse structure of the coefficient matrix does not change and only its numerical values vary from one iteration to another, [so] the ordering and symbolic factorization steps need to be performed only once," worth "roughly a factor of 1/5" over 20 iterations.

**Scope caveat, important:** this is the penalized-triogram / sparse-design regime where `nnz = O(n)` while p grows with n. It argues for **adding a sparse path** to `pinball`, not for a win on the dense `rqfnb.f` shipped today — which has no ordering/symbolic phase to hoist. (A claim that `srqfn` is documented as Mehrotra with a single reused Cholesky per iteration was **refuted 0-3** in verification; don't assert it on this source.)

### Finding 5 — Globbing HAS been warm-started across a τ grid, exactly — 100× for 99 quantiles
*Confidence: high (3-0). This is the most directly actionable finding in the note.*

Chernozhukov, Fernández-Val & Melly, *Fast Algorithms for the Quantile Regression Process*, Empirical Economics 62(1):7-33, 2022 ([arXiv:1909.05782](https://arxiv.org/abs/1909.05782)).

Abstract, verbatim: "The first algorithm applies the preprocessing idea of Portnoy and Koenker (1997) but **exploits a previously estimated quantile regression to guess the sign of the residuals**. This step allows for a reduction of the effective sample size... **The first algorithm is exact**, while the second is only asymptotically equivalent."

Mechanically: because `√n(τ_j − τ_{j−1}) = O_p(1)`, the preliminary fit at `τ_{j−1}` is good enough that "while we kept a sample proportional to n^{2/3} in Algorithm 1, we can keep a sample proportional to **n^{1/2}** in Algorithm 2." Exactness is preserved by PK's own step-4 residual-sign verification loop: the estimates are "numerically equal ... to the estimates that we would obtain using the simplex or interior point algorithms. Thus, there is no statistical trade-off." **Reported 100× speedup for 99 quantiles at n=50,000, p=20.** Footnote 8 credits Thisted's 1997 comment on Portnoy & Koenker for the idea, "which has never been implemented to the best of our knowledge."

**Why this is the key finding:** §0.3 measured that warm-starting the *interior point* across τ fails. CFM warm-start the *combinatorial globbing step* across τ instead — the sign pattern, not the barrier iterate — and get 100×, exactly. The repo has `pfn` (PK Algorithm 1) but **not** the τ-warm-started Algorithm 2, and `_estimator.py:215` cold-solves each τ. That is a published, exact, large win sitting unimplemented.

### On question (2), restated
Nothing verified in this pass connects globbing or LP solvers to gradient
boosting. Combined with the companion note's negative result, the literature
position is unchanged: **no such work exists.** CFM (Finding 5) is the closest
published relative — warm-starting a combinatorial preprocessing step across a
*sequence of related fits* — and is the strongest anchor for novel work here.

---

## Part 2 — Five research directions (mine; not from the literature)

*None of this is published, implemented, or proven. Hypotheses, ordered by my
estimate of expected value.*

### D1. Round-indexed globbing: transplant CFM's warm start from the τ axis to the boosting-round axis

This is the bridge between the two halves of the question, and I think it is
the best idea in the note.

CFM's insight is not really about quantiles. It is: *when you solve a sequence
of nearby fits, the previous fit's residual signs are a good enough guess to
shrink the next problem, and a verification step keeps it exact.* Their
sequence is indexed by τ, with `√n(τ_j − τ_{j−1}) = O_p(1)` supplying the
closeness bound.

A boosting run is also a sequence of nearby fits — indexed by round `m`, with
closeness supplied by the shrinkage `η` instead: `F_m = F_{m−1} + η·h_m`, so
predictions move by at most `η ×` the leaf-value range. That is a *tighter and
more controllable* bound than CFM's stochastic `O_p(1)`, because `η` is a knob
you set rather than a rate you inherit.

Concretely, for boosting with **linear quantile base learners fit by `rqfnb`**:
carry the residual-sign vector from round `m−1`, glob the confidently-signed
points into PK pseudo-observations, fit round `m`'s base learner on the reduced
problem, and run PK's step-4 verification to restore exactness. The companion
note derived per-round margin-gated globbing independently; what CFM adds is
the missing piece — **a published exactness-preserving verification protocol**
for a warm-started glob, plus evidence it delivers 100× on a real sequence.

The leaf-co-location objection from the companion note (§Part 2b) **does not
apply here**, because a linear base learner's fitted value is a smooth function
of x, exactly as in PK's original setting. That objection was specific to trees.
This is the cleanest available path to "globbing + boosting."

**First test:** boosting with linear QR base learners, `η ∈ {0.01, 0.1}`,
measure what fraction of points stay confidently signed across rounds. If that
fraction is high and stable, the rest follows. If it collapses after a few
rounds, D1 is dead and that is worth knowing quickly.

### D2. Dual-recentred τ-homotopy for the Frisch-Newton solver

Motivated directly by the measured failure in §0.3. The failure had a specific,
addressable cause: the primal is warm-started while the dual is rebuilt cold,
producing an inconsistent pair. The research question is whether a *jointly*
recentred start — carry `(x, s, z, w)` together, rescale to a target
complementarity `μ_target` matched to τ's change, and re-derive `y` from the
recentred dual rather than from `d=1` — recovers the iteration savings.

Honest prior: **CFM's success at the globbing level is evidence this is the
harder and less promising level to attack.** IPM warm-starting is notoriously
difficult for exactly this boundary reason, and D1/Finding 5 get the same
practical win more cheaply. Worth one bounded experiment, not a campaign. Its
main value would be diagnostic — a clean characterization of *why* the QR LP
resists interior-point warm starts, which as far as I can tell nobody has
written down.

### D3. Streaming full-process: parametric programming with O(n) memory

§0.4 showed the whole-quantile-process path is blocked by an `(n, 3n)`
allocation, not by the algorithm — which touches only two columns of `dsol` at
a time. A variant that keeps a rolling 3-column window and streams breakpoints
out (or writes only `sol`, the `(p+3, nsol)` coefficient path) would make exact
whole-path solving feasible at n where it is currently impossible.

I verified the feasibility precondition directly: across all of `rqbr.f`, the
only **reads** of `dsol` are of column `lsol` (line 375, plus the
`dsol(i,lsol+1) = dsol(i,lsol)` copy at 368). The `dsol(i,1) = one` at line 531
is a pure write in the `lsol > 2` finalization block. **No stale column is ever
read**, so a rolling window is sound rather than merely plausible.

The open research question is the one that allocation is standing in for: **how
many breakpoints does the QR process actually have?** The `3n` is an upper
bound inherited from `quantreg`; the empirical growth rate of `lsol` in n and p
is the thing to measure, and I have not found it stated anywhere. If it is
`O(n log n)` or better, streaming full-process becomes a genuinely attractive
alternative to a τ grid — and unlike a grid, it returns *every* breakpoint
exactly. Compare against CFM Algorithm 2 (Finding 5), which is the incumbent to
beat for the τ-grid use case.

### D4. Extreme-τ specialization, where three effects line up

§0.2 measured IPM iteration counts rising toward the extremes (14→34, 15→38).
That is the regime where three separate things point the same way:
- The interior point gets **harder on average** (§0.2) — the U-shaped
  iteration profile costs ~1.5–2× in the tails versus the median. (Noted
  honestly: the profile is noisy and non-monotone, so this is a tendency, not
  a clean law, and one design at one (n,p).)
- Globbing gets **easier** — an extreme-τ hyperplane sits far above the bulk,
  so margins are large and the confidently-signed fraction is enormous. (The
  repo's own `local_testing/glob_huge_n.py` was built on precisely this
  reasoning at n=10⁷.)
- Boosting's gradient asymmetry becomes **informative** — at τ=0.5 the two
  pinball gradient values tie in magnitude, at τ=0.999 they differ by 1000×
  (companion note, Part 2b).

The direction: a solver path specialized for `τ → 0` or `1` that leans on
globbing to do the work the IPM finds hardest. This is also the regime with the
most practical demand (VaR, latency SLOs, extreme-weather quantiles).

### D5. *(Recorded negative — not a proposal.)* Sketched normal equations inside the IPM

`stepy` forms `A·D·Aᵀ` in `O(np²)`, which §0.1 measured at 25–32% of an
iteration. A randomized sketch (`Ã = A·S` with `S` an `n × k` sketching matrix,
`k ≪ n`) would cut that to `O(kp²)`.

I include this mainly to **argue against it**, because it is the obvious idea
and I think it is wrong. Three reasons: (a) the remaining ~70% of the iteration
is `O(np)` `dgemv` work that a sketch of the *normal equations* doesn't touch,
so Amdahl caps the win below 32% even at zero sketch cost; (b) IPM Newton
systems become progressively worse-conditioned as `d` spreads across many orders
of magnitude near the solution, which is precisely where sketching guarantees
degrade; (c) it destroys exactness, which is the entire reason to prefer this
solver over `conquer` (Finding 1). If someone wants the approximate-but-fast
regime, smoothing is the better-supported route.

**Counted honestly:** that leaves **D1, D3 and D4 as live bets**, D2 as one
bounded diagnostic experiment, and D5 as a lead I looked at and closed. I would
rather hand you three directions I believe in plus two documented dead ends
than pad to five. If a fifth live candidate is wanted, the best one is
**Gondzio multiple centrality correctors** — the genuine next rung above the
Mehrotra corrector already in `lpfnb`, and the literature search surfaced a
2020 paper applying exactly that to quantile regression at τ ∈ {0.1, 0.25, 0.5,
0.75, 0.9} with a predictor-corrector IPM baseline. I have not verified that
paper's claims, which is why it sits here as a pointer rather than a Part 1
finding.

---

## What I'd actually do, in order

1. **Fix the silent `maxit` failure** (§0.5). It is a correctness bug, it is
   cheap, and §0.3 shows it already hides wrong answers.
2. **Implement CFM Algorithm 2** (Finding 5) for multi-τ fits. Published,
   exact, 100× on 99 quantiles, and the repo already has the PK machinery.
   This is by far the best effort-to-payoff item.
3. **`dsyrk` in `stepy`** (§0.1). ~15–20%, an afternoon, no behaviour change.
4. **Fix `pfn`'s `inv(L)` and RNG** (§0.5).
5. **Then** D1 as the actual research bet.

## Caveats

Part 0's benchmarks are single-machine, single-design (Gaussian X, Gaussian
errors), and the warm-start result is at one `(n, p)`. Part 1 is
search-coverage-bounded: it cannot exclude an obscure or very recent preprint,
and Finding 2 explicitly flags that He et al.'s timing tables were not
verified in this pass. Part 2 is unimplemented speculation throughout.
