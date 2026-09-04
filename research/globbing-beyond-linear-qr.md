# Does "Globbing" Extend Beyond Linear Quantile Regression?

**Status:** exploratory research note, not established documentation. Part
1 is a literature-verified survey (multi-source, adversarially checked).
Part 2 is original derivation with no literature source — it is a
proposal, not a documented technique. The two are kept strictly separate
below; do not cite Part 2 as if it came from a paper.

**Context:** this repo implements Portnoy & Koenker's (1997) preprocessing
technique as the `pfn` solver (see
[`docs/theory/preprocessing.md`](../docs/theory/preprocessing.md)), and
separately implements a Voronoi/optimal-quantization nonparametric
estimator (`QuantizationQuantileEstimator`). The question investigated
here: does the preprocessing/"globbing" idea have a known — or a
theoretically sound — analogue for (a) that quantization estimator, or
(b) gradient-boosted quantile regression (e.g. LightGBM)?

---

## Part 1 — Literature survey (verified, cited)

**Verdict: no.** A multi-source literature search found no paper that
extends Portnoy & Koenker's globbing/preprocessing idea to either target
family. This negative result was consistent across every primary source
examined for both families (search covered 19 fetched sources across 5
angles, with adversarial 3-vote claim verification; see caveats below).

### What globbing actually is, and why it's valid for linear models

Portnoy & Koenker (1997), *"The Gaussian Hare and the Laplacian Tortoise:
Computability of Squared-Error versus Absolute-Error Estimators,"*
Statistical Science 12(4):279–300
([full text](https://people.eecs.berkeley.edu/~jordan/sail/readings/portnoy-koenker.pdf),
[abstract](https://projecteuclid.org/journals/statistical-science/volume-12/issue-4/The-Gaussian-hare-and-the-Laplacian-tortoise--computability-of/10.1214/ss/1030037960.full)):

- The pinball-loss optimality (subgradient) condition is
  \\(g(b, w) = -\sum_i x_i^\top w \cdot \mathrm{sgn}^*(y_i - x_i^\top b, -x_i^\top w)\\)
  — each point's contribution depends only on the **sign** of its
  residual relative to the candidate hyperplane \\(b\\), never on the
  residual's magnitude.
- Points whose residual sign is confidently known (via a preliminary fit
  on a subsample plus a simultaneous confidence band) can therefore be
  replaced — "globbed" — with a single aggregated pseudo-observation:
  \\(x_K = \sum_{i \in J_K} x_i\\) paired with an arbitrarily extreme
  \\(y_K\\). This produces "exactly the same gradient condition as the
  original problem, and therefore the same solutions" (verbatim from the
  paper).
- The procedure is iterative: solve on an initial subsample of size
  \\(m = \lceil \sqrt{p}\, n^{2/3} \rceil\\), glob everything outside the
  confidence band, re-solve, check whether the globbed points' actual
  residual signs match the prediction; if a few are wrong, adjust the
  globs; if too many are wrong, widen \\(m\\) and repeat.
- Confirmed operationalized in R's `quantreg` as `rq.fit.pfn`
  ([CRAN docs](https://search.r-project.org/CRAN/refmans/quantreg/html/rq.fit.pfn.html)),
  which cites Portnoy & Koenker (1997) directly — the same lineage as this
  repo's `pfn` solver.

### Where linear QR scaling went instead (not globbing extensions, but relevant context)

- **Large-*p* (many predictors):** Koenker & Ng adapt a sparse-Cholesky
  factorization to the Frisch-Newton algorithm, explicitly contrasting
  their large-*p* target with Portnoy-Koenker's original "long, thin"
  (large-*n*, small-*p*) scope
  ([paper](http://www.econ.uiuc.edu/~roger/research/sparse/fn3.pdf)).
- **Large-*n* at massive scale:** Yang, Meng & Mahoney propose a
  randomized \\((1+\varepsilon)\\)-approximate algorithm via a low-distortion
  \\(\ell_1\\) subspace-embedding ("sketch") of the design matrix — a
  Johnson-Lindenstrauss-style approach, demonstrated on a \\(10^{10}
  \times 11\\) dataset. The paper cites Portnoy-Koenker directly as prior
  art and positions sketching as superseding, not extending, preprocessing
  ([paper](https://arxiv.org/pdf/1305.0087)).

### (a) Voronoi / optimal-quantization conditional quantile estimation

Charlier, Paindaveine & Saracco's estimator
([theory paper](https://arxiv.org/pdf/1405.2781),
[QuantifQuantile R package paper](https://journal.r-project.org/articles/RJ-2015-021/))
compresses the *covariate space* onto an \\(N\\)-point optimal quantization
grid (via the CLVQ stochastic-gradient algorithm). Critically, the fitted
value is **already piecewise-constant per Voronoi cell**, so the per-cell
pinball-loss minimization

\\[
\hat q_\alpha^{N,n}(x) = \arg\min_a \sum_i \rho_\alpha(Y_i - a)\, \mathbb{1}[\hat X_i^N = \hat x^N]
\\]

reduces — verbatim, per both papers — to "simply... the sample
\\(\alpha\\)-quantile of the \\(Y_i\\)'s whose corresponding \\(\hat X_i^N\\)
is equal to \\(\hat x^N\\)." There is no linear model, no active-set LP, and
therefore nothing for globbing (a trick for avoiding LP re-solves) to
speed up. Neither paper mentions globbing, preprocessing, or
Portnoy/Koenker/Frisch-Newton anywhere.

### (b) Gradient boosting for quantile regression

- The original LightGBM/GOSS paper (Ke et al., NeurIPS 2017,
  [link](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html))
  never mentions quantile regression, pinball loss, or Portnoy/Koenker
  anywhere in its text (confirmed by full-text search). GOSS excludes
  small-gradient instances and keeps large-gradient ones, justified purely
  by generic information-gain reasoning for GBDT splits — no
  quantile-specific argument, and no paper found formally connects it to
  Portnoy-Koenker's mechanism.
- A dedicated extreme-quantile boosting paper (Velthoen, Dombry, Cai &
  Engelke, [link](https://arxiv.org/abs/2103.00808)) sidesteps the pinball
  loss entirely for extreme \\(\tau\\), fitting a covariate-dependent
  generalized Pareto distribution via extreme value theory instead,
  explicitly because "the more natural pinball loss... degenerates in the
  extreme regime."
- A 2024 pinball-loss boosting paper restricted to linear/simple base
  learners (Bauer, Haupt & Linner,
  [link](https://www.sciencedirect.com/science/article/pii/S0167947324001117))
  contains no reference to globbing, preprocessing, coresets, subsampling,
  or GOSS; its own proposed efficiency improvement is smoothing the
  pinball loss for an adaptive learning rate, not data aggregation.
- A 2025 survey of large-scale QR methods self-declares its scope as
  distributed computing, subsampling, and online/streaming updating —
  quantization and gradient boosting are absent from its abstract and
  keywords
  ([link](https://www.mdpi.com/2227-7390/13/5/837), verified via
  Crossref/DOAJ metadata only — the publisher page blocked automated
  access, so full-body coverage can't be fully excluded).

### Caveats

This is a search-coverage-bounded negative result, not a proof of
non-existence. It cannot rule out an obscure workshop paper, blog post, or
very recent (2025–2026) preprint — especially in the fast-moving,
often non-journal-published ML/GBM literature. Two of the 24 verified
claims received 2-1 rather than unanimous 3-0 votes (the `rq.fit.pfn`
citation lineage, and the "long, thin problems" scope claim); both were
still assessed as strong evidence.

---

## Part 2 — Original derivation (not from the literature)

*Everything below is my own reasoning, developed because Part 1 found no
existing answer. None of it should be attributed to a paper or treated as
a validated result — it hasn't been implemented, tested, or reviewed.*

### (a) Voronoi / quantization estimator

**Revised conclusion (see discussion below): no meaningful globbing
analogue survives here.** An earlier draft of this section proposed
quickselect/introselect as a globbing analogue for the per-cell quantile
readout. That proposal doesn't hold up, for two compounding reasons.

First, split the estimator into its two stages and ask where globbing's
mechanism could even apply:

- **CLVQ (grid construction)** only ever sees `X` — it never touches `Y`
  or `τ`. It's unsupervised vector quantization: find `N` points
  minimizing expected distance to the `X` distribution. There is no
  pinball loss, no candidate hyperplane, no residual sign anywhere in
  this step, so globbing's actual mechanism (protect a *sign* invariant
  relative to a candidate parameter) has no foothold — this isn't a
  weaker version of the same idea, it's a different problem. If CLVQ
  itself is slow (its `O(N^2 d)` pairwise-distance step measurably
  dominates at large `N` — see the `N=2000` timing investigation in this
  repo's benchmarking work), the fix is generic vector-quantization
  acceleration (k-d/ball trees for nearest-grid-point search, mini-batch
  CLVQ) — the same literature mini-batch k-means comes from, unrelated to
  Portnoy-Koenker.
- **The per-cell quantile readout** (after the grid is fixed) is a plain
  order-statistic computation on a fixed list — `np.quantile(Y[mask],
  alpha)` in this repo's implementation. This is the step quickselect
  was proposed for. But numpy's quantile implementation already computes
  this via `np.partition` internally, not a full sort — so quickselect
  isn't a new optimization here, it's already what's happening. The
  original proposal named something already implemented rather than
  contributing anything new.

Second, and more fundamentally: **the estimator is conditional, not a
single general quantile, and that changes the character of any possible
saving.** Linear QR's globbing gets its power from there being *one
shared unknown* (β) that every data point jointly informs — aggregating
far-away points works because they're all constraining the *same* global
optimization, so their exact values stop mattering once their sign
relative to β is settled. The quantization estimator has no such
structure: conditioning is handled by *partitioning* the data into `N`
Voronoi cells up front, and each cell's quantile is an independent local
computation with no shared parameter linking it to any other cell. There
is no global object left for a globbing-style trick to protect — a point
can't be aggregated away for being "confidently uninformative to the
model," because there is no single model, only `N` separate small
problems. Any saving can only ever be local to one cell, and the first
point above shows that local saving is already provided by numpy.

**Net assessment:** the quantization estimator's own cost (CLVQ grid
construction, and the `O(nN)` Voronoi assignment repeated once per
bootstrap grid) is real and worth optimizing, but through ordinary
nearest-neighbor/clustering acceleration, not a Portnoy-Koenker-style
extension — there's no version of globbing that meaningfully transfers
here.

### (b) Gradient boosting

This is the more interesting case, and it went through several rounds of
revision below — the conclusion strengthened as the objections got
sharper. Read it in order; each subsection corrects or narrows the one
before it.

#### The pinball-loss gradient is sign-only

Gradient boosting fits a weak learner to the negative gradient of the
loss w.r.t. the current prediction, then adds it in:
\\(F_m = F_{m-1} + \eta \cdot h_m\\). For squared-error loss that gradient
is the ordinary signed residual \\(y - F(x)\\) — it encodes both which
side you're on and how far off you are. For the pinball loss it collapses
to a **two-valued step function**:

\\[
-\partial L / \partial F = \begin{cases} \tau & y > F(x) \\ \tau - 1 & y < F(x) \end{cases}
\\]

A point that misses by 0.001 and a point that misses by 1000 get the
identical pseudo-residual, as long as they're on the same side — all
magnitude information is discarded, only sign survives. This is the same
mathematical shape as the LP optimality condition Portnoy-Koenker exploit
(sign-only, magnitude-blind), independently showing up in a different
algorithm. One practical consequence: because the gradient is only two
constants, the usual gradient/hessian-based split search is mostly
*counting* how many \\(\tau\\)-points vs. \\((\tau-1)\\)-points fall in each
candidate leaf; the real quantile-fitting work happens in the **leaf
values**, which implementations typically reset to the empirical quantile
of the actual \\(y\\)'s landing there rather than trusting the coarse
pseudo-gradient average. (General pattern for quantile GBMs; not verified
against LightGBM's exact code path.)

#### The per-round margin bound

The model changes every round, so a globbing partition computed once
would go stale — a point's residual sign relative to the ensemble can
flip across rounds. But within a *single* round the perturbation is
bounded: with shrinkage \\(\eta\\) and a shallow weak learner,
\\(F_m(x_i)\\) can only move by at most roughly
\\(\eta \times (\text{leaf-value range})\\) from \\(F_{m-1}(x_i)\\). A point
whose current pinball-loss margin exceeds that bound is guaranteed not to
flip sign *this round*. That's a legitimate, re-derivable exactness
certificate — local to one round, refreshed every iteration rather than
computed once: **per-round, margin-gated globbing**.

#### GOSS, mechanically

LightGBM's Gradient-based One-Side Sampling, run at every boosting round:

1. Sort instances by \\(|\text{gradient}|\\).
2. Keep the top \\(a \times 100\%\\) (largest-gradient — the points the
   model currently fits worst; call this set \\(A\\)).
3. From the remaining \\((1-a)\times 100\%\\), randomly sample a fraction
   giving \\(b \times 100\%\\) of the total (\\(B\\)).
4. Weight every point in \\(B\\) by \\((1-a)/b\\) (weight 1 for \\(A\\))
   when computing split-finding statistics, so the gradient-sum estimate
   over \\(A \cup B\\) stays an approximately unbiased estimator of the
   full-data statistic.
5. Grow the round's tree using only \\(A \cup B\\). Typical paper values:
   \\(a=0.2, b=0.1\\) — roughly 70% of data skipped per round.

This is recomputed fresh every round, not once globally — the same
per-round-refresh architecture as the margin-gated proposal above. But
the criterion and the guarantee differ: GOSS's inclusion rule is gradient
*magnitude*; globbing's is residual *sign stability within a bounded
neighborhood of candidate models*. GOSS's protection is a statistical
reweighting that keeps an *estimate* approximately unbiased — no claim
that the resulting tree matches what the full data would produce, just
that it's empirically close. Globbing's protection is algebraic: the
reduced problem is *provably* the same optimum, checked and corrected if
the sign prediction turns out wrong. GOSS trades accuracy for speed;
exact globbing, when it applies, trades nothing.

#### A gap: leaf co-location

Linear globbing can merge far-apart points into one pseudo-observation
because a linear model's fitted value is a smooth function of \\(x\\) —
every candidate \\(\beta\\) in the trust region treats their aggregate sum
identically. A tree-based weak learner doesn't have that property: it
partitions *feature space* via splits, and which leaf a point lands in can
differ across candidate splits evaluated in the same round. So "confidently
correct sign" alone isn't enough to glob two points for a tree — they'd
also need to be guaranteed **co-located in every candidate leaf** under
this round's split search, a much stronger condition than sign stability.

#### The gap narrows: bounded leaf counts

A tree with `num_leaves = L` makes at most \\(L-1\\) splits, full stop,
regardless of whether \\(N\\) is a thousand or five hundred million.
LightGBM's default is 31; common settings run 31/63/127/255. A greedy,
gain-maximizing split search only ever places its small, fixed budget of
cuts where they separate *informative* points (points near the current
decision boundary). Points far from anywhere a split could plausibly land
— deep in the "obviously same side" interior — have no real chance of
being separated from each other this round, precisely because there
aren't enough splits available to reach them. Combined with the per-round
margin bound (sign can't flip either), that's a much more concrete case
for co-location than the previous subsection allowed.

#### Production GBMs already do a close cousin of this

Worth being upfront about before claiming novelty: LightGBM/XGBoost/CatBoost
don't search raw per-point feature values for splits — they pre-bin every
feature into a fixed number of histogram buckets (`max_bin`, default
**255** in LightGBM) and search only over bin boundaries, using
per-bin aggregated gradient/hessian sums. Points sharing a bin are already
collapsed into one aggregate statistic before split search runs. This is
one of LightGBM's three headline scaling tricks (with GOSS and Exclusive
Feature Bundling), and it exists for exactly the reason motivating this
whole section: most points are redundant for finding a good split.

#### The narrower, still-open angle: gradient-informed aggregation at extreme τ

Histogram binning aggregates by **feature-value proximity** — bins are
built from \\(x\\) alone, blind to the current residual or gradient. A
globbing-style scheme would instead aggregate by **gradient-sign-stability
/ pinball margin** — informed by \\(y\\) and the current fit, a different
axis that could combine with (not replace) histogram binning.

This is where extreme \\(\tau\\) specifically strengthens the case. At
\\(\tau=0.5\\), the two gradient values (\\(\tau\\), \\(\tau-1\\)) have equal
magnitude (\\(0.5\\), \\(-0.5\\)) — GOSS's "rank by \\(|\text{gradient}|\\)"
mechanism is uninformative, everyone ties. At \\(\tau=0.999\\), the two
magnitudes are \\(0.999\\) (the rare points above the fit) vs. \\(0.001\\)
(the vast majority below it) — wildly asymmetric. GOSS's magnitude
criterion naturally keeps nearly all the rare tail-relevant points and
aggressively drops/reweights the enormous "obviously below" bulk. So
GOSS's mechanism gets *more* justified, not less, as \\(\tau\\) moves
toward the extremes — precisely the regime (99th/99.9th percentile,
hundreds of millions of rows) where Portnoy-Koenker's original technique
also shines, and for the same underlying reason: most of a huge sample is
uninformative about an extreme tail.

Putting it together for that regime: histogram binning already collapses
feature-proximate points; the enormous "safely below" majority is both
bounded-leaf-uncuttable this round *and* individually near-worthless to
the gradient sum; and GOSS's magnitude criterion naturally aligns with
exactly that majority. A margin/sign-aware refinement on top of standard
binning — more aggressive collapsing specifically for the large-\\(N\\),
tiny-gradient, far-from-boundary bulk — looks like a real, extreme-\\(\tau\\)
motivated opening rather than a restatement of existing practice.

### Open questions for the gradient-boosting proposal

- No formal correctness proof exists for the margin/leaf-co-location
  argument, unlike Portnoy-Koenker's exact LP optimality certificate —
  the bounded-leaf-count argument makes co-location *plausible*, not
  *proven*, for a greedy, non-convex split search.
- Whether the per-round margin bound is tight enough to be practically
  useful (i.e., whether shallow, low-\\(\eta\\) boosting rounds actually
  leave a large enough "confidently-signed" fraction of points to glob) —
  untested.
- Whether globbed points still need to contribute to *leaf-value fitting*
  (not just split-finding), and if so, whether the aggregated
  pseudo-observation trick preserves correctness there the way it does for
  a single linear solve.
- How a margin/sign-aware aggregation scheme would need to interact with
  GOSS's own reweighting if both were active at once (they were derived
  independently here, not shown to compose).
- Whether the extreme-\\(\tau\\) magnitude asymmetry argument holds up
  quantitatively (i.e., how the \\(\tau\\)-vs-\\((\tau-1)\\) ratio actually
  trades off against GOSS's \\(a\\)/\\(b\\) sampling parameters) —
  untested.
- No implementation, benchmark, or correctness proof exists for any
  proposal in this section. Treat all of it as hypotheses worth testing,
  not conclusions.

### (c) Deep learning (SGD/Adam-trained neural nets)

This case is different in kind from the other two: linear QR and a single
boosting round both process *all* the relevant data in one optimization
call. SGD-trained nets never do that — they already only look at a small
random mini-batch per step. So "globbing" here can't mean "shrink one big
optimization problem"; it has to mean "reduce how often the *expensive*
part (forward + backward pass through the network) runs per point."

#### The sign-only gradient holds at the output, not deeper

The pinball loss's gradient w.r.t. the network's *output* is still just
\\(\tau\\) or \\(\tau-1\\) — same fact as the boosting case, doesn't depend
on what the model is. But backprop doesn't stop at the output:
\\(\partial L/\partial \theta = (\partial L/\partial F)\cdot(\partial F/\partial \theta)\\),
and while the first factor is a magnitude-blind constant, the second
factor (the network's own Jacobian) varies continuously per point based on
that point's activations through however many layers. The "only the sign
matters" property that let tree-boosting's split search collapse to
counting doesn't propagate to a deep parameter gradient — the chain rule
reintroduces real per-point magnitude information past the first layer.

#### The proposed mechanism: skip the pass entirely, don't just downweight it

Since the output-layer gradient is a *known constant* once the sign is
settled, a point whose sign is confidently stable doesn't need a forward
or backward pass at all this step — its contribution is already known.
This doesn't need linear globbing's "aggregate many points into one
pseudo-point" trick (that trick existed to shrink a *matrix*; here there's
no matrix to shrink, just per-point compute to skip): you just omit the
point from the step, and periodically do a cheap forward-only pass (no
backward pass needed) to re-verify it hasn't drifted back into contention.
That re-verify-and-widen-if-wrong shape is the closest of the three
families to Portnoy-Koenker's own adaptive loop.

#### Certifying "safe to skip" cheaply is the hard part

The naive Lipschitz bound for how much a weight perturbation \\(\Delta W_i\\)
can move a point's output composes multiplicatively across layers:

\\[
\|\Delta F(x)\| \lesssim \Big(\prod_{j \ne i} \|W_j\|_{op}\Big) \cdot \|\Delta W_i\|_{op} \cdot \|x\|
\\]

Each layer's worst-case stretch (\\(\|W\|_{op}\\), its largest singular
value) rarely aligns with the next layer's worst-case direction for a
*generic* perturbation, and ReLU's Lipschitz constant of 1 is only tight at
the activation boundary — both effects make the naive bound loose, and the
looseness compounds *multiplicatively across depth*. A modest 1.5x
per-layer overestimate becomes \\(1.5^{50} \approx 10^9\\) at 50 layers.
Concretely: 20 layers with operator norm ~2 each gives a "certified" bound
of \\(2^{20} \approx 10^6\times\\) the perturbation size — vacuous; nothing
clears that threshold, so nothing gets certified as skippable. (General
ML-theory knowledge, not verified via a literature search the way Part 1
was; Szegedy et al. 2013 and Virmaux & Scaman, NeurIPS 2018, are the
relevant references for the bound's looseness.)

**This is much less severe for a shallow network.** The universal
approximation theorem (Cybenko 1989, Hornik 1991) says a single hidden
layer, given enough width, suffices to approximate any continuous function
on a compact domain — and critically, the exponential-in-depth compounding
above doesn't exist for a shallow net, because there's no depth to compound
across. `Lip(F) ≤ ‖W₂‖_op · ‖W₁‖_op` for a 2-layer net is just two terms.
Width doesn't hurt this either — looseness compounds across *sequential
composition*, not across a layer's own width. Practical caveat: people use
deep nets for reasons beyond raw approximation power (parameter efficiency,
training dynamics, architectural inductive biases), so this fixes the
*bound's* tractability for shallow architectures specifically, not for an
arbitrary already-deep model.

Cheap bounds exist (naive product-of-norms) and tight bounds exist
(LipSDP, Fazlyab et al. 2019 — solves a semidefinite program for much
tighter estimates), but nothing sits in both categories at once for a
deep net: the cheap one is too loose to certify much, the tight one costs
too much to run every training step. For a 2-3 layer net the naive bound
is plausibly tight enough to be useful — untested here (see empirical
result below, which sidesteps this problem rather than solving it).

#### A valid bound is sound, never risky — the payoff is real if it applies

Worth being explicit: a valid (even loose) upper bound used as a skip
threshold can never wrongly skip a point that was about to flip sign —
looseness costs missed opportunities, not correctness. And the check
itself is cheap once you have per-layer operator norms (trackable via a
running power-iteration estimate, the same low-overhead trick spectral
normalization uses). The payoff, when it applies, is the largest of the
three families studied here: skipping a full forward+backward pass through
however many layers is a bigger unit of avoided compute than a per-cell
quantile lookup or a leaf-assignment check.

#### Existing DL-literature parallel (not verified via search)

"Spend compute on hard/boundary examples, skip or downweight easy/confident
ones" already exists under names like hard example mining (OHEM, in object
detection) and importance sampling for SGD (Katharopoulos & Fleuret). None
of those are pinball-loss-specific or built on an exactness certificate —
they're heuristic or variance-reduction arguments, closer to GOSS's flavor
than to Portnoy-Koenker's. The mechanism above, if it worked, would be a
narrower, loss-specific instantiation using a *known* constant gradient
rather than an *estimated* one. This parallel is asserted from general
knowledge, not a verified literature search.

#### Empirical test

Built a plain baseline and a heuristic margin-gated variant, both on the
full 241,600-row `sgemm` dataset (the largest in the local battery), same
60/20/20 split as every other method in this investigation, standardized
inputs/outputs, 2 hidden layers of width 64, Adam, early stopping on
validation pinball loss. Code: `local_testing/quantile_mlp_torch.py`
(baseline) and `local_testing/quantile_mlp_torch_marginated.py`
(margin-gated).

**Baseline** (full data every step) beat every other method in this whole
investigation on accuracy, at both tails:

| τ | time | pinball | corr |
|---|---|---|---|
| 0.1 | 30.3s | **1.94** | 0.998 |
| 0.99 | 41.3s | **0.38** | 0.996 |

(compare: linear `pfn` 18.3/12.1, LightGBM 3.59/1.53, `pogs` 18.3/12.1
not-converged, quantization 17.6/9.4 — full table in
`local_testing/data/compare_extreme_tau_scale.csv` and
`compare_pytorch_mlp.csv`.)

**Margin-gated variant**: refresh margins with one full forward-only pass
per epoch, train mini-batch Adam only on the bottom 20% by margin
(closest to the decision boundary) for that epoch, repeat. This is *not*
the certified version above — it's the practical fallback, heuristic, no
Lipschitz bound, explicitly labeled as such going in.

| τ | time | pinball | corr | vs. baseline |
|---|---|---|---|---|
| 0.1 | 16.8s | 32.58 | 0.646 | **17x worse** pinball, modest 45% time savings |
| 0.99 | 40.8s | 12.02 | 0.788 | **32x worse** pinball, ~0% time savings (hit the 200-epoch cap, never converged) |

**This failed, clearly, and the reason matters.** The implementation drops
the "safe" 80% from training entirely each epoch, rather than *including*
their known constant contribution the way the derivation called for.
Since every point shares the same weights, training on only the
boundary-adjacent 20% gives the network zero signal to stay correct on the
other 80% — nothing anchors it there, so it drifts, and the periodic
refresh only catches the damage after an epoch of unconstrained drift
rather than preventing it.

There's a deeper problem this surfaces, beyond just fixing the
implementation: even the "correct" version — caching the known
output-level constant (\\(\tau\\) or \\(\tau-1\\)) instead of dropping the
point — doesn't fully avoid the forward pass either. That constant is the
gradient with respect to the network's *output*. Turning it into a
gradient with respect to *parameters* (what Adam actually updates)
requires the chain rule through the network's own Jacobian at that
specific point, which depends on that point's activations. Truly skipping
the pass requires *also* caching a stale Jacobian from the point's last
real visit — reintroducing an approximation whose error grows with however
much the network has moved since, which is the same
certification/staleness problem flagged above as unresolved, not a new
one. This experiment is evidence that the naive way of dropping that
complexity (just omit the point) doesn't work, not evidence about whether
the harder, correct version would.

#### Round 2: restoring the safe-set force (empirical, successful in part)

Round 1's failure has a crisp mechanical reading: the pinball optimum is a
**balance of forces**. Each point contributes a constant-magnitude pull on
the fitted surface — weight \\(\tau\\) upward if it's above, \\(1-\tau\\)
downward if below — and the τ-quantile surface is where those pulls
cancel. Dropping the deep-margin 80% removes most of one side's force
outright; the surface drifts to the retained subset's own equilibrium
until the next refresh notices, an epoch too late. That's a systematically
biased gradient, not an approximation of the full one.

Two corrections were built and tested
(`local_testing/quantile_mlp_glob_v2.py`; single hidden layer of width 64
this round — see below for why that's now load-bearing — MAX_EPOCHS raised
to 500 so early stopping, not the epoch cap, ends every arm):

**Fix A — restore the force in expectation.** Each epoch, the active 20%
trains normally and a uniform random 10% of the safe set joins with loss
weight 10x — GOSS's reweighting applied at the SGD level. The gradient
estimator is unbiased again; every region keeps a nonzero expected
restoring force.

**Fix B — restore the force exactly: the actual globbing analogue.** For
a single-hidden-layer ReLU net
\\(F(x) = \sum_h w^{(2)}_h\,\mathrm{relu}(w^{(1)}_h \cdot x + b^{(1)}_h) + b^{(2)}\\),
a safe point's loss is linear in \\(F\\) with known slope
\\(a_i \in \{-\tau,\, 1-\tau\}\\), and with its ReLU activation pattern
\\(d_{ih} = \mathbb{1}[w^{(1)}_h \cdot x_i + b^{(1)}_h > 0]\\) frozen,
the whole safe set's loss collapses (up to an additive constant) to

\\[
L_{safe}(\theta) = \sum_h w^{(2)}_h \big( w^{(1)}_h \cdot s_h + b^{(1)}_h c_h \big) + b^{(2)} A,
\qquad
s_h = \sum_{i \in S} a_i d_{ih} x_i,\quad
c_h = \sum_{i \in S} a_i d_{ih},\quad
A = \sum_{i \in S} a_i .
\\]

That is \\(H(d{+}1){+}1\\) floats (~961 here) whose autograd gradient
equals the exact safe-set pinball gradient **while signs and patterns
hold** — verified mechanically by a float64 unit test comparing it against
direct autograd through the safe set's pinball losses (max gradient
difference 5.7e-14). The \\(s_h\\) are pseudo-observations in precisely
Portnoy-Koenker's sense (\\(x_K = \sum_{J_K} x_i\\)) — one glob per hidden
unit. This is the first construction in this investigation that genuinely
reproduces the globbing mechanism for a neural net. It's recomputed per
refresh (one forward-only pass, needed for margins anyway) and added to
every minibatch loss at \\(O(Hd)\\) cost, independent of \\(|S|\\).

Single hidden layer is what makes this exact and tiny: at depth ≥ 2 the
layer-1 gradient involves \\(D^{(2)}_i W^{(2)} D^{(1)}_i\\) cross-products,
requiring aggregates per activation-pattern *pair* (\\(H^2 d\\)) and
multiplying the exactness conditions. (This vindicates the earlier
single-layer instinct for a sharper reason than the Lipschitz-bound
argument.)

Safe/active selection for both fixes uses **raw residual distance**
\\(|y - F(x)|\\), not the loss-weighted margin round 1 used — sign-flip
risk lives in y-units (as does Portnoy-Koenker's confidence band), and at
extreme τ the weighted margin wrongly marks deep-below points as
near-boundary. Fix B trains full-data for 10 warmup epochs first (the
analogue of P-K's preliminary fit).

**Results** (full sgemm, 241,600 rows, `[64]` hidden;
`local_testing/data/quantile_mlp_glob_v2.csv`):

| arm | τ=0.1 pinball / corr / time | τ=0.99 pinball / corr / time |
|---|---|---|
| baseline | **4.50** / 0.977 / 38.9s | **0.709** / 0.967 / 35.0s |
| naive_drop (round 1) | 20.90 / 0.703 / 10.3s | 6.71 / 0.753 / 21.1s |
| reweighted (Fix A) | 5.59 / 0.958 / 34.5s | 0.800 / 0.968 / 27.2s |
| glob_exact (Fix B) | 8.24 / 0.906 / 8.0s | 0.895 / 0.958 / 30.6s |

**The collapse is fixed.** Both corrections recover correlation to
0.91–0.97 from the naive method's 0.70–0.75, and cut the pinball gap from
4.6–9.5x down to 1.13–1.83x — direct confirmation that the missing
safe-set force, not capacity or tuning, was the failure mode. Neither fix
fully matches the baseline, though: Fix A lands within 13% of baseline
pinball at τ=0.99 (22% faster) but pays an epoch-count penalty for its
gradient variance (383–401 epochs vs the baseline's 210–264); Fix B cuts
*backward* point-visits 3.4–11.8x (the transportable win — on this tiny
CPU MLP the full-data refresh forward pass eats much of the wall-clock
saving, but backward dominates for larger nets) at a 26–83% pinball cost.

**The flip-rate log is the sharpest finding**
(`data/quantile_mlp_glob_v2_refresh_log.csv`). At τ=0.99, the exactness
premise holds almost perfectly: median sign-flip rate between refreshes
0.0000, pattern flips ~0.3% — the globbed bulk sits far below the tail
surface and never moves relative to it. At τ=0.1, the surface runs
through sgemm's dense data bulk, margins are narrow, and late-training
sign-flip storms (up to 51% of the safe set in one refresh) destabilize
training — Fix B early-stopped at epoch 72 at visibly degraded quality.
**Globbing's premise holds for a neural net exactly where it holds for
linear QR: at extreme quantiles, where most of the sample is far from the
fitted surface.** The τ=0.1 instability also identifies the concrete
culprit: a *fixed* 80% glob quota, which over-globs whenever true margins
are narrow. Portnoy-Koenker never use a quota — they glob points outside
an adaptive confidence band. The direct next step is a margin-threshold
safe set (glob only points with \\(|y - F(x)|\\) exceeding a bound tied to
observed per-refresh prediction drift), which would adaptively shrink the
glob at τ=0.1 and keep it large at τ=0.99.

#### Round 3: step-fair evaluation and the adaptive band — accuracy gap closed

Round 2's residual gap had two suspected causes beyond the method itself:
its harness checked validation per *epoch* while gated arms take ~5x fewer
optimizer steps per epoch (so epoch-based patience cut them off after ~1/4
of the baseline's Adam steps), and its glob set was a *fixed 80% quota*
rather than anything resembling P-K's adaptive confidence band. Round 3
(`local_testing/quantile_mlp_glob_v3.py`) fixes both: every arm checks
validation every 142 optimizer steps with patience in checks (step-fair),
and a new `glob_band` arm globs only points satisfying

\\[ |y_i - F(x_i)| > C \cdot q_{0.995}\big(|F_t(x) - F_{t-\Delta}(x)|\big), \quad C = 3, \\]

i.e. "residual larger than 3x the observed per-refresh-window prediction
drift" — a point that can't plausibly cross the surface before the next
refresh. Refresh every 150 steps for glob arms.

**Results** (`local_testing/data/quantile_mlp_glob_v3.csv`):

| arm | τ=0.1 pinball / corr | τ=0.99 pinball / corr |
|---|---|---|
| baseline | 4.26 / 0.976 | 0.816 / 0.966 |
| reweighted (Fix A) | 5.22 / 0.971 | **0.760** / 0.967 |
| glob_quota (fixed 80%) | 10.32 / 0.865 | 0.897 / 0.962 |
| glob_band (adaptive) | 4.49 / **0.979** | **0.663** / **0.973** |

Three findings:

1. **The accuracy problem is solved by the band.** `glob_band` matches
   baseline at τ=0.1 (+5% pinball, *better* correlation) and **beats it by
   19% at τ=0.99** at the same step budget. Its certificate is sound in
   practice: mean sign-flip rate among globbed points was 0.00000 at both
   taus (max 0.0003). Step-fairness also rehabilitates Fix A, which now
   matches baseline (and beats it at τ=0.99) — most of Round 2's
   reweighted gap was the epoch-based referee, not the method.

2. **The τ=0.99 improvement over full-data SGD suggests the glob acts as
   variance reduction**: the globbed points contribute an *exact,
   deterministic* full-sum force to every step (an SVRG-flavored control
   variate) while minibatches concentrate on the informative boundary
   region. The glob didn't just avoid hurting — it helped.

3. **The compute win is not yet realized, and the refresh log says why.**
   The sound band only certified ~12–14% of points globbable
   (active_frac ≈ 0.86–0.88 throughout) — far short of the quota's 80%,
   which round-tripped through sign-flip storms up to 72% per refresh and
   still failed (pinball 10.3 at τ=0.1) even step-fairly. The binding
   constraint is the *global* drift threshold: \\(q_{0.995}\\) of drift is
   set by a few volatile heavy-tail points whose predictions swing every
   window, inflating θ for everyone. P-K's band is per-point
   (covariance-scaled in \\(x\\)); the analogous fix is a **per-point
   drift estimate** (glob iff \\(|y_i - F(x_i)| > c \cdot d_i\\) with
   \\(d_i\\) that point's own drift scale), which should certify a far
   larger glob — especially the deep-bulk majority at extreme τ — plus a
   refresh cadence chosen so the full-data forward pass amortizes against
   the certified savings.

#### Width-for-depth: does the single-layer restriction actually cost anything?

The exact glob needs a single hidden layer; Round 1's 2-layer baseline
beat every 1-layer run by ~2x. Since universal approximation lets width
substitute for depth — and the glob's aggregates grow only *linearly* in
width (\\(H(d{+}1){+}1\\); ~16k floats even at H=1024) versus the
\\(H^2 d\\) pattern-pair blowup depth 2 would force — the natural question
is whether widening closes the depth gap.
(`local_testing/quantile_mlp_width_sweep.py`; step-fair harness; the
2-layer reference rerun under the same harness; L_safe unit test passes
at every width.)

| arch (params) | τ=0.1 baseline / glob_band | τ=0.99 baseline / glob_band |
|---|---|---|
| [64, 64] (5.2k) | 1.69 / — | 0.361 / — |
| [256] (4.1k) | 3.32 / 3.16 | 0.494 / 0.519 |
| [512] (8.2k) | 2.62 / 2.72 | 0.389 / 0.392 |
| [1024] (16.4k) | 2.01 / 2.03 | 0.394 / 0.448 |

**At τ=0.99, width closes the depth gap almost completely by H=512**
(0.389 vs 0.361, within 8%, then saturates). At τ=0.1 the gap shrinks a
steady ~22% per width doubling (still 19% short at H=1024; extrapolation
says ~H=2048 matches) — so width substitutes for depth here at roughly
2–4x the parameter cost, cheapest exactly in the extreme-τ regime this
whole investigation targets. glob_band holds accuracy parity with its own
baseline at every width (±5% at 256/512; a modest lag at 1024/τ=0.99),
with sign-flip rates of exactly 0.0000 throughout — the certificate scales.

One caution from the refresh log: the certified glob *shrinks* with width
(active_frac 0.92 → 0.97 from H=256 to H=1024) — more parameters moving
per refresh window inflates the global drift quantile — which further
sharpens the case for per-point drift scales as the next lever.

#### The Lipschitz certificate, measured (and what it revealed)

At depth 1 the certified approach becomes implementable: \\(W_2\\) and
\\(\Delta W_2\\) are vectors (operator norm = Euclidean norm, exact), only
\\(\|\Delta W_1\|_{op}\\) needs a spectral norm, and the circularity
objection ("can't bound the next update without the gradient you're
skipping") dissolves by running the certificate *a posteriori*: cache
\\(\theta_{ref}\\), compute exact drift norms from \\(\theta_t -
\theta_{ref}\\) each step (~3000x cheaper than a minibatch), refresh when
the weakest globbed margin's budget is exhausted. A diagnostic run
(`local_testing/quantile_mlp_lipschitz_diag.py`) trained exactly like
Round 3's glob_band but retrospectively evaluated, at every refresh, what
each certificate variant would have certified under the drift that
actually occurred:

| certified fraction (mean, sgemm H=64) | Adam τ=0.1 | Adam τ=0.99 |
|---|---|---|
| global drift band (round 3, actual) | 0.14 | 0.12 |
| plain Lipschitz (exact spectral) | 0.03 | 0.04 |
| hierarchical (pattern-routed) | 0.01 | 0.01 |
| **oracle** (margin > actual \\(|\Delta F|\\)) | **0.87** | **0.96** |

Three conclusions:

1. **The prize is much bigger than anything we've claimed**: under Adam's
   real drift, 87–96% of points were *actually* safe over each window.
   Every method in play is leaving most of that on the table.
2. **The worst-case bound is the wrong tool under Adam**: median
   looseness 56–80x (plain) / 16–22x (hierarchical). The reason is
   specific: Adam's normalized updates keep every coordinate moving at
   ~lr scale forever, so parameter-space drift norms are large, but the
   motions mostly *cancel in function space* (diffusive oscillation) —
   which a worst-case-alignment bound must ignore. The Lipschitz
   certificate under-certifies the global band it was meant to replace.
3. **The optimizer knob works exactly as predicted — and exposes the real
   trade**: under SGD + momentum + StepLR decay, even the plain bound
   certified 97–99%... because the decayed optimizer had nearly stopped
   moving, and the fit underperformed Adam's by 2.5–3.3x pinball
   (this SGD config underfit; its certification ease is an upper bound,
   not evidence that a well-tuned SGD certifies as easily). A certificate
   is precisely a proof that the model is no longer learning much about
   those points — certification-friendliness and optimization progress
   are the same coin, spent either way.

The instructive irony: Portnoy-Koenker themselves never used a worst-case
bound — their band is a *statistical* confidence band from a preliminary
fit, backed by a cheap verify-and-correct loop. Our architecture already
has the verify-and-correct loop (the refresh). So the faithful analogue —
and the pragmatic path to the 87–96% oracle — is a **per-point empirical
drift scale** (e.g. an EMA of each point's observed \\(|\Delta F(x_i)|\\)
per window, with a modest safety factor), not a worst-case certificate:
per-point like the oracle, adaptive like P-K's band, and safety-netted by
the refresh check rather than by a proof.

#### Optimizer round 2: the failed SGD, autopsied and fixed

The first SGD config failed for two identifiable reasons, both fixed and
retested (same diagnostic harness): (1) **timetable decay** — StepLR
halved unconditionally every 5000 steps, ~8–10 halvings per run,
strangling learning on a clock; replaced with ReduceLROnPlateau (halve
only when val pinball stalls 5 checks). (2) **One LR for all τ** — the
pinball gradient's average magnitude scales \\(\sim 2\tau(1-\tau)\\), 9x
smaller at τ=0.99 than τ=0.1; Adam's normalization hides this, plain SGD
doesn't. Fixed with a per-τ LR probe (short grid from a fixed init),
which picked lr=0.3 at τ=0.1 and lr=1.0 at τ=0.99 — 30–100x the failed
config's 0.01, in exactly the predicted direction. An `adam_plateau`
config (Adam + the same plateau decay) tests whether Adam's perpetual
motion was self-inflicted by never decaying.

| config | τ=0.1 pinball | τ=0.99 pinball | late certified: plain / hier |
|---|---|---|---|
| adam_const | 4.49 | 0.663 | 0.03 / 0.02 — 0.02 / 0.02 |
| adam_plateau | 4.56 | 0.644 | 0.87 / 0.95 — 0.74 / 0.82 |
| sgd (failed, ref) | 11.12 | 2.19 | (0.99+, but underfit) |
| **sgd_tuned** | **3.00** | **0.587** | 0.74 / 0.95 — 0.82 / 0.93 |

Two conclusions, one expected and one not:

- **Adam's perpetual motion was self-inflicted.** Plateau decay alone
  takes the *sound Lipschitz certificate* from ~3% to 74–87% certified
  late in training, at full accuracy — the bound is exactly as loose as
  before (median ~50–100x), but drift shrinks by orders of magnitude once
  the LR decays, and margins clear the bound anyway. Caveat: the big
  certified fractions arrive only after the LR has collapsed toward
  min_lr, i.e. once the model is nearly frozen.
- **Tuned SGD is the best fit of the entire investigation at this
  architecture** — pinball 3.00 at τ=0.1 (33% better than any Adam run at
  H=64) and 0.587 at τ=0.99 (11% better) — *and* it certifies 74–95%
  late **while still learning** (late LR a healthy 1.6e-3–1.2e-2, not
  collapsed). SGD's drift tracks gradient magnitude, which decays
  naturally near the optimum, so certification-friendliness and
  continued progress turn out not to be the same coin after all — that
  trade is an *Adam* artifact, not a law. The hierarchical bound is also
  intrinsically tighter under SGD (median looseness ~9x vs ~16–24x under
  Adam): SGD's updates are less adversarially aligned with the bound's
  worst case.

Net: the certified trust-region glob is viable after all — with tuned,
plateau-decayed SGD as the optimizer, the sound per-point Lipschitz
certificate (hierarchical variant) certifies ~93–95% of points late in
training at the best accuracy yet measured, within ~5 points of the
oracle ceiling.

#### v4: the cash-in run — first realized speedup, at the extreme quantile

`local_testing/quantile_mlp_glob_v4.py` converts the certified glob into
actual training speed. Three tuned-SGD configs: `sgd_base` (full-data
B=1024, no glob), `glob_fullB` (certified glob, B=1024), `glob_smallB`
(certified glob, B tracking the active fraction). Verdict metric:
wall-clock and backward point-visits to FIRST REACH `sgd_base`'s best
validation loss, from per-val-check curves.

Two scheme iterations were needed, both instructive failures:
(1) capping per-unit drift with *global* norms (\\(\|\Delta w^{(1)}_h\| \le
\|\Delta W^{(1)}\|_F\\), ~\\(\sqrt{H}\\) loose) collapsed the certified
glob to ~0–10% — per-unit allowances are mandatory; (2) per-step
drift-triggered refresh thrashed against momentum (drift is *ballistic*
under momentum, and oscillation-cancelled windows produce tiny allowances
that trip instantly — 30-step windows, 15x slowdown). Final scheme:
fixed windows sized so the refresh forward pass stays ~20% of window
compute (~250–300 steps at B=1024), per-unit allowances forecast as
\\(\kappa \times\\) EMA of per-step drift rates \\(\times\\) window
length, and P-K-style *verification* at each refresh (count actual sign
flips among globbed points) as the safety net instead of hard triggers.

**Results** (sgemm, H=64; verified flip rates 0.00000–0.00017):

- **τ=0.99: realized speedup, 1.6–2.3x wall-clock and 2.3x backward
  point-visits** to reach `sgd_base`'s best validation loss (9.2s /
  8.7M visits for glob_fullB vs 21.0s / 19.9M), *plus* 12–13% better
  final test pinball (0.590–0.599 vs 0.679). Reproduced across both glob
  configs and consistent with the earlier banded-glob run (0.587), so
  not trajectory luck.
- **τ=0.1: no speed win** — the glob configs converge to slightly better
  final quality (2.96–3.34) but never reach the baseline's best val
  faster. Consistent with everything upstream: the τ=0.1 surface runs
  through the dense data bulk, margins are narrow, the certified glob is
  small until late, and the refresh overhead is never repaid.
- **Attribution: the speedup is variance reduction, not cheaper steps.**
  The active fraction only falls below ~0.9 in the mid-game and reaches
  ~0.46 in the endgame — by the time batches actually shrink, the
  quality target has already been hit. The τ=0.99 win comes from the
  exact `L_safe` force making each step's gradient better (fewer steps to
  target), with the batch-shrink lever still largely unexercised. The
  compute story for batch shrinking therefore lives in the *post-parity
  tail* (long fine-tuning at high certified fractions) or at larger n,
  where the certified glob is large from earlier on.

Standing conclusion for the whole Part 2(c) line: **globbing for neural
quantile regression is real, exact-in-expectation, accuracy-positive, and
— at extreme quantiles — now measurably faster**, with the same
regime-dependence as Portnoy-Koenker's original: it pays where most of
the sample is far from the fitted surface.

#### The n=10,000,000 test: certificate scales, economics don't (yet)

`local_testing/glob_huge_n.py` ran the v4 trainer at n=10M on the
synthetic scenarios with closed-form oracles (τ=0.99, chunked refresh so
no n×H tensor materializes). Findings:

- **The certificate scales perfectly**: at 10M rows the glob held for
  2,150-step windows with a measured sign-flip rate of exactly 0.0, and
  both configs reached oracle-level excess loss (4–6e-5 above the true
  conditional quantile's pinball risk).
- **No speed win, for a structural reason worth recording**: SGD's step
  count to convergence is driven by the problem's difficulty, not by n —
  the d=2 scenarios converge in 3.5k–9k steps whether n is 145k or 10M
  (heavy_tailed, whose quantile surface is linear, converged *within
  warmup*). A 10M-point refresh pass costs the equivalent of ~3,000
  batches, and a run that only lasts 7–9k steps can never amortize even
  three of them: glob wall-clock 25.7s vs baseline 13.1s on `nonlinear`.
  Huge n alone does not create the globbing payoff regime; what does is
  a large *step count relative to refresh cost* (hard/high-dimensional
  tasks, long fine-tuning) — or a refresh that is o(n).
- **The o(n) refresh is the concrete next design**, and the flip-rate
  data says it's within reach: with measured flips at exactly zero, the
  frozen aggregates remain *valid* across refreshes — a refresh need only
  (1) verify on a subsample and (2) re-band, moving the few points that
  cross the band between glob and active with incremental aggregate
  updates (add/subtract their individual contributions). That makes
  refresh cost O(subsample + moved points) instead of O(n), which is
  exactly the piece P-K's LP setting got for free from its
  check-and-adjust step.

### Open questions for the deep-learning proposal

- ~~Whether a version that injects the excluded points' contributions
  would recover the baseline's accuracy~~ — **answered by Rounds 2–3**:
  yes, fully. With an adaptive drift band the exact-aggregate glob matches
  baseline at τ=0.1 and beats it at τ=0.99 (and the unbiased-sampling fix
  also matches baseline under step-fair evaluation).
- ~~Replace the fixed quota with an adaptive margin threshold~~ —
  **answered by Round 3**: the global-drift band restores full accuracy
  with a near-perfect flip certificate, but only certifies ~13% globbable.
- Per-point drift scales (the true analogue of P-K's covariance-scaled
  band) to grow the certified glob toward the extreme-τ bulk, plus
  refresh-cadence tuning so the full-data forward pass amortizes — the
  remaining path from "accuracy parity" to an actual compute win.
- Whether the variance-reduction effect (glob_band beating full-data SGD
  at τ=0.99) is robust across datasets/seeds, and whether it connects
  formally to SVRG-style control variates — if so, the glob may be
  valuable even at zero compute savings.
- The cached-Jacobian question from Round 1 is now moot for single-hidden-
  layer nets — \\(L_{safe}\\) *is* the exact cached contribution, no
  staleness — but reopens for depth ≥ 2, where the pattern-pair aggregate
  blowup (\\(H^2 d\\)) is untested.
- No literature search was run on the deep-learning case specifically
  (unlike Parts 1(a)/(b)), so "hard example mining" being the closest
  existing analogue is asserted, not verified — and the same caveat now
  applies to the frozen-pattern aggregate construction itself: it may
  exist in the local-linearity/NTK or lazy-training literature under
  another name.
