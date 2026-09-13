# diffusionkit

Estimate how fast a particle diffuses -- and whether its motion is normal or
anomalous -- from single-particle tracking (SPT) data, two ways:

- **classic** (`diffusionkit.classic`) -- fit a curve to the mean squared
  displacement. The standard method, fast, and a useful cross-check.
- **Bayesian** (`diffusionkit.bayes`) -- fit the exact likelihood of the raw
  displacements. No MSD curve is computed anywhere. Slower, and much more
  honest when data are scarce.

Both read the same localization table and write per-track results that join
on `track_id`, so the two estimators can be compared directly on the same
particles.

| Document | What's in it |
| --- | --- |
| **README.md** (this file) | Operating principles, install, the APIs, why the Bayesian side is computationally involved |
| `WORKFLOW.md` | Which call to make for the data you have |
| `FINDINGS.md` | Empirical results: what was measured, and why every default is what it is |
| `TABLES.md` | Column reference for every output table |

---

## Install

```bash
git clone <this repo> && cd diffusionkit
pip install -e ".[nested]"     # or: pip install -e .  (without anisotropy)
```

Python >=3.11. Pulls in numpy, polars, scipy, matplotlib, seaborn, jax,
numpyro, tqdm. CPU-only jax is fine -- nothing here needs a GPU.

The `nested` extra adds `jaxns`, needed only by the anisotropy workflow
(`diffusionkit.bayes.nested`). It is optional and imported lazily, so the
rest of the package works without it. Note that jaxns currently depends on
a `tfp-nightly` build, which is why it is not a core dependency.

Verify:

```bash
python -c "from diffusionkit.bayes import fit_track; print('ok')"
```

The scripts in `scripts/` also run straight from a clone without installing
-- they put the repo root on `sys.path` themselves.

---

## The 60-second example

One short track, both estimators. This is the whole argument for the
Bayesian side, in twelve lines.

```python
import polars as pl
from diffusionkit.classic import AcquisitionParams, load_tracks, compute_all_tamsd
from diffusionkit.classic.fitting import fit_normal_diffusion, n_fit_points
from diffusionkit.bayes import fit_track

params = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)
tracks = load_tracks("mobile_beads_1to200.csv", params)
track  = tracks.filter(pl.col("track_id") == 13)      # 10 frames -- a short one

# classic: build the MSD curve, fit a line to its first few lags
msd = compute_all_tamsd(track, dt_s=params.dt_s).sort("lag")
tau, y, w = msd["tau_s"].to_numpy(), msd["msd_um2"].to_numpy(), msd["n_pairs"].to_numpy()
classic = fit_normal_diffusion(tau, y, n_fit_points(len(tau)), weights=w)

# Bayesian: fit the 9 displacements directly
bayes = fit_track(track, params.dt_s, model="normal")
```

What comes back:

```
classic   D = -0.0103 +/- 0.0254 um^2/s        <- negative diffusion coefficient
Bayesian  D =  0.1004  (0.0700, 0.1440) um^2/s
          sigma_loc = 0.0142 um
```

The classic fit returns a **negative diffusion coefficient**, silently. It
is not a bug in the implementation -- a least-squares line through three
noisy, correlated MSD points has no reason to come back with a positive
slope, and nothing in the method forbids it. Ask the same track for an
exponent and it returns `alpha = -0.16`, which is not a number the physics
admits either.

The Bayesian fit cannot do this. `D` and `sigma_loc` are positive by their
prior's support and `alpha` is bounded to `(0, 2)`, so an impossible answer
is not in the parameter space to begin with -- and it reports an interval,
which on nine displacements is wide, as it should be. Its anomalous fit
says `alpha = 0.45 (0.17, 0.73)`: sub-diffusive, with the uncertainty
stated, rather than `-0.16` stated as fact. That difference follows entirely
from the operating principle below, not from more data or a better
optimizer.

Runnable version: `scripts/quickstart_single_track.py`.

---

## Terms

Used consistently throughout the code, this README, `WORKFLOW.md`,
`FINDINGS.md`, and `TABLES.md`.

| Term | Meaning |
| --- | --- |
| **track** | One particle's trajectory: `track_length` localizations on a gapless, uniform frame grid |
| **displacement** | `dx[k] = x[k+1] - x[k]`. A track has `n_disp = track_length - 1` of them |
| **lag** | A frame separation `n`; `tau = n * dt_s` is the corresponding time |
| **D** | Brownian diffusion coefficient, um^2/s. From a model with `alpha` pinned to 1 |
| **D_alpha** | Generalized diffusion coefficient, um^2/s^alpha. From a model with `alpha` free |
| **alpha** | Anomalous exponent. `<1` sub-diffusive, `1` Brownian, `>1` super-diffusive |
| **sigma_loc** | Static localization precision, um. Per-frame position uncertainty. Named `sigma` as a model parameter (`fit.params["sigma"]`), `sigma_*_um` in output columns |
| **eps** | Anisotropy fraction `(D_par - D_perp)/(D_par + D_perp)`, in `[0,1)`. `0` = isotropic |
| **psi** | Orientation of the anisotropy axis, radians in `[0, pi)` |
| **single-track regime** | A handful of tracks, examined individually |
| **population regime** | Hundreds to thousands of tracks, one row each |

Two conventions that hold everywhere: 2D MSD is `4*D_alpha*tau^alpha` (so
per-axis it is `2*D_alpha*tau^alpha`), and a `_um`/`_um2_s`/`_um2_s_alpha`
column suffix marks physical units while a bare name is dimensionless.

---

## Operating principle

Both pipelines start from the same raw material: a track's positions. They
differ in **which statistic of that track they consume**, and everything
else follows.

### Classic: summarize, then fit

Compress the track into its time-averaged MSD curve, then fit a model to
that curve.

```
MSD(tau) = 4*D*tau + b                 normal      (weighted least squares)
MSD(tau) = 4*D_alpha*tau^alpha         anomalous   (least squares in log-log space)
```

`b` is the static-localization offset, `~4*sigma_loc^2`. Fits use only the
first few lags, because MSD at large lag is computed from few, heavily
overlapping pairs and its variance grows accordingly.

Two structural consequences, neither fixable by fitting more carefully:

1. **The summary is lossy.** MSD at each lag is a *mean* over overlapping
   displacement pairs from one trajectory. Those pairs are correlated, and
   the curve keeps only the mean, discarding the correlation structure that
   carries real information about D and alpha.
2. **Nothing constrains the answer to be physical.** The fit is linear
   algebra on a curve. Negative D, negative intercepts, and alpha outside
   `(0,2)` are all reachable, and on short tracks they are reached
   regularly. The classic pipeline therefore *flags* these
   (`D_negative`, `intercept_negative`) rather than silently dropping them.

### Bayesian: skip the summary

Don't build a curve. Michalet & Berglund (2012) showed the efficient
estimator works on the raw displacement sequence, and Vestergaard et al.
(2014) wrote down its exact distribution. For fractional Brownian motion
observed with static localization noise, the displacement sequence is a
zero-mean multivariate Gaussian with a covariance you can write in closed
form:

```
dx ~ Normal(0, Sigma)          Sigma = Sigma_motion + Sigma_noise

Sigma_motion[i,j] = gamma(|i-j|),   gamma(k) = D_alpha * dt^alpha *
                                               (|k+1|^a - 2|k|^a + |k-1|^a)
Sigma_noise[i,i]   =  2*sigma_loc^2
Sigma_noise[i,i+-1] = -sigma_loc^2
```

That is the entire model. `Sigma_motion` is the autocovariance of
fractional Gaussian noise; `Sigma_noise` is what differencing an iid
per-frame error does to neighboring displacements. They add because motion
and localization error are independent.

Three things follow from this being *exact* rather than a summary:

- **alpha = 1 is exact, not approximate.** At `alpha=1`, `gamma(0) = 2*D*dt`
  and `gamma(k>=1) = 0` -- independent increments, ordinary Brownian
  motion. The normal model is a restriction of the anomalous one, so
  comparing them is a comparison of nested models, not of two separately
  derived formulas.
- **Localization noise is a parameter, not a correction.** `sigma_loc` is
  estimated jointly with D from the same likelihood. The classic pipeline
  instead subtracts an offset estimated separately, which overcorrects when
  that estimate is itself noisy.
- **Impossible answers are unreachable.** D and `sigma_loc` get LogNormal
  priors, alpha a Beta prior rescaled to `(0,2)`. Positivity and bounds are
  properties of the parameter space, so no post-hoc flag is needed.

### Bayes, in one line

```
posterior(theta | data)  proportional to  Normal(dx; 0, Sigma(theta)) * prior(theta)
```

`theta` is `(D_alpha, alpha, sigma_loc)`, or `(D, sigma_loc)` with alpha
pinned. There is no second estimator hiding anywhere: a near-flat prior
(`WEAK_*_PRIOR`) turns the same code into a maximum-likelihood fit, and an
informative prior turns it back. One model, one likelihood, one code path.

**The statistics are this simple. The computation is not** -- see
[Making it fast](#making-it-fast) below, which is where most of the code in
`diffusionkit/bayes/` actually goes.

---

## The two APIs

Deliberately separate. Same input schema, same `track_id`, different
namespaces -- so it is always obvious which estimator produced a number.

| You have | Call |
| --- | --- |
| A handful of tracks, exploratory | `diffusionkit.bayes.fit_track` |
| Hundreds to thousands of tracks | `diffusionkit.classic.fit_population` (fast cross-check) + `diffusionkit.bayes.fit_population` |
| "Is *this* track anisotropic?" | `diffusionkit.bayes.anisotropy.analyze` |

`WORKFLOW.md` works each of these through in full.

### Load data (both pipelines)

```python
from diffusionkit.classic import AcquisitionParams, load_tracks, assert_contiguous_tracks

params = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)
tracks = load_tracks("mobile_beads_1to200.csv", params)
assert_contiguous_tracks(tracks)     # both pipelines assume no frame gaps
```

Input CSV: one row per `(track_id, frame)` with `x`, `y`, `sigma_x`,
`sigma_y` in pixels. `tracks` comes back tidy and in physical units
(`x_um`, `sigma_x_um`, `t_s`, ...), with `track_length` derived internally.
Every function below takes this DataFrame or a single-track slice of it.

### Classic API -- `diffusionkit.classic`

```python
from diffusionkit.classic import fit_population

fit = fit_population(tracks, params.dt_s)

fit.per_track                # one row per track: D, alpha, quality flags
fit.ensemble                 # n_pairs-weighted MSD curve across all tracks
fit.ensemble_normal_fit      # D from the ensemble curve
fit.ensemble_anomalous_fit   # alpha from the ensemble curve
fit.mean_localization_offset_um2
```

Underneath: `compute_all_tamsd` -> `ensemble_average_msd` ->
`fit_normal_diffusion` / `fit_anomalous_diffusion`, all exposed
individually if you want the pieces.

### Bayesian API -- `diffusionkit.bayes`

Two entry points, split by regime.

```python
from diffusionkit.bayes import fit_track, fit_population

# single-track regime
fit = fit_track(track, params.dt_s, model="anomalous")
fit.params["alpha"], fit.lo["alpha"], fit.hi["alpha"]

# population regime
table = fit_population(tracks, params.dt_s, model="both")
```

| Argument | Options | Notes |
| --- | --- | --- |
| `model` | `"normal"`, `"anomalous"`, `"both"` | `"both"` fits each and joins on `track_id` |
| `prior` | `None` (default) or a prior dataclass | `None` builds an informative `sigma_loc` prior from *this track's own* measured precision -- the honest default when data are scarce |
| `method` (`fit_track`) | `"map"` (default), `"nuts"` | MAP + Laplace interval; NUTS when the posterior's *shape* matters |
| `max_batch_size` (`fit_population`) | `20` (default) | Tracks per optimizer call. See [Making it fast](#making-it-fast) |

`prior=None` anchors `sigma_loc` to the precision the tracking software
already measured for those frames -- independent information, never that
track's own MSD, so it is not smuggling the classic estimate back in. It is
rejected with `model="both"`, since the normal and anomalous models take
different prior types.

`fit_population` deliberately has no engine switch. If you want the SVI or
NUTS table for comparison, the builders are importable directly:

```python
from diffusionkit.bayes import fit_table_svi, fit_table_nuts, batched_anomalous_diffusion_model
```

They take a model function and a `prior_fn` rather than a `model=` string --
that is the layer below `fit_population`, not a different estimator.

### Anisotropy -- `diffusionkit.bayes.anisotropy`

Its own module rather than a third `model=` option, because it answers a
different *kind* of question: not "what is this track's D" but "which of two
models does this track's data prefer".

```python
from diffusionkit.bayes import anisotropy

per_track = anisotropy.analyze(tracks, params.dt_s)   # one row per trajectory
per_track.select("track_id", "track_length", "log_bf10", "log_bf10_stderr", "evidence")
```

The comparison is between a rotated diffusion tensor and a single scalar
`D`. The tensor is carried in log-Euclidean coordinates,
`Sigma = 2*dt*D_g*expm(h1*sigma_z + h2*sigma_x)`, so `(log D_g, h1, h2, log
sigma)` is unconstrained in `R^4` with positive-definiteness automatic and
isotropy at the *interior* point `h = (0,0)`. `D_par/D_perp = exp(2|h|)`,
`eps = tanh|h|`, `psi = atan2(h2,h1)/2`. A lab-frame rotation by `theta`
rotates `(h1,h2)` by `2*theta`, so an isotropic prior on `h` is exactly
invariant to the mounting angle.

Both evidences are computed by nested sampling (`bayes/nested.py`, jaxns):

```
log_BF10 = log p(data | h free) - log p(data | h = 0)
```

Nested sampling rather than an approximation, because this has to stay
correct across the whole range of track lengths. A Laplace approximation
is 2.8 nats wrong on a 100-displacement track; the Monte Carlo estimator
this replaced was only valid while tracks were too short to answer the
question anyway.

**Read `log_bf10_stderr` next to `log_bf10`.** Nested sampling returns a
stochastic evidence, and on a short track its uncertainty is larger than the
evidence itself -- which is the honest statement that four displacement
vectors cannot resolve the question. The `evidence` column says
`inconclusive (below sampler noise)` when that happens.

Whether a single track *can* answer depends on its length. Fraction reaching
strong evidence (`log_bf10 > 3`) on their own:

| track_length | true eps=0 | true eps=0.5 | true eps=0.8 |
| --- | --- | --- | --- |
| 5 | 0% | 0% | 0% |
| 20 | 0% | 3% | 33% |
| 50 | 0% | 35% | 97% |
| 200 | 2% | 98% | 100% |

Short tracks are not excluded and there is no length cap -- run them and
read the honest near-zero answer. What this workflow will *not* do is pool
tracks into an ensemble score: that is ensemble averaging, and a pooled fit
with one shared `D` invents anisotropy out of ordinary `D`-heterogeneity
once the spread reaches ~0.5 decades. Per-track inference is immune, since
every track carries its own `D`.

Nested sampling is an optional dependency: `pip install -e ".[nested]"`.

---

## Making it fast

The statistics above are a page of algebra. Nearly everything else in
`diffusionkit/bayes/` exists to make evaluating them tractable on real datasets. If you
are reading the code and wondering why it is not shorter, this section is
the answer.

### Numerical ground rules

- **float64 everywhere.** `diffusionkit/bayes/__init__.py` sets
  `jax.config.update("jax_enable_x64", True)` on import. jax defaults to
  float32, which is not enough precision for a covariance whose motion and
  noise terms can differ by orders of magnitude (a slow particle observed
  with ordinary localization error). Note this is a *process-wide* jax
  setting: importing `bayes` changes it for everything in the process.
- **Dense covariance on purpose.** `Sigma` is Toeplitz, so an O(n log n)
  solver exists. Tracks here are <=200 frames, making a dense `n_disp^2`
  matrix trivially cheap, and staying dense lets the motion and noise terms
  be built and added the same way. Not worth the complexity.
- **One covariance implementation, four uses.** `likelihood.py` is plain
  `jax.numpy` with no numpyro import, and its functions are written as
  broadcasting arithmetic rather than explicit matrix assembly. The same
  code therefore serves single-track inference, batched inference,
  simulation of ground truth, and the Bayes-factor integrand -- so the
  generative model and the inference model can never drift apart.

### The bottleneck is tracing, not linear algebra

Fitting 365 real tracks one at a time took **~17 minutes**. The linear
algebra accounts for almost none of that: each Python-level call pays a
fresh JAX trace and compile, regardless of the shape being identical to the
last call's.

The fix is to fit many tracks in one call. Tracking datasets conveniently
have many tracks of *exactly* the same length (everything that survived to
an acquisition cutoff), and a `batched_*` model wraps the same per-track
sample statements in a `numpyro.plate`:

```
group tracks by track_length  ->  one plated call per group  ->  ~9 minutes
```

The physics is untouched -- passing `(n_tracks, 1, 1)`-shaped parameters
through the same broadcasting covariance code yields a
`(n_tracks, n_disp, n_disp)` batch for free. Only the trace cost changes,
amortized across every track in the group.

That grouping is written once. `_per_track_table` selects eligible tracks,
splits them into length-homogeneous batches, optionally sub-batches those,
runs the caller's fit, and stacks the result; the three table builders and
the per-track anisotropy table all go through it and differ only in which
engine they call and which columns they return.

### Three table builders, and why the slowest is production

`inference.py` has three single-batch engines -- `fit_map`,
`sample_posterior` (NUTS), `fit_batch_svi` -- and three per-track table
builders wrapping them, all sharing one grouping loop:

| | `fit_table_map` | `fit_table_svi` | `fit_table_nuts` |
| --- | --- | --- | --- |
| Method | L-BFGS-B on numpyro's unconstrained `potential_fn`, exact JAX gradient + Hessian | Mean-field `AutoNormal` guide, Adam-optimized ELBO | Full NUTS |
| Uncertainty | Laplace, from the exact Hessian | Guide quantiles | HPDI from draws |
| 365 real tracks | ~21 min | ~9 min | far slower |
| Calibration | alpha stderr within 1-4% of NUTS | **3-10x too narrow** | reference |

Mean-field SVI assumes the parameters are independent in the posterior.
They are not -- D_alpha, alpha, and `sigma_loc` are strongly correlated,
and that correlation carries much of the real uncertainty. Throwing it away
produces intervals that look great and are wrong.

So `fit_population` runs `fit_table_map` and offers no engine switch: a
keyword that silently changed how trustworthy the intervals are (and, since
the two report different statistics, what the columns are named) is not a
convenience. `fit_table_svi` remains importable and is what the recovery
scripts compare against; `fit_table_nuts` is for when posterior shape
matters. The anisotropy workflow uses none of the three -- one nested-
sampling run yields its evidence and its `eps`/`psi` posterior together.

MAP's accuracy has a price. Its Hessian is dense over *every* free
parameter in the call at once, so cost is superlinear in batch size: 10
tracks ~6s, 20 ~8s, 40 ~27s, and 140 tracks **ran out of memory**.
`fit_table_map` therefore splits each length-group into sub-batches of
`max_batch_size` (default 20) and appends results -- trading some
amortization back for a memory ceiling that does not depend on how many
tracks share a length. (A block-diagonal Hessian via `jax.vmap` over
per-track blocks would fix this properly; not yet implemented.)

### Report in log-space

The Laplace approximation is Gaussian in numpyro's *unconstrained* space --
`log(D)` for a LogNormal-supported parameter. So the results table reports
D and `sigma_loc` as `exp(log_mean +- log_stderr)`: an asymmetric interval
in physical units, plus `log10_*` columns. Pushing that Gaussian through
`exp()` and quoting a symmetric `mean +- stderr` instead understates the
skew and can produce an interval touching zero for a strictly positive
quantity. alpha, being bounded rather than positive-scaled, keeps a
symmetric interval.

### The Bayes factor, computed efficiently

`log_BF10` needs two marginal likelihoods, each an integral over the prior.
Three choices make it cheap and stable:

- **Prior-predictive Monte Carlo, not Savage-Dickey.** `eps=0` sits at the
  boundary of a Beta support, and no continuous NUTS draw lands exactly
  there, so estimating the posterior density at that point is fragile.
  Averaging the closed-form Gaussian likelihood over prior draws needs no
  density estimation at all. This works *because* the target regime has a
  weak likelihood relative to the prior -- for long, informative tracks it
  would need bridge sampling instead.
- **Common random numbers.** `D_mean` and `sigma_loc` are nuisance
  parameters shared by H0 and H1, so the *same* prior draws are used for
  both integrals. Their Monte Carlo error largely cancels in the ratio,
  even though each marginal likelihood individually still carries it.
- **One distribution against all tracks.** For each Monte Carlo draw, a
  single `MultivariateNormal` is evaluated against every track's
  displacement vector at once by broadcasting, with `jax.vmap` stacking
  that over the `n_mc` draws. No per-track loop.

### Known cost centers

- The dense Hessian, as above -- the one real scaling limit.
- Every table builder marshals a length-group from polars into stacked
  numpy arrays (`_stack_tracks`). It is linear and cheap next to the fit
  itself, but it is a Python-level step in an otherwise vectorized path.

Plotting is deliberately *not* on this list any more: `viz` is a submodule
of each pipeline rather than a re-export, so neither `import diffusionkit`
nor `from diffusionkit import bayes` pulls in matplotlib.

---

## Layout

```
diffusionkit/
  __init__.py     load_tracks + AcquisitionParams (polars only, no jax)
  classic/        MSD-curve pipeline
    io.py           CSV -> tidy polars DataFrame in physical units
    msd.py          per-track TAMSD, n_pairs-weighted ensemble MSD
    fitting.py      normal (linear) and anomalous (log-log) curve fits
    api.py          fit_population
  bayes/          exact-likelihood pipeline
    likelihood.py   the covariance -- the only place the physics lives
    model.py        numpyro models (single-track and batched_*)
    priors.py       prior dataclasses, WEAK_* presets, per-track sigma prior
    inference.py    engines (fit_map / sample_posterior / fit_batch_svi) and
                    table builders (fit_table_map / _svi / _nuts)
    api.py          fit_track, fit_population
    nested.py       anisotropy evidence by nested sampling (optional: jaxns)
    anisotropy.py   analyze, evidence_label
scripts/          runnable studies (below)
results/          tables/ and figures/, one subfolder per script
```

Both packages also carry `simulate.py` (ground-truth generators, same
schema as `load_tracks`) and `viz.py` (plots). Every function in both is
pure -- arrays or DataFrames in, new data out, no shared mutable state --
which is what lets the two pipelines be composed for direct comparison
despite consuming entirely different statistics.

---

## Reproducing the study

```bash
python scripts/quickstart_single_track.py       # the example above

python scripts/run_msd_analysis.py              # classic, real data
python scripts/validate_localization_bias.py    # classic, simulated ground truth

python scripts/run_bayes_analysis.py            # Bayesian, real data (run classic first --
                                                #   the comparison table joins against it)
python scripts/validate_bayes_recovery.py       # Bayesian, simulated ground truth

python scripts/run_anisotropy_analysis.py       # anisotropy, real data (--limit N for a quick look)
python scripts/validate_anisotropy_recovery.py  # anisotropy, simulated ground truth
```

Each script overwrites its own subfolder under `results/` in place --
`results/` reflects the latest run of each script, not a history.
`FINDINGS.md` interprets what these produce; `TABLES.md` documents every
column.

---

## Known limits

Stated plainly; `FINDINGS.md` has the measurements behind each.

- **Sub-diffusive alpha is biased toward 1.** At true `alpha = 0.5` the
  fitted median is ~0.66. Anti-persistent fBm increments and localization
  noise are both negatively correlated at lag 1, so the two are hard to
  separate. Real limitation, not a validation artifact.
- **D_alpha degrades faster than D on short tracks.** Treat D (normal
  model) and alpha (anomalous model) as the primary per-track quantities;
  D_alpha is diagnostic.
- **Motion blur is not modeled.** Both pipelines assume `R = 0` (negligible
  exposure duty cycle), because camera exposure is not in the input schema.
- **Per-track anisotropy needs track length, not track count.** Below
  `track_length` ~20 a single trajectory cannot resolve its own anisotropy
  at any effect size, and `log_bf10` correctly returns near zero with a
  sampler uncertainty larger than itself. This is an information limit, not
  a method limit; the workflow reports it rather than pooling around it.
- **`converged` from `fit_table_map` is per sub-batch**, not per track --
  there is one optimizer call per sub-batch. Keep `max_batch_size` modest
  if per-track granularity matters.
- **Tracks must be gapless.** `assert_contiguous_tracks` enforces it; there
  is no gap-filling.
- **`classic.fit_population` needs enough tracks to form an ensemble curve.**
  Below `min_tracks_for_ensemble` (default 10) every lag is dropped and the
  ensemble fit fails with a `ZeroDivisionError` rather than a useful message.
  Lower the threshold, or use the per-track pieces directly, on small sets.

---

## References

- Michalet & Berglund, *Phys. Rev. E* **85**, 061916 (2012) -- MSD fitting
  is a lossy summary statistic; optimal estimation from displacements.
- Vestergaard, Blainey & Flyvbjerg, *Phys. Rev. E* **89**, 022726 (2014) --
  exact covariance for Brownian motion with static localization noise.
- Kepten, Bronshtein & Garini, *Phys. Rev. E* **87**, 052713 (2013) --
  fractional Gaussian noise for anomalous-exponent estimation.
