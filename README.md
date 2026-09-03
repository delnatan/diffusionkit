# diffusionkit

Estimate how fast a particle diffuses -- and whether its motion is normal or
anomalous -- from single-particle tracking (SPT) data, two ways:

- **classic** (`analysis/`) -- fit a curve to the mean squared displacement.
  The standard method, fast, and a useful cross-check.
- **Bayesian** (`bayes/`) -- fit the exact likelihood of the raw
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
pip install -e .
```

Python >=3.11. Pulls in numpy, polars, scipy, matplotlib, seaborn, jax,
numpyro, tqdm. CPU-only jax is fine -- nothing here needs a GPU.

Verify:

```bash
python -c "from bayes import fit_track; print('ok')"
```

---

## The 60-second example

One short track, both estimators. This is the whole argument for the
Bayesian side, in twelve lines.

```python
import polars as pl
from analysis import AcquisitionParams, load_tracks, compute_all_tamsd
from analysis.fitting import fit_normal_diffusion, n_fit_points
from bayes import fit_track

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
`bayes/` actually goes.

---

## The two APIs

Deliberately separate. Same input schema, same `track_id`, different
namespaces -- so it is always obvious which estimator produced a number.

| You have | Call |
| --- | --- |
| A handful of tracks, exploratory | `bayes.fit_track` |
| Hundreds to thousands of tracks | `analysis.fit_population` (fast cross-check) + `bayes.fit_population` |
| Short tracks, orientation arbitrary | `bayes.anisotropy.analyze` |

`WORKFLOW.md` works each of these through in full.

### Load data (both pipelines)

```python
from analysis import AcquisitionParams, load_tracks, assert_contiguous_tracks

params = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)
tracks = load_tracks("mobile_beads_1to200.csv", params)
assert_contiguous_tracks(tracks)     # both pipelines assume no frame gaps
```

Input CSV: one row per `(track_id, frame)` with `x`, `y`, `sigma_x`,
`sigma_y` in pixels. `tracks` comes back tidy and in physical units
(`x_um`, `sigma_x_um`, `t_s`, ...), with `track_length` derived internally.
Every function below takes this DataFrame or a single-track slice of it.

### Classic API -- `analysis`

```python
from analysis import fit_population

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

### Bayesian API -- `bayes`

Two entry points, split by regime.

```python
from bayes import fit_track, fit_population

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
| `engine` (`fit_population`) | `"map"` (default), `"svi"` | See [Making it fast](#making-it-fast) |

`prior=None` anchors `sigma_loc` to the precision the tracking software
already measured for those frames -- independent information, never that
track's own MSD, so it is not smuggling the classic estimate back in.

### Anisotropy -- `bayes.anisotropy`

Kept in its own module rather than added as a third `model=` option,
because it answers a different *kind* of question.

```python
from bayes import anisotropy

result = anisotropy.analyze(tracks, params.dt_s, min_track_length=5, max_track_length=10)
result.per_track   # log_bf10 (the detector) + eps/psi posterior (descriptive)
result.ensemble    # sum_log_bf10, optionally grouped by label_col
```

The model generalizes the Brownian one to a rotated diffusion tensor with
principal diffusivities `D_mean*(1 +- eps)` at angle `psi`; `eps=0` reduces
to it exactly, so H0 is nested in H1.

At `track_length` 5-10 there are only 4-9 displacement vectors, and any
continuous estimate of `eps` from that sits above a large sampling-noise
floor -- a genuinely isotropic track often looks elongated by chance. So
`eps` is reported as an honestly wide descriptive interval, and the
**Bayes factor is the detector**:

```
log_BF10 = log p(data | eps free) - log p(data | eps = 0)
```

Individual short tracks come back inconclusive almost every time -- that is
correct, not a failure. Evidence accumulates by summing `log_bf10` across
tracks that plausibly share the behavior (group by any column on `tracks`
via `label_col=`), and `anisotropy.null_calibration` turns that sum into a
p-value against a matched-composition isotropic null. Calibration costs
~100 simulated replicate datasets, so it is a separate call.

---

## Making it fast

The statistics above are a page of algebra. Nearly everything else in
`bayes/` exists to make evaluating them tractable on real datasets. If you
are reading the code and wondering why it is not shorter, this section is
the answer.

### Numerical ground rules

- **float64 everywhere.** `bayes/__init__.py` sets
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

### Two engines, and why the default is the slower one

| | `engine="map"` (default) | `engine="svi"` |
| --- | --- | --- |
| Method | L-BFGS-B on numpyro's own unconstrained `potential_fn`, exact JAX gradient + Hessian | Mean-field `AutoNormal` guide, Adam-optimized ELBO |
| Uncertainty | Laplace, from the exact Hessian | Guide quantiles |
| 365 real tracks | ~21 min | ~9 min |
| Calibration | alpha stderr within 1-4% of NUTS | **3-10x too narrow** |

Mean-field SVI assumes the parameters are independent in the posterior.
They are not -- D_alpha, alpha, and `sigma_loc` are strongly correlated,
and that correlation carries much of the real uncertainty. Throwing it away
produces intervals that look great and are wrong. So MAP is the production
default and SVI is the documented escape valve for when throughput is the
binding constraint. Same call, one keyword.

MAP's accuracy has a price. Its Hessian is dense over *every* free
parameter in the call at once, so cost is superlinear in batch size: 10
tracks ~6s, 20 ~8s, 40 ~27s, and 140 tracks **ran out of memory**.
`fit_batch_map` therefore splits each length-group into sub-batches of
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

- `_stack_tracks` marshals a length-group into arrays with one polars
  filter per `track_id`. It is a Python loop, and on large groups it is a
  measurable fraction of wall time. `partition_by` would be faster.
- The dense Hessian, as above.
- `analysis/__init__.py` and `bayes/__init__.py` both import their `viz`
  module, which pulls in matplotlib and seaborn at import time.

---

## Layout

```
analysis/     classic MSD pipeline
  io.py         CSV -> tidy polars DataFrame in physical units
  msd.py        per-track TAMSD, n_pairs-weighted ensemble MSD
  fitting.py    normal (linear) and anomalous (log-log) curve fits
  api.py        fit_population
bayes/        exact-likelihood Bayesian pipeline
  likelihood.py the covariance -- the only place the physics lives
  model.py      numpyro models (single-track and batched_*)
  priors.py     prior dataclasses, WEAK_* presets, per-track sigma prior
  inference.py  fit_map, sample_posterior, fit_batch_map, fit_batch_svi
  api.py        fit_track, fit_population
  bayes_factor.py / anisotropy.py   log BF10, analyze, null_calibration
scripts/      runnable studies (below)
results/      tables/ and figures/, one subfolder per script
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

python scripts/run_anisotropy_analysis.py       # anisotropy, real data
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
- **Per-track anisotropy sensitivity is limited by design** at
  `track_length` 5-10. Only the population-level sum is a detector.
- **`converged` from `fit_batch_map` is per sub-batch**, not per track --
  there is one optimizer call per sub-batch. Keep `max_batch_size` modest
  if per-track granularity matters.
- **Tracks must be gapless.** `assert_contiguous_tracks` enforces it; there
  is no gap-filling.

---

## References

- Michalet & Berglund, *Phys. Rev. E* **85**, 061916 (2012) -- MSD fitting
  is a lossy summary statistic; optimal estimation from displacements.
- Vestergaard, Blainey & Flyvbjerg, *Phys. Rev. E* **89**, 022726 (2014) --
  exact covariance for Brownian motion with static localization noise.
- Kepten, Bronshtein & Garini, *Phys. Rev. E* **87**, 052713 (2013) --
  fractional Gaussian noise for anomalous-exponent estimation.
