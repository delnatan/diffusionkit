# diffusionkit

Small, data-oriented tools for analyzing 2D single-particle tracks. Trajectory
data is threaded through as `polars.DataFrame`s end to end; explicit inputs,
per-observation localization errors, and inspectable fit results are the
design center. The package is a set of independent modules that share one
table format and one `Acquisition`; use any one without the others.

| module | question it answers | per track | across experiments | extra deps |
|---|---|---|---|---|
| `gridpost` (recommended) | how fast is each track, and how is `D` distributed? | exact grid posterior over `D`, information, localization floor | shared `D`; log-normal or free-shape `D` distribution per sample or replicate, with intervals; split by track length | none |
| `classic` | what do MSD fits say? | MSD curve, linear `D` and power-law `K`/alpha (the only place alpha is reported) | ensemble MSD vs lag per sample, fitted, with a cluster-bootstrap interval | none |
| `drift` | what motion do all tracks share? | | one drift field per movie, subtracted before the above | none |
| `bayes` | what does one weak track's full posterior look like? | NUTS via NumPyro, a diagnostic | not provided | `bayes` |

Everything that is not per-track is built from per-track results or tables,
never from a re-fit of merged data that forgets which track a number came from.
Tracks are combined by adding their log-likelihoods under a population model,
never by averaging their posteriors or histogramming their medians. The
napari plugin [napari-gemscape2](https://github.com/delnatan/napari-gemscape2)
runs `gridpost` on its tracks. No workflow carries a calibration claim on
experimental tracks: the intervals are checked in simulation under the model.

## Install

```bash
pip install -e .                    # numpy, polars, scipy -- classic and gridpost
pip install -e ".[bayes,plots]"      # NumPyro/JAX Bayesian pipeline and plotting modules
```

Python >=3.11. `classic`, `gridpost` and `drift` do not import JAX, NumPyro, or
plotting libraries; `gridpost.viz` and `bayes.viz` need the `plots` extra.

## Classical analysis: time-averaged and ensemble-averaged MSD

Two curves, as in any diffusion textbook, and they are kept apart:

- **time-averaged MSD (TA-MSD):** one track's mean squared displacement at
  each lag, averaged over time along that track. This section fits it per track.
- **ensemble-averaged MSD (EA-MSD):** the TA-MSDs averaged over tracks (a
  sample, a replicate, everything). See
  [Ensemble MSD](#ensemble-averaged-msd-over-experiments) below.

For Brownian motion they agree. Where they differ (non-ergodic motion, a
mixture of slow and fast tracks) the difference is informative. The same two
fits and the same options apply to either curve.

```python
from diffusionkit import Acquisition, AcquisitionParams, load_tracks
from diffusionkit.classic import MSDOptions, analyze_tracks

tracks = load_tracks(
    "mobile_beads_1to200.csv",
    AcquisitionParams(pixel_size_um=0.1043, dt_s=0.035),
)
result = analyze_tracks(
    tracks,
    Acquisition(dt_s=0.035, exposure_s=0.020),  # MSD fits are excluded when exposure_s > 0
    MSDOptions(max_lag=3, min_frames=5, localization="provided"),
)

result.fits     # two rows per track: MSD D (brownian), power-law K/alpha
result.msd      # measured MSD, pair-specific noise offset, corrected MSD
result.options # the settings that produced these results
```

Two estimators:

- **Brownian:** ordinary least squares of corrected MSD = `4 D tau`,
  through the origin. Negative D estimates are retained and flagged.
- **Power law:** nonlinear least squares of corrected MSD = `4 K tau**alpha`,
  with `K >= 0` and `0 <= alpha <= 2`. Fits at the boundaries are flagged.
  Negative corrected MSD points remain in the fit; there is no log(MSD).

The MSD fits use equal weights across the selected lags. The window is part of
the analysis: `max_lag=3` is an explicit comparison window, not an optimized
choice. Alternatively `MSDOptions(max_lag=None, lag_fraction=.3)` fits the first
fraction of each track's own curve (`window_lags`; 25-40% is the usual rule,
since long-lag MSDs rest on few, heavily overlapping pairs), so long tracks use
more of their data and short ones keep the three-lag minimum. Exactly one of the
two is set. `status="ok"` means the numerical fit passed its checks, not that
the motion model is established or the parameters are precise.

**The textbook pair.** Two plain functions fit the same `MSDCurve` (a track's
`compute_msd`, or an ensemble curve) and compose by passing a number:

```python
from diffusionkit.classic import compute_msd, fit_linear_msd, fit_loglog_msd

curve = compute_msd(track, acquisition, MSDOptions(max_lag=None, lag_fraction=.3, localization="ignore"))
lin = fit_linear_msd(curve)          # raw MSD = 4 D tau + b: D, and the offset b = 4 sigma^2 (localization_sd_um)
log = fit_loglog_msd(curve, lin.parameters["offset_um2"])   # log(MSD - b) vs log(tau): alpha, K
```

No SD columns are needed. `fit_linear_msd` uses a free intercept, which is
the classical way to read localization error off an MSD plot; a negative
intercept is flagged `nonphysical`, and it is up to you not to subtract it
(`fit_textbook` clips it at 0). `fit_loglog_msd` subtracts the offset first,
then drops lags whose corrected MSD is not positive (the row's `message`
counts them, and the dropped short lags bias alpha upward), and does not hold alpha
to [0, 2] (outside it the row is `nonphysical`). The intercept is
poorly determined on one short track and well determined on an ensemble curve;
confinement and motion blur also move it, so it is an empirical offset.
The log of a noisy, correlated mean is biased, which is why the constrained
linear-space fit above (`fit_anomalous_msd`) stays the per-track default and the
log-log line is for illustration and cross-checks. `EnsembleMSD.fit` does exactly this composition on an ensemble curve (and on every bootstrap resample); for a single track
you write the two lines.

**The MSD fits use no priors.** Supplied localization SDs are treated as
known measurement-error inputs. Their uncertainty is not inferred or
propagated. Model bounds on alpha are constraints, not a probability
distribution.

**The MSD fits report no confidence intervals.** They carry
`uncertainty_method="not_estimated"`. Correlated MSD residuals do not
justify the standard independent-residual regression error bars.

## The D posterior

```python
from diffusionkit import Acquisition, AcquisitionParams, load_tracks
from diffusionkit.gridpost import GridPostOptions, analyze_tracks

tracks = load_tracks(
    "mobile_beads_1to200.csv",
    AcquisitionParams(pixel_size_um=0.1043, dt_s=0.035),
)
result = analyze_tracks(
    tracks,
    Acquisition(dt_s=0.035, exposure_s=0.020),  # the posterior models the exposure's motion blur
    GridPostOptions(min_frames=3, level=.9, D_min_um2_s=1e-5, D_max_um2_s=10., n_D=601),
)

result.fits  # one row per track: the D posterior's median, interval, information and localization floor
```

An exact grid posterior over `D` from the Gaussian likelihood of all
consecutive displacements, using each frame's localization SD and a flat
(least-informative) prior in `ln D` over `[D_min_um2_s, D_max_um2_s]` (the
defaults shown). That range is the prior's support, so it is part of the
analysis: every gridpost function reads it from `GridPostOptions` (or takes
the grid array explicitly), and a posterior cut by an edge says so in its
row's `message`. `analyze_tracks(..., keep_likelihoods=True)` also returns
each track's log-likelihood on the grid (`result.likelihoods`), what tracks are
combined from. There is no lag window. This is the
per-track information to report for `D`: a short, uninformative track
produces a wide posterior rather than a falsely confident point estimate (see
[docs/gridpost.md](docs/gridpost.md)). The motion's shape (alpha) is not part of
`gridpost`: its per-track posterior did not hold up on short tracks (see
docs/gridpost.md, "What is reported"), and the MSD fits remain the place for a
power-law exponent.

The default range, 1e-5 to 10 um^2/s, sits well below any localization
floor, so tracks that can't be told from still fall into a low tail, read as
upper bounds, instead of into a separate D = 0 class.

The D posterior's interval is a genuine credible interval under a flat
prior, but its calibration is a simulation check under the model (see
`scripts/validate_posterior.py`), not a claim about experimental tracks. It
conditions on the reported localization SDs: noise beyond them reads as motion.
In simulation, 40 nm of unreported per-frame noise (what fast GEMs appear to
carry, likely from the spot smearing during the exposure) turns D = 0.3 into
~0.43 um^2/s.

## Distribution of D across tracks, and by track length

```python
from diffusionkit.gridpost import (analyze_tracks, by_track_length, deconvolve_tracks, lognormal_tracks,
                                   shared_D_tracks)

result = analyze_tracks(tracks, acquisition, keep_likelihoods=True)
shared = shared_D_tracks(result)         # complete pooling: one D for every track
logn = lognormal_tracks(result)          # partial pooling: ln D ~ N(mu, sigma) across tracks
logn.summary(.9)                         # median D, spread sigma, mean D, each with its interval
pop = deconvolve_tracks(result)          # partial pooling, any shape: how D is distributed
comp = by_track_length(result.likelihoods, result.options.u_D(), pop, weight="detections")
```

Each is built from every track's log-likelihood on the D grid, the tracks'
logs added under a model of the population: one D for all (`shared_D_tracks`;
when the tracks differ it lands near their mean D, with an interval that is
too narrow), a log-normal (`lognormal_tracks`, whose sigma says whether they
differ), or a smooth density of any shape (`deconvolve_tracks`). See
[docs/gridpost.md](docs/gridpost.md#combining-tracks-no-complete-and-partial-pooling).

`deconvolve_tracks` is a population-level comparator to an ensemble MSD fit,
not a replacement for the per-track posteriors: it estimates how `D` is
distributed across the tracks from each track's likelihood, without averaging
away each track's own uncertainty first. The estimate is a smooth log density
whose smoothness is chosen by the data (Laplace evidence), and `samples` are
posterior draws of the whole distribution, so any band or mass comes with an
interval (`pop.band(.68, cumulative=True)`).

`by_track_length` splits the unpooled (the tracks' own posteriors) and the
partially pooled distribution (each track's posterior under the population)
into track-length groups. Fast particles leave the focal depth within a few frames,
so short tracks come mostly from fast particles and long tracks from slow ones.
Counted per track, the groups add up to the distribution; counted per detection
(each track once per frame), they give the make-up of the spots seen in focus,
which can differ a lot (a GEM movie: 52% of tracks but 67% of detections below
0.035 um^2/s). `gridpost.viz.plot_by_track_length` draws it. See
[docs/gridpost.md](docs/gridpost.md#distribution-of-d-across-tracks-deconvolve).

## Batches of experiments: replicates and samples

```python
from diffusionkit import Acquisition, Experiment
from diffusionkit import classic, gridpost

experiments = [
    Experiment("wt_1", tracks_wt1, Acquisition(dt_s=.02, exposure_s=.01), sample="wt"),
    Experiment("wt_2", tracks_wt2, Acquisition(dt_s=.02, exposure_s=.01), sample="wt"),
    Experiment("mut_1", tracks_mut1, Acquisition(dt_s=.02, exposure_s=.01), sample="mut"),
]
```

An `Experiment` is one movie: its track table, its `Acquisition`, and the
`sample` its replicates share (default: its own name, i.e. no replicates).
Nothing is renumbered or merged: `track_id` only has to be unique within a
movie, dt and exposure may differ between movies, and every result table
carries leading `sample`, `experiment` and `track_id` columns. Tables are
validated for every experiment before any work starts. `progress(done, total)`
counts tracks over the whole batch.

**D posteriors, per track, then combined.**

```python
from concurrent.futures import ThreadPoolExecutor
from diffusionkit.gridpost import analyze_experiments, by_track_length, cdf_distance

with ThreadPoolExecutor(8) as pool:                      # one pool serves every movie
    batch = analyze_experiments(experiments, map_fn=pool.map)

batch.fits                                  # every track of every movie, labelled
pops = batch.populations("sample")          # {"wt": DeconvolvedPopulation, "mut": ...}
reps = batch.populations("experiment")      # one per movie: do the replicates agree?
together = batch.populations("all")[None]   # everything, one distribution
logn = batch.populations("experiment", model="lognormal")   # median D and spread per replicate
batch.shared_D("sample")                    # one D per sample (complete pooling)

pops["wt"].mass(0, .035)                    # draws of the mass below 0.035 um^2/s: any interval
pops["wt"].mass(0, .035) - pops["mut"].mass(0, .035)   # the difference, with its uncertainty
cdf_distance(pops["wt"], pops["mut"])       # draws of the W1 distance in ln D
sel = batch.select(sample="wt")             # a GridPosteriorAnalysis: feed it to by_track_length etc.
by_track_length(sel.likelihoods, batch.options.u_D(), pops["wt"])
```

`batch.select(...)` is the seam: it returns the same `GridPosteriorAnalysis`
a single movie does, so every function that reads one (`deconvolve_tracks`,
`by_track_length`, the plots) reads a sample or a replicate unchanged. What
is combined is the kept per-track log-likelihoods on the one `D` grid; every
movie must therefore use the same `GridPostOptions`, and the
`D_post_info_bits` are comparable. Replicates of a sample are combined by
adding their tracks' log-likelihoods in a single fit, so
each track counts once whatever movie it came from; run `by="experiment"`
for the per-replicate view. `cdf_distance` between two samples means little
alone, because two draws of the same population are still apart: compare it with the
replicate-to-replicate distances inside a sample.
`gridpost.viz.plot_populations(pops)` overlays the distributions and their bands.
A batch need not come from `analyze_experiments`: `GridPostBatch.from_analyses({name: analysis}, {name: sample})`
assembles one from per-experiment `GridPosteriorAnalysis`es obtained any way, such as posteriors restored from
disk (they must share one `GridPostOptions`), so combining never forces a refit.

### Ensemble-averaged MSD over experiments

```python
opts = classic.MSDOptions(max_lag=None, lag_fraction=.4, localization="ignore")
cbatch = classic.analyze_experiments(experiments, opts)       # per-track TA-MSD fits, labelled
ens = classic.ensemble_msd(cbatch, by="sample", n_boot=200)   # EA-MSD of each sample, with resamples kept

ens.curves                   # per group and lag: n_units, n_pairs, msd, offset, bootstrap SE
n = classic.window_lags(len(ens.curve("wt").lag), .3)   # a rule of thumb for the window (or choose it)
ens.fit(n)                   # textbook fits of the first n lags: D, localization SD, alpha, K, with _lo/_hi
ens.fit(n + 2)               # a different window refits; nothing is resampled again
ens.fit(n, alpha_points=len(ens.curve("wt").lag))   # alpha over the whole curve, D over its first n lags
```

The EA-MSD at each lag is a weighted mean of the tracks' TA-MSDs.
`weight="pairs"` (default) is the standard ensemble estimator: every squared
displacement counts once, so long tracks dominate. `weight="tracks"` is the
plain mean of the TA-MSDs, one vote per track. Supplied localization offsets
are averaged with the same weights. Building the curve and fitting it are
separate steps, and **`fit(n_points)` requires the window**: the number of
lags used changes D and alpha, so it is never a hidden default. Each fit is the
textbook pair above: a free-intercept line for D and the offset
(`offset="fit"`, default), then the log-log line on the offset-subtracted curve;
`offset="provided"` uses the supplied SDs instead. The log-log fit can take a
wider window than D (`alpha_points`): alpha needs a span of lags to show
curvature, D the first, best-measured ones; with `offset="fit"` the offset is
still the intercept over D's window. Every bootstrap resample is
fitted the same way, so the offset estimate is inside the interval.

The intervals come from resampling tracks (`resample="track"`) or whole movies
(`resample="experiment"`, the only choice that sees replicate-to-replicate
variation, and it needs several replicates per sample): overlapping pairs are
correlated, so pairs are never the resampling unit. The intervals assume
correct localization SDs and no drift shared by a movie's tracks; correct
drift first. A group pools lags by index, so it must share one dt (pool
experiments with different dt separately), and its high lags rest on fewer
tracks (`n_units`). Movies with `exposure_s > 0` have no MSD fits and so no
ensemble curve. For the constrained fit on an ensemble curve, pass
`ens.curve(group).head(n)` to `fit_anomalous_msd`.

The two views answer different questions and do not have to agree: an
ensemble MSD is a (pair-weighted) average over everything seen, the `D`
distribution is how the tracks are spread. A population that is a mixture of
slow and fast tracks has an ensemble `D` in between that belongs to neither.

## Localization and acquisition contract

The table uses `track_id`, `frame`, `x_um`, `y_um`, and normally
`sigma_x_um`, `sigma_y_um`. The sigma columns contain position standard
errors in micrometers, not spot/PSF widths. Spotsolve's `se_x`/`se_y` map to
these columns after pixel conversion; napari-gemscape2 makes that conversion.

For a pair of positions i and j, the expected noise contribution to squared
2D displacement is:

```
sigma_x[i]**2 + sigma_y[i]**2 + sigma_x[j]**2 + sigma_y[j]**2
```

The MSD correction averages those contributions over the actual pairs at
each lag. Different frames and axes may have different errors.

Assumptions: independent, zero-mean static localization errors; consecutive
unique frames; uniform positive frame interval; instantaneous positions.
Missing frames are rejected, not compressed into one time step. The
`gridpost` D posterior models continuous-exposure blur through
`Acquisition(exposure_s=...)` (Berglund's closed-form box-shutter average).
The MSD fits do not, and they are `excluded` when `exposure_s > 0`.
`exposure_s=0` is an instantaneous-observation assumption, not a statement
that your camera has zero exposure.

Omitting correction requires explicit `localization="ignore"`; missing SD
columns do not silently become zero. Calibrated zero SDs can be supplied
for noise-free synthetic data.

## Drift

```python
from diffusionkit.drift import estimate_drift, neighbour_correlation, subtract

corrected = subtract(tracks, estimate_drift(tracks, acquisition, degree=2))
neighbour_correlation(corrected)   # motion still shared between neighbours = local drift
```

The drift field every track shares (stage drift, slow tissue motion) is estimated from the tracks themselves. It is a
generalized-least-squares fit on displacements, with each track's own Brownian covariance and its D marginalized by EM,
so still spots carry the estimate without any classification. Correct before any per-track analysis when the sample
drifts: uncorrected drift of ~17 nm/frame moves a D ~ 1e-3 um^2/s population to 0.01-0.05. Local flow of ~1 nm/frame,
which no global field removes, biases D by only ~2e-5 um^2/s. See [docs/drift.md](docs/drift.md).

## Data and algorithms

Trajectory data is a `polars.DataFrame` (`track_id`, `frame`, `x_um`, `y_um`,
`sigma_x_um`, `sigma_y_um`) end to end; there is no intermediate per-track
object. Plain dataclasses hold only per-analysis metadata and results:
`Acquisition` and `Experiment` (shared), `MSDOptions`, `MSDCurve`, `MSDFit`,
`ClassicBatch`, `EnsembleMSD` (classic), `GridPostOptions`, `PosteriorD`,
`GridLikelihoods`, `SharedD`, `LogNormalPopulation`, `DeconvolvedPopulation`,
`LengthComposition`, `GridPostBatch` (gridpost) and
`Drift` (drift). Standalone
functions validate the table, compute MSD or the grid posteriors, and fit or
summarize them. `diffusionkit.io.validated_track_frame` is the single
validation boundary every per-track algorithm calls first. The batch layer sits on top: `analyze_experiments` calls the per-movie
`analyze_tracks` and labels its tables, and the combined reads take its tables
or kept likelihoods; nothing in it re-reads raw tracks. No GUI, file
writing, global JAX settings, or worker creation is part of `classic`,
`gridpost` or `drift` (a pool is passed in as `map_fn`, and owned by the caller).

See [WORKFLOW.md](WORKFLOW.md) for table examples, [TABLES.md](TABLES.md) for
output semantics, [docs/classical.md](docs/classical.md) for the MSD
estimators' equations and scope, [docs/gridpost.md](docs/gridpost.md) for
the grid posteriors', and [docs/drift.md](docs/drift.md) for drift.

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/validate_classic.py --output /tmp/classic_validation.json
python scripts/validate_posterior.py --output /tmp/posterior_validation.json
python scripts/validate_deconvolve.py --output /tmp/deconvolve_validation.json
```

Tests include batch-equals-single-movie checks, ensemble-MSD recovery with
its intervals, an independent pair-sum oracle, nonlinear objective checks,
Brownian recovery with varying localization error, and data/status contracts.
The recovery study uses an independent position-space simulator at 5, 10,
and 20 frames. See [FINDINGS.md](FINDINGS.md) for the current evidence and
limits. These checks are simulations under the model, not a calibration on
experimental tracks.

`diffusionkit.bayes` (NumPyro) fits the same exact displacement likelihood
directly, without an MSD curve, via `fit_track` -- a per-track diagnostic
tool (full NUTS posterior) for inspecting posterior shape on a short or
weakly-identified track. It models the same exposure blur as `gridpost`
(`fit_track(track, dt_s, exposure_s=...)`), but fits one iid localization SD
per track rather than using the per-frame ones. It is not a bulk production pipeline: bulk
per-track diffusivity estimation is `gridpost`'s `D`-posterior job.
MAP inference and the anisotropy (nested-sampling) workflow have been
removed: MAP conflated the unconstrained-space mode with the physical-space
posterior mode, and anisotropy was an archived research direction whose
prior-provenance issues were never resolved.
