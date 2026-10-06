# diffusionkit

Small, data-oriented tools for analyzing 2D single-particle tracks. Trajectory
data is threaded through as `polars.DataFrame`s end to end; explicit inputs,
per-observation localization errors, and inspectable fit results are the
design center. Three independent analysis paths are supported:

- `diffusionkit.gridpost`, the recommended one: an exact-likelihood grid
  posterior over `D` per track, the distribution of `D` across tracks, and
  that distribution split by track length;
- `diffusionkit.classic`: linear MSD fits, D and the power-law exponent alpha
  (the only place alpha is reported);
- `diffusionkit.bayes`: per-track NUTS via NumPyro, as a diagnostic.

`diffusionkit.drift` estimates the drift every track shares, from the tracks
themselves, for correction before any of them. The napari plugin
[napari-gemscape2](https://github.com/delnatan/napari-gemscape2) runs
`gridpost` on its tracks. No workflow carries a calibration claim on
experimental tracks: the intervals are checked in simulation under the model.

## Install

```bash
pip install -e .                    # numpy, polars, scipy -- classic and gridpost
pip install -e ".[bayes,plots]"      # NumPyro/JAX Bayesian pipeline and plotting modules
```

Python >=3.11. `classic`, `gridpost` and `drift` do not import JAX, NumPyro, or
plotting libraries; `gridpost.viz` and `bayes.viz` need the `plots` extra.

## Classical analysis

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

The MSD fits use equal weights across the selected lags. `max_lag=3` is an explicit
comparison window, not an optimized choice or an accuracy guarantee.
`status="ok"` means the numerical fit passed its checks, not that the motion
model is established or the parameters are precise.

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
row's `message`. `analyze_tracks(..., keep_posteriors=True)` also returns
each track's full log posterior (`result.posteriors`). There is no lag window. This is the
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
from diffusionkit.gridpost import analyze_tracks, by_track_length, deconvolve_tracks

result = analyze_tracks(tracks, acquisition, keep_posteriors=True)
pop = deconvolve_tracks(result)          # how D is distributed across the tracks
comp = by_track_length(result.posteriors, result.options.u_D(), pop, weight="detections")
```

`deconvolve_tracks` is a population-level comparator to an ensemble MSD fit,
not a replacement for the per-track posteriors: it estimates how `D` is
distributed across the tracks from each track's likelihood, without averaging
away each track's own uncertainty first. The estimate is a smooth log density
whose smoothness is chosen by the data (Laplace evidence), and `samples` are
posterior draws of the whole distribution, so any band or mass comes with an
interval (`pop.band(.68, cumulative=True)`).

`by_track_length` splits the pooled and the deconvolved distribution into
track-length groups. Fast particles leave the focal depth within a few frames,
so short tracks come mostly from fast particles and long tracks from slow ones.
Counted per track, the groups add up to the distribution; counted per detection
(each track once per frame), they give the make-up of the spots seen in focus,
which can differ a lot (a GEM movie: 52% of tracks but 67% of detections below
0.035 um^2/s). `gridpost.viz.plot_by_track_length` draws it. See
[docs/gridpost.md](docs/gridpost.md#distribution-of-d-across-tracks-deconvolve).

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
`Acquisition`, `MSDOptions`, `MSDCurve`, `MSDFit` (classic),
`GridPostOptions`, `PosteriorD`, `PopulationDistribution`, `LengthComposition`
(gridpost) and `Drift` (drift). Standalone
functions validate the table, compute MSD or the grid posteriors, and fit or
summarize them. `diffusionkit.io.validated_track_frame` is the single
validation boundary every per-track algorithm calls first. No GUI, file
writing, global JAX settings, or worker creation is part of `classic`,
`gridpost` or `drift`.

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

Tests include an independent pair-sum oracle, nonlinear objective checks,
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
