# diffusionkit

Small, data-oriented tools for analyzing 2D single-particle tracks. Trajectory
data is threaded through as `polars.DataFrame`s end to end; explicit inputs,
per-observation localization errors, and inspectable fit results are the
design center. Three independent analysis paths are supported: classical
linear MSD fits (`diffusionkit.classic`), exact-likelihood grid posteriors
over `D` and `alpha` (`diffusionkit.gridpost`, the recommended per-track
information to extract), and Bayesian NUTS via NumPyro for per-track
diagnostics (`diffusionkit.bayes`). No workflow currently carries a
production-readiness or short-track confidence-interval calibration claim.

## Install

```bash
pip install -e .                    # numpy, polars, scipy -- classic and gridpost
pip install -e ".[bayes,plots]"      # NumPyro/JAX Bayesian pipeline and plotting modules
```

Python >=3.11. `classic` and `gridpost` do not import JAX, NumPyro, or
plotting libraries.

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

## D and alpha posteriors

```python
from diffusionkit import Acquisition, AcquisitionParams, load_tracks
from diffusionkit.gridpost import GridPostOptions, analyze_tracks

tracks = load_tracks(
    "mobile_beads_1to200.csv",
    AcquisitionParams(pixel_size_um=0.1043, dt_s=0.035),
)
result = analyze_tracks(
    tracks,
    Acquisition(dt_s=0.035, exposure_s=0.020),  # the D posterior models blur; the alpha posterior excludes it
    GridPostOptions(min_frames=3, level=.9),
)

result.fits  # two rows per track: posterior_D, posterior_alpha
```

An exact grid posterior over `D` from the Gaussian likelihood of all
consecutive displacements, using each frame's localization SD and a flat
(least-informative) prior in `ln D`. There is no lag window. This is the
per-track information to report for `D`: a short, uninformative track
produces a wide posterior rather than a falsely confident point estimate (see
[docs/gridpost.md](docs/gridpost.md) and
[prototypes/README.md](prototypes/README.md), which this module is built
from). `alpha` (the fBm exponent) gets its own grid posterior with the
generalized diffusion coefficient `K` marginalized out as a nuisance
parameter -- `D` and `alpha` are two independent per-track measurements, not
a joint fit, which sidesteps the well-known `K`/`alpha` MLE degeneracy.

The D posterior's interval is a genuine credible interval under a flat
prior, but its calibration is a simulation check under the model (see
[prototypes/README.md](prototypes/README.md)), not a claim about
experimental tracks.

## Localization and acquisition contract

The table uses `track_id`, `frame`, `x_um`, `y_um`, and normally
`sigma_x_um`, `sigma_y_um`. The sigma columns contain position standard
errors in micrometers, not spot/PSF widths. Spotsolve's `se_x`/`se_y` map to
these columns after pixel conversion; the current spt-pipeline adapter
already makes that conversion.

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
`Acquisition(exposure_s=...)` (a closed-form Berglund box-shutter average).
The MSD fits do not, and they are `excluded` when `exposure_s > 0`.
`exposure_s=0` is an instantaneous-observation assumption, not a statement
that your camera has zero exposure.

Omitting correction requires explicit `localization="ignore"`; missing SD
columns do not silently become zero. Calibrated zero SDs can be supplied
for noise-free synthetic data.

## Data and algorithms

Trajectory data is a `polars.DataFrame` (`track_id`, `frame`, `x_um`, `y_um`,
`sigma_x_um`, `sigma_y_um`) end to end; there is no intermediate per-track
object. Plain dataclasses hold only per-analysis metadata and results:
`Acquisition`, `MSDOptions`, `MSDCurve`, `MSDFit` (classic) and
`GridPostOptions`, `PosteriorD`, `PosteriorAlpha` (gridpost). Standalone
functions validate the table, compute MSD or the grid posteriors, and fit or
summarize them. `diffusionkit.io.validated_track_frame` is the single
validation boundary every per-track algorithm calls first. No GUI, file
writing, global JAX settings, or worker creation is part of `classic` or
`gridpost`.

See [WORKFLOW.md](WORKFLOW.md) for table examples, [TABLES.md](TABLES.md) for
output semantics, [docs/classical.md](docs/classical.md) for the MSD
estimators' equations and scope, and [docs/gridpost.md](docs/gridpost.md) for
the grid posteriors'.

## Short-track posterior prototypes

`diffusionkit.gridpost.posterior` (the production `D`-posterior module used
above) is adapted from `prototypes/posterior_1d.py`, which remains the
standalone reference implementation (it imports nothing from `diffusionkit`)
with its own calibration checks and two population-level comparators to an
ensemble MSD fit (a shared-`D` posterior and a deconvolved distribution of
`D` across tracks, not used in production -- pooling is a distinct question
from per-track inference). `diffusionkit.gridpost.posterior_alpha`, adapted
from `prototypes/posterior_alpha.py`, is now production too: an honest grid
posterior over the fBm exponent `alpha` (the generalized diffusion
coefficient `K` marginalized out as a nuisance parameter) for "how correlated
are consecutive steps", staying wide on a short, uninformative track rather
than forcing a confident answer the way a calibrated hypothesis-test
statistic would. It is reported alongside, not combined with, the `D`
posterior -- `D` and `alpha` are deliberately two independent per-track
measurements, not a joint fit, which sidesteps the well-known `K`/`alpha`
MLE degeneracy. See [prototypes/README.md](prototypes/README.md).

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/validate_classic.py --output /tmp/classic_validation.json
python scripts/validate_posterior.py --output /tmp/posterior_validation.json
```

Tests include an independent pair-sum oracle, nonlinear objective checks,
Brownian recovery with varying localization error, and data/status contracts.
The recovery study uses an independent position-space simulator at 5, 10,
and 20 frames. See [FINDINGS.md](FINDINGS.md) for the current evidence and
limits. These checks are not a general calibration of alpha on experimental
tracks.

`diffusionkit.bayes` (NumPyro) fits the same exact displacement likelihood
directly, without an MSD curve, via `fit_track` -- a per-track diagnostic
tool (full NUTS posterior) for inspecting posterior shape on a short or
weakly-identified track. It is not a bulk production pipeline: bulk
per-track diffusivity estimation is `gridpost`'s `D`-posterior job.
MAP inference and the anisotropy (nested-sampling) workflow have been
removed: MAP conflated the unconstrained-space mode with the physical-space
posterior mode, and anisotropy was an archived research direction whose
prior-provenance issues were never resolved.
