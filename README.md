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
    Acquisition(dt_s=0.035, exposure_s=0.020),  # both posteriors model the exposure's motion blur
    GridPostOptions(min_frames=3, level=.9, D_min_um2_s=1e-4, D_max_um2_s=10., n_D=501),
)

result.fits  # two rows per track: posterior_D, posterior_alpha
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
[docs/gridpost.md](docs/gridpost.md)). `alpha` (the fBm exponent) gets its own grid posterior with the
motion's scale marginalized out as a nuisance parameter -- `D` and `alpha` are
reported as two per-track measurements, not a joint point estimate, which
sidesteps the well-known `K`/`alpha` MLE degeneracy. The scale is the
apparent D (the Brownian D with the same blurred one-frame step variance), so
the D posterior is the `alpha = 1` slice of each track's joint (alpha, D)
posterior, which is kept for population reads (below).

The D posterior's interval is a genuine credible interval under a flat
prior, but its calibration is a simulation check under the model (see
`scripts/validate_posterior.py`), not a claim about experimental tracks.

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
`gridpost` D and alpha posteriors model continuous-exposure blur through
`Acquisition(exposure_s=...)` (a closed-form box-shutter average: Berglund's at
alpha=1, its fBm generalization for alpha).
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
so still spots carry the estimate without any classification. Correct before any per-track analysis: uncorrected
drift of ~17 nm/frame moves a D ~ 1e-3 um^2/s population to 0.01-0.05. See [docs/drift.md](docs/drift.md).

## Data and algorithms

Trajectory data is a `polars.DataFrame` (`track_id`, `frame`, `x_um`, `y_um`,
`sigma_x_um`, `sigma_y_um`) end to end; there is no intermediate per-track
object. Plain dataclasses hold only per-analysis metadata and results:
`Acquisition`, `MSDOptions`, `MSDCurve`, `MSDFit` (classic) and
`GridPostOptions`, `PosteriorD`, `PosteriorAlpha` (gridpost). Standalone
functions validate the table, compute MSD or the grid posteriors, and fit or
summarize them. `diffusionkit.io.validated_track_frame` is the single
validation boundary every per-track algorithm calls first. No GUI, global JAX
settings, or worker creation is part of `classic` or `gridpost`; the one file
`gridpost` writes is the per-track joint posterior (`gridpost.joint`), so that
whatever produces it and whatever reads it share one format.

See [WORKFLOW.md](WORKFLOW.md) for table examples, [TABLES.md](TABLES.md) for
output semantics, [docs/classical.md](docs/classical.md) for the MSD
estimators' equations and scope, and [docs/gridpost.md](docs/gridpost.md) for
the grid posteriors'.

## Distribution of D across tracks

`gridpost.deconvolve_tracks(table, acquisition)` is a population-level
comparator to an ensemble MSD fit, not a replacement for the per-track
posteriors: it estimates how `D` is distributed across a table of tracks from
each track's likelihood, without averaging away each track's own uncertainty
first. The estimate is a smooth log density whose smoothness is chosen by the
data (Laplace evidence), and `samples` are posterior draws of the whole
distribution, so any band or mass comes with an interval
(`result.band(.68, cumulative=True)`). See
[docs/gridpost.md](docs/gridpost.md#distribution-of-d-across-tracks-deconvolve).

## Distribution of alpha and D across tracks

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from diffusionkit.gridpost import (GridPostOptions, analyze_tracks, deconvolve_joint, joint_posteriors,
                                   pool_joint_posteriors, read_joint_posteriors, write_joint_posteriors)
from diffusionkit.gridpost.deconvolve import deconvolve

options = GridPostOptions()
with ThreadPoolExecutor(8) as pool:
    result = analyze_tracks(tracks, acquisition, options, keep_posteriors=True, map_fn=pool.map)

# 1D reads: D (the alpha = 1 model) and alpha, each from its own per-track posteriors
post = result.posteriors
D_pop = deconvolve(post.log_post_D, options.u_D())
alpha_pop = deconvolve(post.log_post_alpha, options.alphas())

# Each track's joint (alpha, ln D) posterior, as one compact file per sample
wt = joint_posteriors(result, sample="wt")
write_joint_posteriors("wt_joint.parquet", wt)

# The joint read of one sample
pop = deconvolve_joint(wt.log_post, wt.alphas, wt.u)
pop.weights                                            # (n_alpha, n_D), sums to 1
region = wt.alphas < .4                                # any region you define, e.g. one mode
mass = pop.samples[:, region].sum(axis=(1, 2))          # its mass, one value per draw
np.quantile(mass, [.05, .5, .95])
raw = np.exp(wt.log_post).mean(axis=0)                 # raw pooled posterior, before deconvolution

# ... or of several samples pooled
both = pool_joint_posteriors([read_joint_posteriors("wt_joint.parquet"),
                              read_joint_posteriors("ko_joint.parquet")])
pooled = deconvolve_joint(both.log_post, both.alphas, both.u)
```

`deconvolve_joint` keeps what the two 1D reads lose: which D goes with which
alpha. A 3-frame track says little about alpha but much about D, so it takes
the alpha mix of the tracks with D like its own instead of the whole
population's. That assumes the alpha mix at a given D does not depend on track
length: where a low-alpha population makes longer tracks than the mobile one
(particles that stay in focus), the joint read over-counts it, by ~0.12 in
simulation. Below the localization floor alpha is not identified, and
immobile particles end up in the lowest alpha cells there, so read low-alpha
mass together with its D. Pooling requires the same cells, and dt and
exposure equal to 0.1%. The D axis is the apparent D, so on it the `alpha = 1`
row is the ordinary D. See
[docs/gridpost.md](docs/gridpost.md#distribution-of-alpha-d-across-tracks-deconvolve_joint).

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/validate_classic.py --output /tmp/classic_validation.json
python scripts/validate_posterior.py --output /tmp/posterior_validation.json
python scripts/validate_deconvolve.py --output /tmp/deconvolve_validation.json
python scripts/validate_deconvolve_joint.py --output /tmp/deconvolve_joint_validation.json
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
weakly-identified track. It models the same exposure blur as `gridpost`
(`fit_track(track, dt_s, exposure_s=...)`), but fits one iid localization SD
per track rather than using the per-frame ones. It is not a bulk production pipeline: bulk
per-track diffusivity estimation is `gridpost`'s `D`-posterior job.
MAP inference and the anisotropy (nested-sampling) workflow have been
removed: MAP conflated the unconstrained-space mode with the physical-space
posterior mode, and anisotropy was an archived research direction whose
prior-provenance issues were never resolved.
