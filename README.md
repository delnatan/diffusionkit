# diffusionkit

Small, data-oriented tools for analyzing 2D single-particle tracks.
The classical workflow is being rebuilt around explicit inputs, per-observation
localization errors, and inspectable fit results. Three analysis paths are
supported: classical linear MSD fits (kept for historical reasons), a
per-track grid posterior over `D` (the recommended per-track information to
extract), and Bayesian NUTS via NumPyro for per-track diagnostics. Neither
workflow currently carries a production-readiness or short-track
confidence-interval calibration claim.

## Install

```bash
pip install -e .                    # numpy, polars, scipy
pip install -e ".[bayes,plots]"      # existing Bayesian and plotting modules
```

Python >=3.11. The classical core does not import JAX, NumPyro, plotting
libraries, or the archived estimators.

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

result.fits     # three rows per track: MSD D, power-law K/alpha, D posterior
result.msd      # measured MSD, pair-specific noise offset, corrected MSD
result.options # the settings that produced these results
```

Three estimators:

- **Brownian:** ordinary least squares of corrected MSD = `4 D tau`,
  through the origin. Negative D estimates are retained and flagged.
- **Power law:** nonlinear least squares of corrected MSD = `4 K tau**alpha`,
  with `K >= 0` and `0 <= alpha <= 2`. Fits at the boundaries are flagged.
  Negative corrected MSD points remain in the fit; there is no log(MSD).
- **D posterior:** an exact grid posterior over `D` from the Gaussian
  likelihood of all consecutive displacements, using each frame's
  localization SD and a flat (least-informative) prior in `ln D`. There is no
  lag window. This is the per-track information to report for `D`: a short,
  uninformative track produces a wide posterior rather than a falsely
  confident point estimate (see [docs/classical.md](docs/classical.md) and
  [prototypes/README.md](prototypes/README.md), which this module is built
  from).

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
justify the standard independent-residual regression error bars. The D
posterior's interval is a genuine credible interval under a flat prior, but
its calibration is a simulation check under the model (see
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
Missing frames are rejected, not compressed into one time step. The D
posterior models continuous-exposure blur through `Acquisition(exposure_s=...)`
(a closed-form Berglund box-shutter average). The MSD fits do not, and they
are `excluded` when `exposure_s > 0`. `exposure_s=0` is an
instantaneous-observation assumption, not a statement that your camera has
zero exposure.

Omitting correction requires explicit `localization="ignore"`; missing SD
columns do not silently become zero. Calibrated zero SDs can be supplied
for noise-free synthetic data.

## Data and algorithms

Plain dataclasses hold `Track`, `Acquisition`, `MSDOptions`, `MSDCurve`,
`MSDFit`, and workflow outputs. Standalone functions validate/convert data,
compute MSD, and fit it. Arrays are copied and made read-only at the track
validation boundary. No GUI, file writing, global JAX settings, or worker
creation is part of the new analysis workflow.

See [WORKFLOW.md](WORKFLOW.md) for arrays and table examples,
[TABLES.md](TABLES.md) for output semantics, and
[docs/classical.md](docs/classical.md) for equations and scope.

## Short-track posterior prototypes

`diffusionkit.classic.posterior` (the production `D`-posterior module used
above) is adapted from `prototypes/posterior_1d.py`, which remains the
standalone reference implementation (it imports nothing from `diffusionkit`)
with its own calibration checks and two population-level comparators to an
ensemble MSD fit (a shared-`D` posterior and a deconvolved distribution of
`D` across tracks, not used in production -- pooling is a distinct question
from per-track inference). `prototypes/posterior_alpha.py` is a newer,
not-yet-production companion: an honest grid posterior over the fBm exponent
`alpha` (K marginalized out) for the question "is this motion Brownian?",
staying wide on a short, uninformative track rather than forcing a
confident answer the way a calibrated hypothesis-test statistic would. See
[prototypes/README.md](prototypes/README.md).

## Validation and migration

```bash
python -m unittest discover -s tests -v
python scripts/validate_classic.py --output /tmp/classic_validation.json
```

Tests include an independent pair-sum oracle, nonlinear objective checks,
Brownian recovery with varying localization error, and data/status contracts.
The recovery study uses an independent position-space simulator at 5, 10,
and 20 frames. See [FINDINGS.md](FINDINGS.md) for the current evidence and
limits. These checks are not a general calibration of alpha on experimental
tracks.

The previous classical implementation, including in-progress GLS work, is
preserved in `diffusionkit.legacy.classic`. Old import paths still resolve to
that implementation; they do **not** silently run the new estimators.
The existing napari panel therefore still uses the legacy workflow until it
is migrated to `analyze_tracks` and the new result schema.

Previous documentation is preserved in [docs/archive](docs/archive/INDEX.md)
for reproducibility. Its recommendations and production claims are withdrawn.

`diffusionkit.bayes` (NumPyro) fits the same exact displacement likelihood
directly, without an MSD curve, via `fit_track` -- a per-track diagnostic
tool (full NUTS posterior) for inspecting posterior shape on a short or
weakly-identified track. It is not a bulk production pipeline: bulk
per-track diffusivity estimation is the classical `D`-posterior's job.
MAP inference and the anisotropy (nested-sampling) workflow have been
removed: MAP conflated the unconstrained-space mode with the physical-space
posterior mode, and anisotropy was an archived research direction whose
prior-provenance issues were never resolved (see the historical `AUDIT.md`).
