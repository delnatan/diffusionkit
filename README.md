# diffusionkit

Small, data-oriented tools for analyzing 2D single-particle tracks.
The classical workflow is being rebuilt around explicit inputs, per-observation
localization errors, and inspectable fit results. Bayesian inference will be
revisited separately. Neither workflow currently carries a production-readiness
or short-track confidence-interval calibration claim.

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

result.fits     # three rows per track: MSD D, power-law K/alpha, Brownian MLE
result.msd      # measured MSD, pair-specific noise offset, corrected MSD
result.options # the settings that produced these results
```

Three estimators:

- **Brownian:** ordinary least squares of corrected MSD = `4 D tau`,
  through the origin. Negative D estimates are retained and flagged.
- **Power law:** nonlinear least squares of corrected MSD = `4 K tau**alpha`,
  with `K >= 0` and `0 <= alpha <= 2`. Fits at the boundaries are flagged.
  Negative corrected MSD points remain in the fit; there is no log(MSD).
- **Brownian MLE:** the Gaussian likelihood of all consecutive displacements,
  using each frame's localization SD, maximized over `D >= 0`. There is no lag
  window. `D_hat = 0` is reported as `unresolved`. Each resolved track also
  gets `z_nonbrownian`, a bootstrap-calibrated signed score for deviation
  from Brownian motion. Under the Brownian model it is ~N(0,1) for any length,
  noise or D, which makes it the intended second axis beside D (see
  [docs/classical.md](docs/classical.md)).

The MSD fits use equal weights across the selected lags. `max_lag=3` is an explicit
comparison window, not an optimized choice or an accuracy guarantee.
`status="ok"` means the numerical fit passed its checks, not that the motion
model is established or the parameters are precise.

**No priors are used.** Supplied localization SDs are treated as known
measurement-error inputs. Their uncertainty is not inferred or propagated.
Model bounds on alpha are constraints, not a probability distribution.

**The MSD fits report no confidence intervals.**
They carry `uncertainty_method="not_estimated"`. The MLE's upper limit and
p-values are asymptotic, and their short-track behavior is measured in
`audit/brownian_mle_validation.json`. Correlated MSD
residuals do not justify the standard independent-residual regression error
bars. The Bayesian prior and Laplace interval contract will be addressed in
its own subsequent revision.

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
Missing frames are rejected, not compressed into one time step. The Brownian
MLE models continuous-exposure blur through `Acquisition(exposure_s=...)`.
The MSD fits do not, and they are `excluded` when `exposure_s > 0`.
`exposure_s=0` is an instantaneous-observation assumption, not a statement
that your camera has zero exposure. Omitting a real exposure biases
`z_nonbrownian` upward.

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

## Validation and migration

```bash
python -m unittest discover -s tests -v
python scripts/validate_classic.py --output /tmp/classic_validation.json
python scripts/validate_brownian_mle.py --output /tmp/brownian_mle_validation.json
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
The existing Bayesian and anisotropy code is unchanged and outside this
classical revision; anisotropy is an archived research direction for now.
