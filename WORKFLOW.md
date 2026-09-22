# Classical workflow

Use `diffusionkit.classic.analyze_track` or `analyze_tracks` for the rebuilt
core. `classic.fit_population` and the old fitting modules remain legacy
compatibility paths. `diffusionkit.bayes.fit_track` is a separate, per-track
diagnostic tool (full NUTS posterior via NumPyro), not part of this
workflow's bulk table.

## One track, arrays

```python
import numpy as np
from diffusionkit import Acquisition, Track
from diffusionkit.classic import MSDOptions, analyze_track

track = Track(
    track_id=42,
    frames=np.arange(5),
    positions_um=np.array([[0, 0], [.02, .01], [.01, .04], [.04, .03], [.03, .06]]),
    localization_sd_um=np.full((5, 2), .01),
)
result = analyze_track(track, Acquisition(dt_s=.033), MSDOptions(max_lag=3))
print(result.brownian.parameters, result.brownian.status)
print(result.anomalous.parameters, result.anomalous.status)
```

`Track` contains data only. Functions validate it and return new data.
Single-track invalid inputs raise `ValueError`; a valid but too-short track
returns `excluded` fits. A result does not imply statistical identifiability
just because the optimizer converged.

## An existing physical-unit table

```python
import polars as pl
from diffusionkit import Acquisition, track_from_table
from diffusionkit.classic import analyze_track, analyze_tracks

acquisition = Acquisition(dt_s=.033)
# `tracks` is a Polars table, already in micrometers.
one = track_from_table(tracks.filter(pl.col("track_id") == 42), acquisition)
selected = analyze_track(one, acquisition)
all_results = analyze_tracks(tracks, acquisition)
```

Required columns are
`track_id`, `frame`, `x_um`, `y_um`; `sigma_x_um` and `sigma_y_um` are also
required by the default correction. Their units are micrometers, not pixels.
If `t_s` or `track_length` is present, it must agree with `frame * dt_s` or
the actual row count. Those columns are not otherwise required.

Batch analysis keeps three fit rows per track, including invalid or excluded
tracks. A malformed table schema raises before fitting. An empty input with
valid column types returns typed empty result tables. There is no minimum
population size and no ensemble calculation.

## The D posterior

```python
from diffusionkit.classic import posterior as P

s = P.track_posterior(one, Acquisition(dt_s=.033, exposure_s=.03))
s["median"], s["lo"], s["hi"]   # D_post_median/lo/hi_um2_s, flat prior by default
```

Set `exposure_s` to the camera exposure. The posterior models blur; the MSD
fits then report `excluded`. This is the field also attached to
`analyze_track`/`analyze_tracks`' output as `result.posterior_D`
(`model="posterior_D"` rows in the bulk `fits` table) -- a track's own
uncertainty stays visible as how wide its interval is, rather than being
collapsed to a point estimate. A short or noise-dominated track producing a
wide interval is expected, not a failure.

## Inspect or change the analysis

```python
from diffusionkit.classic import compute_msd, fit_brownian_msd, fit_anomalous_msd

curve = compute_msd(one, acquisition)
D_fit = fit_brownian_msd(curve)
alpha_fit = fit_anomalous_msd(curve)
```

Change `MSDOptions(max_lag=...)` to inspect dependence on lag selection.
Changing the window changes the estimator; it is recorded in the result.
To deliberately omit localization correction, use
`MSDOptions(localization="ignore")`. There is no fallback from missing
measurement errors to uncorrected fitting.

For a GUI, pass `progress(done, total)` to `analyze_tracks`. It is invoked
in the calling thread. Start a GUI-managed worker outside this library.
Join fit rows on both `track_id` and `model`, and keep `status`/`message`
visible. `status="ok"` is a numerical result status, not a motion class.
Persist `result.acquisition` and `result.options` alongside exported tables
(e.g. `dataclasses.asdict`); the tables alone are not a complete run record.

The old spt-pipeline `PopulationFit` consumer requires an explicit migration:
replace the ensemble view with per-track measured/corrected MSD, consume
`fits` by model, and surface statuses. Its existing imports continue to use
legacy results until that migration is made.
