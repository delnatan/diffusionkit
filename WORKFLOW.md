# Classical workflow

Use `diffusionkit.classic.analyze_track` or `analyze_tracks` for the rebuilt
core. `classic.fit_population` and the old fitting modules remain legacy
compatibility paths. The Bayesian workflow is unchanged and awaits revision.

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

## Brownian MLE and the non-Brownian axis

```python
from diffusionkit.classic import MLEOptions, fit_brownian_mle

fit = fit_brownian_mle(one, Acquisition(dt_s=.033, exposure_s=.03), MLEOptions(n_boot=500))
fit.status                        # 'ok', 'unresolved' (D_hat = 0), 'failed'
fit.parameters["D_um2_s"], fit.parameters["z_nonbrownian"]
```

Set `exposure_s` to the camera exposure. The MLE models blur; the MSD
fits then report `excluded`. A 2D histogram for a population:

```python
import numpy as np
mle = all_results.fits.filter(pl.col("model") == "brownian_mle")
resolved = mle.filter(pl.col("z_nonbrownian").is_not_null())
n_unresolved = mle.filter(pl.col("status") == "unresolved").height   # report separately
H, D_edges, z_edges = np.histogram2d(np.log10(resolved["D_um2_s"]), resolved["z_nonbrownian"],
                                     bins=(30, np.linspace(-4, 4, 33)))
```

Under the Brownian model every D column is ~N(0,1). Compare each column's
mean z with 0, using a standard error of about 1/sqrt(tracks in column).
Stratify by `n_frames` when comparing conditions. A shift indicates
non-Brownian behavior or miscalibrated localization SDs; it does not
classify individual short tracks.

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
