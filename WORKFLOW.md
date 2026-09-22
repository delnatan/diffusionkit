# Classical and grid-posterior workflows

Use `diffusionkit.classic.analyze_track`/`analyze_tracks` for MSD fits and
`diffusionkit.gridpost.analyze_track`/`analyze_tracks` for the D and alpha
grid posteriors -- two independent entry points, each returning its own
`fits` table; join on `track_id` if you want both. `diffusionkit.bayes.fit_track`
is a separate, per-track diagnostic tool (full NUTS posterior via NumPyro),
not part of either bulk table.

## One track, a DataFrame

```python
import numpy as np
import polars as pl
from diffusionkit import Acquisition
from diffusionkit.classic import MSDOptions, analyze_track

track = pl.DataFrame({
    "track_id": [42]*5, "frame": np.arange(5),
    "x_um": [0, .02, .01, .04, .03], "y_um": [0, .01, .04, .03, .06],
    "sigma_x_um": [.01]*5, "sigma_y_um": [.01]*5,
})
result = analyze_track(track, Acquisition(dt_s=.033), MSDOptions(max_lag=3))
print(result.brownian.parameters, result.brownian.status)
print(result.anomalous.parameters, result.anomalous.status)
```

A track is just rows of a `polars.DataFrame`; there is no intermediate
per-track object. Functions validate it (`diffusionkit.validated_track_frame`
is the shared boundary) and return new data. Single-track invalid inputs
raise `ValueError`; a valid but too-short track returns `excluded` fits. A
result does not imply statistical identifiability just because the optimizer
converged.

## An existing physical-unit table

```python
import polars as pl
from diffusionkit import Acquisition
from diffusionkit.classic import analyze_track, analyze_tracks

acquisition = Acquisition(dt_s=.033)
# `tracks` is a Polars table, already in micrometers.
one = tracks.filter(pl.col("track_id") == 42)
selected = analyze_track(one, acquisition)
all_results = analyze_tracks(tracks, acquisition)
```

Required columns are
`track_id`, `frame`, `x_um`, `y_um`; `sigma_x_um` and `sigma_y_um` are also
required by the default correction. Their units are micrometers, not pixels.
If `t_s` or `track_length` is present, it must agree with `frame * dt_s` or
the actual row count. Those columns are not otherwise required.

Batch analysis keeps two fit rows per track, including invalid or excluded
tracks. A malformed table schema raises before fitting. An empty input with
valid column types returns typed empty result tables. There is no minimum
population size and no ensemble calculation.

## The D posterior

```python
from diffusionkit.gridpost import posterior as P

s = P.track_posterior(one, Acquisition(dt_s=.033, exposure_s=.03))
s["median"], s["lo"], s["hi"]   # D_post_median/lo/hi_um2_s, flat prior by default
```

Set `exposure_s` to the camera exposure. The posterior models blur; the
alpha posterior then reports `excluded`. This is the field also attached to
`gridpost.analyze_track`/`analyze_tracks`' output as `result.posterior_D`
(`model="posterior_D"` rows in the bulk `fits` table) -- a track's own
uncertainty stays visible as how wide its interval is, rather than being
collapsed to a point estimate. A short or noise-dominated track producing a
wide interval is expected, not a failure.

## The alpha posterior

```python
from diffusionkit.gridpost import posterior_alpha as PA

s = PA.track_alpha_posterior(one, Acquisition(dt_s=.033))
s["median"], s["lo"], s["hi"]   # alpha_post_median/lo/hi, flat prior over K by default
```

A separate per-track measurement from the D posterior, not a joint fit:
`D` comes from the alpha=1 model, `alpha` from the fBm model with the
generalized diffusion coefficient K marginalized out entirely -- this
answers "how correlated are consecutive steps" independently of "how big
are the steps", sidestepping the well-known K/alpha MLE degeneracy. It has
no exposure-blur model, so `exposure_s` must be 0; `result.posterior_alpha`
(`model="posterior_alpha"` rows) reports `excluded` otherwise.

## Inspect or change the classical analysis

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

For a GUI, pass `progress(done, total)` to either `analyze_tracks`. It is
invoked in the calling thread. Start a GUI-managed worker outside this
library. Join fit rows on both `track_id` and `model`, and keep
`status`/`message` visible. `status="ok"` is a numerical result status, not a
motion class. Persist `result.acquisition` and `result.options` alongside
exported tables (e.g. `dataclasses.asdict`); the tables alone are not a
complete run record.
