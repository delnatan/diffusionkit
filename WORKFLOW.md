# Classical and grid-posterior workflows

Use `diffusionkit.classic.analyze_track`/`analyze_tracks` for MSD fits and
`diffusionkit.gridpost.analyze_track`/`analyze_tracks` for the D
grid posterior -- two independent entry points, each returning its own
`fits` table; join on `track_id` if you want both. `diffusionkit.bayes.fit_track`
is a separate, per-track diagnostic tool (full NUTS posterior via NumPyro),
not part of either bulk table; pass it the same `exposure_s` as the grid
posteriors' `Acquisition`, so both fit the blurred motion.

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

Set `exposure_s` to the camera exposure; the posterior models the blur. This is the field also attached to
`gridpost.analyze_track`/`analyze_tracks`' output as `result.posterior_D`
(`model="posterior_D"` rows in the bulk `fits` table) -- a track's own
uncertainty stays visible as how wide its interval is, rather than being
collapsed to a point estimate. A short or noise-dominated track producing a
wide interval is expected, not a failure.

`analyze_tracks` fits tracks one after another by default. For a large table pass a thread pool's
`map`; the linear algebra releases the GIL, and the result is the serial one:

```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(8) as pool:
    result = analyze_tracks(tracks, acquisition, map_fn=pool.map)
```

## The distribution of D across tracks

```python
import numpy as np
from diffusionkit.gridpost import analyze_tracks, by_track_length, deconvolve_tracks

result = analyze_tracks(tracks, Acquisition(dt_s=.033, exposure_s=.03), keep_posteriors=True)
pop = deconvolve_tracks(result)       # built from the kept per-track posteriors
pop.u, pop.weights                    # ln D grid, distribution (sums to 1)
lo, hi = pop.band(.68)                # pointwise band on the weights
lo, hi = pop.band(.95, cumulative=True)
above = pop.samples[:, pop.u > np.log(.05)].sum(axis=1)  # any mass: read its interval off the draws
```

A population-level comparator to an ensemble MSD fit, built from each
track's likelihood rather than its point estimate. Its smoothness is chosen
by the data, not set by hand; a peak narrower than the tracks can resolve
comes out as wide as that resolution, and below the localization floor the
bands widen because the tracks cannot tell those D values apart.

### Split by track length

```python
comp = by_track_length(result.posteriors, result.options.u_D(), pop, weight="detections")
comp.labels(), comp.n_tracks, comp.n_detections   # the groups: '3', '5-6', '25+', ...
comp.pooled                  # (groups, grid): flat-prior posteriors summed per group
comp.deconvolved             # (draws, groups, grid): each track's posterior under a draw of pop
comp.deconvolved[:, :, comp.u < np.log(.035)].sum(2)   # mass below 0.035 per group, one row per draw

from diffusionkit.gridpost.viz import plot_by_track_length   # needs the plots extra
fig = plot_by_track_length(comp, result.fits["D_floor_um2_s"].drop_nulls().to_numpy())
```

Fast particles leave the focal depth within a few frames, so short tracks come
mostly from fast particles and long tracks from slow ones. The groups add up to
the whole distribution. With `weight="tracks"` they are fractions of tracks, and
the deconvolved groups add up to about the population. With
`weight="detections"` each track counts once per frame, which gives the
composition of the spots seen in focus.

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
