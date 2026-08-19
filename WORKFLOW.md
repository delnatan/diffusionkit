# How to use this repo

Task-oriented companion to `README.md` (layout, reproduction, full column
reference) and `FINDINGS.md` (empirical results and the reasoning behind
every default below). This file is just: which call do I make, for what I
have.

## Which workflow do I want?

| You have | Call | Why |
| --- | --- | --- |
| A handful of tracks, interactive/exploratory use | `bayes.fit_track` | Bayesian is the more honest estimator with little data (no MSD-curve summary-statistic loss, priors do real work) -- see FINDINGS.md's "D should be reported in log-space" and short-track sections. |
| Hundreds-to-thousands of tracks, a full-dataset table | `analysis.fit_population` + `bayes.fit_population` | Classic MSD is fast and a useful cross-check; the Bayesian fit costs more at this scale but is worth it for the same honesty reasons, and is now itself a one-liner. |
| Short (track_length 5-10) tracks where each dataset's orientation is arbitrary | `bayes.anisotropy.analyze` (+ `null_calibration` for a population verdict) | A model-*comparison* question, not a point estimate -- see below. Kept as its own module, not a third `model=` option, because it's used differently: population-pooled, not per-track-table-shaped. |

All three sit on top of the same validated primitives (`bayes.fit_map`,
`fit_batch_map`, `sample_posterior`, `bayes_factor.py`, ...) -- nothing below
changes what those compute, only how many lines it takes to call them.

## Load data (every workflow starts here)

```python
from analysis import AcquisitionParams, load_tracks, assert_contiguous_tracks

params = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)
tracks = load_tracks("mobile_beads_1to200.csv", params)
assert_contiguous_tracks(tracks)  # both pipelines assume a gapless, uniform frame grid
```

`tracks` is a tidy polars DataFrame, one row per (track_id, frame), physical
units (`x_um`, `y_um`, `sigma_x_um`, `sigma_y_um`, ...). Every function below
takes a `tracks`-shaped DataFrame (or a single-track slice of one).

## Low-data workflow: `bayes.fit_track`

```python
import polars as pl
from bayes import fit_track

track = tracks.filter(pl.col("track_id") == 42)
fit = fit_track(track, params.dt_s, model="anomalous")  # model="normal" for D, alpha pinned to 1

fit.params["D_alpha"], fit.lo["D_alpha"], fit.hi["D_alpha"]  # median + interval, physical units
fit.params["alpha"]
```

`prior=None` (the default) builds an informative prior from *this track's
own* measured localization precision (`sigma_prior_from_localization`) --
the honest default for low-N data, not a flat/MLE-equivalent fit. Pass a
`NormalModelPrior`/`AnomalousModelPrior` instance (e.g. `WEAK_ANOMALOUS_PRIOR`)
to override it.

`method="map"` (default) is fast MAP + a Laplace interval, adequate for D
even at N=5 (FINDINGS.md). Reach for `method="nuts"` when the posterior's
*shape* matters, not just its center -- e.g. a very short track where a
Gaussian approximation is suspect:

```python
nuts_fit = fit_track(track, params.dt_s, model="anomalous", method="nuts")
samples, mcmc = nuts_fit.raw  # full posterior draws, for bayes.plot_posterior_corner etc.
```

Full runnable example: `scripts/quickstart_single_track.py`.

## Bulk workflow: thousands of tracks

```python
from analysis import fit_population as fit_population_classic
from bayes import fit_population as fit_population_bayes

classic = fit_population_classic(tracks, params.dt_s)
# classic.per_track, classic.ensemble, classic.ensemble_normal_fit, ...

bayes_fit = fit_population_bayes(tracks, params.dt_s, model="both")
# one row per track: D (normal model, primary), alpha (anomalous model,
# primary), D_alpha (anomalous model, secondary/diagnostic) -- see
# README's "Results tables" reference for every column.
```

Both run the same production path the two pipelines have always used
(`compute_all_tamsd`/`fit_all_tracks` for classic; batched exact MAP for
Bayes) -- this is a repackaging, not a different estimator.

`bayes.fit_population`'s `engine="map"` (default) is FINDINGS.md's
production choice: more accurate, better-calibrated, and the one worth its
extra cost per-track. If the dataset is large enough that raw throughput
becomes the binding constraint, `engine="svi"` is the documented escape
valve (~9 vs. ~21 minutes on 365 real tracks in FINDINGS.md's benchmark,
at some cost to calibration) -- same call, one keyword.

Full runnable examples: `scripts/run_msd_analysis.py`,
`scripts/run_bayes_analysis.py`.

## Anisotropy workflow: short tracks, per-dataset orientation

```python
from bayes import anisotropy

result = anisotropy.analyze(tracks, params.dt_s, min_track_length=5, max_track_length=10)
# result.per_track: log_bf10 (the detector) + eps/psi posterior (descriptive) + geometry
# result.ensemble:  sum_log_bf10 across all tracks (or grouped by label_col)
```

`eps`'s posterior interval describes one track's own diffusivity tensor,
honestly wide at N=5-10 -- a descriptive summary, not a detector.
`log_bf10` (and its ensemble sum) is the actual detector: individual short
tracks are almost always "inconclusive" by design (correctly, not a bug),
but summed across a population that plausibly shares real anisotropic
behavior, evidence accumulates correctly (FINDINGS.md, "Anisotropy
detection"). Group by any column already on `tracks` via `label_col=`.

For a rigorous population-level verdict (not just eyeballing
`sum_log_bf10` against the generic Jeffreys scale), calibrate against a
matched-composition null:

```python
composition = {5: 39, 6: 49, 7: 32, ...}  # {track_length: n_tracks}, e.g. from a group_by
null_sums = anisotropy.null_calibration(composition, params.dt_s, anisotropy.AnisotropicModelPrior(),
                                          n_null=200)
p_value = (null_sums >= result.ensemble["sum_log_bf10"][0]).mean()
```

This costs real time (O(100) simulated replicate datasets) -- not run by
default inside `analyze`.

Full runnable example: `scripts/run_anisotropy_analysis.py`.

## Where to go next

- **README.md** -- repo layout, exact reproduction steps, full results-table
  column reference.
- **FINDINGS.md** -- why every default above is what it is: empirical
  results, known pitfalls, and the checks that motivated each production
  decision.
