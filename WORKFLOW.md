# How to use this repo

Task-oriented companion to `README.md` (layout, reproduction, full column
reference) and `FINDINGS.md` (empirical results and the reasoning behind
every default below). This file is just: which call do I make, for what I
have.

## Which workflow do I want?

| You have | Call | Why |
| --- | --- | --- |
| A handful of tracks, interactive/exploratory use | `diffusionkit.bayes.fit_track` | Bayesian is the more honest estimator with little data (no MSD-curve summary-statistic loss, priors do real work) -- see FINDINGS.md's "D should be reported in log-space" and short-track sections. |
| Hundreds-to-thousands of tracks, a full-dataset table | `diffusionkit.classic.fit_population` + `diffusionkit.bayes.fit_population` | Classic MSD is fast and a useful cross-check; the Bayesian fit costs more at this scale but is worth it for the same honesty reasons, and is now itself a one-liner. |
| "Is *this* track diffusing anisotropically?" | `diffusionkit.bayes.anisotropy.analyze` | A model-*comparison* question, not a point estimate -- see below. Its own module, not a third `model=` option: it compares two models by nested sampling rather than fitting one, and importing it pulls in matplotlib. |

The first two sit on the same validated primitives (`diffusionkit.bayes.fit_map`,
`fit_table_map`, `sample_posterior`) -- nothing below changes what those
compute, only how many lines it takes to call them. The anisotropy workflow
runs a different engine (`bayes.nested`, nested sampling) because it answers
a different kind of question.

## Load data (every workflow starts here)

```python
from diffusionkit.classic import AcquisitionParams, load_tracks, assert_contiguous_tracks

params = AcquisitionParams(pixel_size_um=0.1043, dt_s=0.033)
tracks = load_tracks("mobile_beads_1to200.csv", params)
assert_contiguous_tracks(tracks)  # both pipelines assume a gapless, uniform frame grid
```

`tracks` is a tidy polars DataFrame, one row per (track_id, frame), physical
units (`x_um`, `y_um`, `sigma_x_um`, `sigma_y_um`, ...). Every function below
takes a `tracks`-shaped DataFrame (or a single-track slice of one).

## Low-data workflow: `diffusionkit.bayes.fit_track`

```python
import polars as pl
from diffusionkit.bayes import fit_track

track = tracks.filter(pl.col("track_id") == 42)
fit = fit_track(track, params.dt_s, model="anomalous")  # model="normal" for D, alpha pinned to 1

fit.params["K"], fit.lo["K"], fit.hi["K"]  # median + interval, physical units
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
from diffusionkit.classic import fit_population as fit_population_classic
from diffusionkit.bayes import fit_population as fit_population_bayes

classic = fit_population_classic(tracks, params.dt_s)
# classic.per_track, classic.ensemble, classic.ensemble_normal_fit, ...

bayes_fit = fit_population_bayes(tracks, params.dt_s, model="both")
# one row per track: D (normal model, primary), alpha (anomalous model,
# primary), K (anomalous model, secondary/diagnostic) -- see
# README's "Results tables" reference for every column.
```

Both run the same production path the two pipelines have always used
(`compute_all_tamsd`/`fit_all_tracks` for classic; batched exact MAP for
Bayes) -- this is a repackaging, not a different estimator.

`diffusionkit.bayes.fit_population` runs `inference.fit_table_map` (batched
exact MAP), FINDINGS.md's production choice: more accurate and much better
calibrated, and worth its extra cost per track. The faster SVI path
(~9 vs. ~21 minutes on 365 real tracks) reports uncertainty 3-10x too
narrow, so it is not offered as a keyword here -- call
`inference.fit_table_svi` directly if you want the comparison the recovery
scripts make.

Full runnable examples: `scripts/run_msd_analysis.py`,
`scripts/run_bayes_analysis.py`.

## Anisotropy workflow: is *this* track anisotropic?

```python
from diffusionkit.bayes import anisotropy

per_track = anisotropy.analyze(tracks, params.dt_s)   # one row per trajectory
per_track.select("track_id", "track_length", "log_bf10", "log_bf10_stderr", "evidence")
```

`log_bf10` is the answer: positive favours anisotropy, negative favours
isotropy, near zero means this track does not say. It is self-calibrating --
a proper Bayes factor already accounts for how much apparent elongation
sampling noise produces at this track length, so there is no null reference
distribution to simulate and none is shipped. `eps_median`/`eps_lo`/`eps_hi`
come from the same run and describe *how much*, once `log_bf10` has
established *whether*.

Read `log_bf10_stderr` alongside it. Nested sampling returns a stochastic
evidence, and on a short track that uncertainty exceeds the evidence itself.
The `evidence` column does this for you and says `inconclusive (below
sampler noise)` when it happens.

### How long a track do you need?

Anisotropy is answerable per track only when the track is long enough to
carry the information. Measured fraction of single tracks reaching strong
evidence (`log_bf10 > 3`) on their own, at D=0.05 um^2/s, dt=0.033 s:

| track_length | true eps=0 (false positives) | true eps=0.5 | true eps=0.8 |
| --- | --- | --- | --- |
| 5 | 0% | 0% | 0% |
| 20 | 0% | 3% | 33% |
| 50 | 0% | 35% | 97% |
| 200 | 2% | 98% | 100% |

Short tracks are not excluded -- `analyze` has no track-length cap, and it
is worth running them precisely to see the honest near-zero answer. Just do
not read structure into it.

### What this workflow deliberately will not do

It will not pool tracks. Summing `log_bf10` across trajectories is a form of
ensemble averaging, which is the thing this package exists to avoid; worse,
the sum answers "does each track have its own independent anisotropy",
not "do these tracks share an axis". A pooled fit with one shared `D` also
manufactures anisotropy out of ordinary `D`-heterogeneity once the spread
reaches ~0.5 decades (FINDINGS.md). Per-track inference is immune to that,
because every track carries its own `D`.

Full runnable example: `scripts/run_anisotropy_analysis.py`
(`--limit N` for a quick look at the N longest tracks).

## Where to go next

- **README.md** -- repo layout, exact reproduction steps, full results-table
  column reference.
- **FINDINGS.md** -- why every default above is what it is: empirical
  results, known pitfalls, and the checks that motivated each production
  decision.
