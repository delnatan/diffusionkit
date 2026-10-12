# fBm grid posterior: per-track alpha and D

`diffusionkit.fbmpost` is the second per-track posterior pipeline, next to `diffusionkit.gridpost`.

- `gridpost`: the Brownian D posterior (alpha = 1) and its population models.
- `fbmpost`: the joint posterior of the anomalous exponent alpha and D under blurred fractional Brownian motion.

Both use the same exact Gaussian displacement likelihood with known per-frame localization SDs and box-shutter
blur. At alpha = 1 the two likelihoods are identical (tested). Both are evaluated on a fixed grid; neither uses
MCMC or an MSD curve.

Use `gridpost` for D when the motion is Brownian. Use `fbmpost` when alpha is part of the question, or when a
single Brownian D is not well defined (the motion is non-Brownian). Don't pair `gridpost`'s D with `fbmpost`'s
alpha: they come from different posteriors.

```python
from concurrent.futures import ThreadPoolExecutor
from diffusionkit import Acquisition
from diffusionkit.fbmpost import FbmPostOptions, analyze_tracks
from diffusionkit.fbmpost.viz import plot_alpha_D   # needs the `plots` extra

with ThreadPoolExecutor() as pool:
    result = analyze_tracks(tracks, Acquisition(dt_s=.044, exposure_s=.02), FbmPostOptions(tau_ref_s=.1),
                            map_fn=pool.map)
fig = plot_alpha_D(result.fits)
```

## Model

Per axis, the m = n - 1 displacements are Gaussian:

```
Sigma = K C_alpha + B
```

- C_alpha is the fGn covariance per unit K, averaged over the exposure (`fbmpost.covariance`, closed form).
  - The true motion's MSD is 2 K tau^alpha per axis.
  - At alpha = 1 this is gridpost's Berglund matrix.
- B is the localization term (`gridpost.likelihood.localization_covariance`).

For each alpha, gridpost's whitening applies (B = L Lᵀ, L⁻¹C_αL⁻ᵀ = Q diag(λ) Qᵀ). The whitened data have
independent variances 1 + K λ_k. So the grid costs one eigendecomposition per (axis, alpha), and O(m) per
amplitude value.

## Prior: what makes alpha independent of K

K has units of um²/s^alpha, so "flat in ln K" fixes nothing on its own. The bounds of the amplitude prior are
what decide whether an uninformative track leans in alpha:

- **Anchor at dt.** Below the localization floor the likelihood is flat in the amplitude. With bounds on D at
  dt, that flat stretch is longer for small alpha, so tracks that see only noise lean to low alpha.
- **Anchor at the longest lag.** Bounds on D there still lean, unless the range spans many decades.

The amplitude axis is therefore the track's **signal-to-noise**, SNR = K ‖λ_α‖, the norm of the whitened motion
covariance over both axes. The prior is flat in alpha on `[alpha_min, alpha_max]` (default 0.02-1.98) and flat
in ln SNR on `[snr_min, snr_max]` (1e-4 to 1e8).

- Within one alpha this is flat in ln K, so it changes nothing for a track that measures its amplitude.
- Below detection the likelihood depends on K only through K λ, so in SNR units the noise-only stretch is
  equally long at every alpha. A track that sees only noise returns the flat alpha prior: within 3-4% at 10-40
  frames in `scripts/validate_fbmpost.py`.

`D_grid_edge` = "low" means the amplitude posterior reaches `snr_min`: the track only bounds D from above.

## Output

There is one fit row per track (`model = "posterior_alpha_D"`).

**alpha**
- Mean, equal-tailed `level` interval and SD.
- `alpha_post_info_bits`: KL to the flat prior. 0 means the track left alpha at its prior.
- `alpha_post_p_below_1`: the posterior probability that alpha < 1.

**D_ref = MSD_1D(tau_ref) / (2 tau_ref)**
- This is the true (unblurred) motion's D at the lag `tau_ref_s` (default 0.1 s), in um²/s.
- At alpha = 1 it is the Brownian D at any tau_ref. Otherwise tau_ref is part of its definition: keep it fixed
  in seconds across datasets.
- It is **not a Brownian D**, and not comparable with `gridpost`'s D or an MSD D: it changes with tau_ref
  whenever alpha differs from 1, and it carries the alpha prior. Use `gridpost` for how fast tracks move.
- Columns: mean E[D_ref], interval, and `D_ref_floor_um2_s` = <s²>/tau_ref, the D_ref at which the motion's
  MSD at tau_ref equals the noise's 2<s²>.

**Ellipse**
- `alpha_post_mean`, `log10_D_ref_post_mean`, their SDs and `alpha_log10_D_ref_corr`.
- `viz.plot_alpha_D` draws them as 1-SD ellipses. An ellipse's projections are ±1 marginal SD, like error bars,
  and its tilt is the track's own trade-off between alpha and D.
- That correlation runs from about -0.75 (low alpha, near the floor) to +0.7 (fast superdiffusive tracks).
  No single tau_ref removes it for every track, so crosses would misstate the uncertainty.

With `keep_posteriors=True`, `result.posteriors.log_post_alpha` holds each track's marginal posterior of alpha.

## Prior and likelihood

Every per-track number here is a posterior, and the prior matters:

- **alpha**: flat on `[alpha_min, alpha_max]`. A track with 0 bits returns exactly this prior: mean 1, SD 0.57.
- **The amplitude**: flat in ln SNR. Summing it out is what makes a track's alpha curve. Well above the floor
  the amplitude is measured and the prior only adds a constant. Near the floor, the prior decides how the
  curve leans.
- **D_ref**: its margin inherits the alpha prior (see Calibration).

Only the joint (alpha, D) likelihood is free of priors. In the grid's ln SNR it is `log_likelihood_grid`, and
moving it to a common ln D_ref axis is a shift per alpha row, with no Jacobian.

So tracks do not combine by adding their alpha curves. On simulated populations (1500 tracks of 5-80
frames, D_ref log-normal around the floor), a 1D deconvolution of the alpha curves was biased:

- Two alpha modes: P(alpha < 0.75) came out 0.43 against a true 0.60.
- Alpha correlated with D: it came out 0.05 against 0.49.

Integrating the amplitude under the true D population instead recovered 0.59. A population of alpha has to
be estimated jointly with the population of D: the deconvolution of `gridpost.deconvolve` on the (alpha, ln
D_ref) plane.

## Calibration (`scripts/validate_fbmpost.py`)

The simulation uses fBm drawn on a fine time grid and averaged over each exposure, independent of the closed
form: dt 44 ms, exposure 20 ms, SD 13 nm, alpha 0.3-1.4, D_ref 1e-3 to 3e-2, 40 tracks per cell.

- **90% coverage:** alpha 0.94-0.95 and D_ref 0.87-0.90, pooled by length (10, 20, 40 frames).
- **Alpha posterior SD, well above the floor:** about 0.35 at 10 frames, 0.25 at 20 and 0.18 at 40.
- **Below the floor:** a track's D_ref interval inherits the alpha prior, because high-alpha hypotheses set a
  lower detection limit on D_ref. Coverage there drops to 0.65-0.8; read D_ref as an upper limit.

On the 2026-09-23 gelatin cool-down, the liquid movies' P(alpha < 1) matches Brownian twins (each track
re-simulated at alpha = 1 with its own length, SDs and D) in every information bin.

## Limits

- **Localization SDs are taken as exact.** Near the floor, alpha depends on them: under-reported SDs read as
  low alpha.
- **Model.** fBm is the only Gaussian, stationary-increment model with a power-law MSD. Confinement with a cage
  time inside the track, or a single-frame linking outlier, is misspecified for it. Both push alpha down.
- **Cost.** Per track it is O(n_alpha m³): ~25 ms at 40 frames, ~0.1 s at 100 and ~0.8 s at 300.
