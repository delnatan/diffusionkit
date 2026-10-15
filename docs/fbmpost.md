# fbmpost: per-track alpha and D_ref by profile likelihood

`diffusionkit.fbmpost` is the second per-track pipeline, next to `diffusionkit.gridpost`.

- `gridpost`: the Brownian D posterior (alpha = 1) and its population models.
- `fbmpost`: the anomalous exponent alpha and the amplitude D_ref under blurred fractional Brownian motion,
  and alpha across tracks.

Both use the same exact Gaussian displacement likelihood with known per-frame localization SDs and box-shutter
blur. At alpha = 1 the two likelihoods are identical (tested). Neither uses MCMC or an MSD curve.

Use `gridpost` for D when the motion is Brownian. Use `fbmpost` when alpha is part of the question.

```python
from concurrent.futures import ThreadPoolExecutor
from diffusionkit import Acquisition
from diffusionkit.fbmpost import FbmPostOptions, analyze_tracks, shared_alpha
from diffusionkit.fbmpost.viz import plot_alpha_D   # needs the `plots` extra

with ThreadPoolExecutor() as pool:
    result = analyze_tracks(tracks, Acquisition(dt_s=.044, exposure_s=.02), FbmPostOptions(tau_ref_s=.1),
                            keep_profiles=True, map_fn=pool.map)
shared_alpha(result.profiles).summary()   # one alpha for all tracks, with its interval
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
- B is the localization term (`gridpost.likelihood.localization_covariance`), from each frame's reported SD.

For each alpha, B = L Lᵀ and L⁻¹C_αL⁻ᵀ = Q diag(λ) Qᵀ, so the whitened data y = Qᵀ L⁻¹ Δ are independent with
variances 1 + K λ_k (`likelihood.whiten_batch`). One eigendecomposition per (track, axis, alpha) makes every
later evaluation O(m). Tracks of one length are whitened together.

## Per track: the profile likelihood

The amplitude K is a nuisance parameter. It is maximized at each alpha, not integrated:

```
l_p(alpha) = max over K >= 0 of l(alpha, K)
```

This follows the ridge of the joint likelihood: each alpha gets its own best K. Integrating K under a fixed
prior instead weighs each alpha by how much of the prior's range lies below detection (with known noise the
likelihood stays finite as K -> 0) and by how wide the likelihood is in K. Both depend on alpha, so the prior
tilts every track's alpha curve the same way, and in a sum over tracks the tilt adds up.

K is profiled per (track, alpha) by a scan of ln K in e-fold steps, in units of the track's own noise, then
safeguarded Newton (`profile.profile_K`). When K = 0 is best, the track's likelihood is the noise-only value,
the same at every alpha.

**Output**, one row per track (`model = "fbm_alpha_D"`, `method = "profile_likelihood"`):

- `alpha_hat`: the maximum of l_p, refined between grid points (default grid 0.02-1.98 in steps of 0.02).
  It is null when the track shows no motion above its noise (its likelihood peaks at K = 0): alpha is then not
  identified.
- `alpha_lo`, `alpha_hi`: the likelihood-ratio interval, every alpha with l_p within chi2_1(level)/2 of the
  maximum (1.92 nats at the default `level` 0.95).
- `alpha_edge`: "low", "high" or "both" when that interval reaches an end of the grid. A short track whose
  alpha_hat sits at an end is honest ambiguity; its interval says how much.
- `alpha_dll_brownian`, `alpha_p_brownian`: the likelihood-ratio test of alpha = 1. 2 dll is chi2_1 under
  Brownian motion, since alpha = 1 is inside the range.
- `K_hat` (um²/s^alpha) and `D_ref_hat_um2_s`, `log10_D_ref_hat` at alpha_hat.
- `D_ref_lo_um2_s`, `D_ref_hi_um2_s`: D_ref's profile-likelihood interval, alpha profiled out. It is the
  projection of the region {l >= max l - chi2_1(level)/2} onto D_ref. `D_ref_lo` is 0 when the region reaches
  K = 0: the track only bounds D_ref from above.
- `D_ref_floor_um2_s` = <s²>/tau_ref, the D_ref at which the motion's MSD at tau_ref equals the noise's 2<s²>.
- `loglik_max`: the maximized log-likelihood.

With `keep_profiles=True`, `result.profiles` holds every "ok" track's l_p(alpha) - max l_p.

**D_ref = MSD_1D(tau_ref) / (2 tau_ref)** is the true (unblurred) motion's D at the lag `tau_ref_s` (default
0.1 s). At alpha = 1 it is the Brownian D at any tau_ref. Otherwise tau_ref is part of its definition, so keep
it fixed in seconds across datasets. It is not a Brownian D: use `gridpost` for how fast tracks move.

**Coverage** (`scripts/validate_fbmpost.py`: fBm simulated on a fine time grid and averaged over each exposure,
dt 44 ms, exposure 20 ms, SD 13 nm, alpha 0.3-1.4, D_ref 1e-3 to 3e-2, 40 tracks per cell, nominal 95%):

| frames | alpha interval | D_ref interval |
|---|---|---|
| 10 | 0.99 | 0.98 |
| 20 | 0.96 | 0.98 |
| 40 | 0.95 | 0.96 |

At 10 frames up to 40% of the subdiffusive tracks have alpha_hat at a grid end. Their intervals are wide, which
is why coverage there runs above nominal.

## One alpha for all tracks: `shared_alpha`

`shared_alpha(result.profiles)` sums the tracks' profiles and reports the maximum, its likelihood-ratio interval
and the test of alpha = 1. Each track keeps its own amplitude. This is complete pooling, the alpha counterpart of
`gridpost`'s shared D. When tracks differ in alpha it lands near an information-weighted average and its
interval is too narrow for the spread; the spread is the population's job (next section).

Profiles add because each is a likelihood. Curves with K integrated under a prior do not: the prior would enter
once per track. In the coverage study above the shared interval covered the true alpha in 45 of 48 cells.

## Population of alpha and D_ref

How alpha varies across tracks cannot be read from the alpha profiles alone, because they say nothing about how
alpha and D_ref vary together, and near the floor that relation decides what each track's alpha curve looks like.
The population is fitted on both axes:

```python
from diffusionkit.fbmpost import deconvolve_tracks

fbm = analyze_tracks(tracks, acquisition, FbmPostOptions(), keep_likelihoods=True)
pop = deconvolve_tracks(fbm.likelihoods)           # or fbm.likelihoods.rows(mask) for a subset
pop.alpha_weights(), pop.alpha_band(.9)            # alpha's distribution across tracks, pointwise band
pop.p_alpha_below(1.), pop.alpha_sd(), pop.corr()  # one value per posterior draw: take quantiles
```

`keep_likelihoods=True` keeps each "ok" track's likelihood on (alpha, log10 D_ref) cells (`JointGrid`: alpha
cells 0.06 wide, D_ref cells 0.2 decades wide over 1e-6 to 1e2), averaged over each cell from points inside it
(`likelihood.cells`), so a narrow likelihood is not aliased. No prior enters. `deconvolve_tracks` fits
g(alpha, log10 D_ref) with `gridpost.deconvolve_2d`: a smooth log density, with the smoothness along each axis
chosen by maximizing the Laplace-approximated evidence, then HMC draws at that smoothness. The penalty's null
space includes alpha × log10 D_ref, so a correlation between them is not smoothed toward zero. Alpha's
distribution is g's margin.

The fit uses only the D_ref cells some track tells apart (`informative_range`). Above them every track is at
least 5 nats below its best; below them every track is on its noise-only plateau to 1%.

Below the floor the tracks barely tell D_ref values apart, so g's alpha there is mostly the smoothness prior
continuing alpha from above. `AlphaDPopulation`'s `above_floor` reads leave that part out, and
`mass_below_floor` says how much of the population it is.

## Limits

- **Localization SDs are taken as exact.** Noise beyond the reported SDs looks like motion with alpha near 0:
  under-reported SDs read as low alpha.
- **Model.** fBm is the only Gaussian, stationary-increment model with a power-law MSD. Confinement whose cage
  relaxes within about a frame looks like extra static noise, which is fBm's alpha -> 0 limit: such tracks put
  alpha_hat at the low end of the grid. A single-frame linking outlier also pushes alpha down.
- **Cost.** The whitening is O(n_alpha m³) per track and dominates long tracks; the profile is O(n_alpha m)
  per Newton step. One thread: about 6 ms per track at 10 frames, 40 ms at 40, 0.2 s at 100 and 1.8 s at 300.
  `keep_likelihoods` adds the cells. Pass a pool's `map` as `map_fn` to use more cores.
