# Drift

`diffusionkit.drift` estimates the displacement field that every track shares (stage drift, slow tissue motion) from
the tracks themselves, and subtracts it before any per-track analysis.

```python
from diffusionkit.drift import estimate_drift, neighbour_correlation, subtract

dr = estimate_drift(tracks, acquisition, degree=2)   # Legendre field of total degree 2 over the field of view
dr.at((x_um, y_um))                                  # (n_frames, 2) drift path there, um
dr.path_cov((x_um, y_um))                            # (2, n, n) its covariance
corrected = subtract(tracks, dr)                     # field evaluated at each track's mean position
neighbour_correlation(corrected)                     # is motion still shared between neighbours?
```

## Model

Per axis, track i's displacements are

    delta_i = S_i dd phi(x_i) + e_i,    e_i ~ N(0, D_i A + B_i)

- dd are the field's per-frame increments in the polynomial basis phi, evaluated at the track's mean position.
- A and B_i are `gridpost`'s blurred Brownian and per-frame localization covariances. Positions are exposure
  averages, so subtracting the estimated (exposure-averaged) drift is exact.
- D_i is unknown. It is marginalized by EM over a D grid with a flat prior in ln D, and the M-step is generalized
  least squares with W_i = E[Sigma_i^-1].
  - Still spots carry the estimate: a GEM at D = 0.3 um^2/s gets ~1/25 of a still spot's weight.
  - Nothing is classified and no length cut is needed.

`degree=0` is one rigid path, 1 adds rotation, shear and stretch, and 2 a smooth bend. The increments' covariance
(`dd_cov`) is shared by every track. It is a correlated error, and at a few nm over a movie it is negligible for D
(~3e-5 um^2/s even at 3x).

## Choosing the degree, and local motion

Fit degree 2, then check what is left with `neighbour_correlation`. For each long track (>= 40 frames), the median net
displacement of its neighbours within r predicts its own. Compare the correlation with the shuffled null
(`null_mean` +- `null_sd`).
- Shared drift on a scale L gives a correlation well above the null for r < L.
- A particle's own motion gives the null.

Motion still shared after degree 2 is local deformation. A smooth local flow v (um per frame) biases a track's D by
about |v|^2 / (4 dt (1 - 2R)), with R = exposure / (6 dt). That is ~2e-5 um^2/s for 1 nm/frame, 5e-4 for 5 nm/frame
and 5e-3 for 17 nm/frame. Compare it with the slowest population's D before correcting further.

On C. elegans hypodermis GEM movies (49 frames of 20 ms):
- **anc-1:** 800 nm of global drift. Uncorrected, it moved the whole slow population (D ~ 1e-3) to 0.01-0.05.
- **wt_3:** the global drift was ~0, but there was local deformation of ~1 nm/frame (neighbour correlation 0.48 at
  8 um). A space-time flow model removed it and changed neither the D population nor step memory. That model is kept
  as a prototype, not in the package.
- Two shortcuts don't work:
  - A per-frame neighbour average within a fixed radius is noisier than the flow itself (SE ~3 nm per frame at
    5 um). Subtracting it biases slow D by ~3e-4.
  - A graph with every track step as a node had too many degrees of freedom, and the evidence rejected any flow.

## Validation (`tests/test_drift.py`)

- A rigid ~0.3 um path is recovered to < 4 nm rms from 600 mixed still/mobile tracks.
- With no drift, the estimate stays within its reported SE.
- Frame offsets are handled, gaps raise, and the neighbour check separates shared motion from own motion.
- On the GEM movies the package reproduces the prototype exactly (anc-1: -802 nm in x over 48 frames).
