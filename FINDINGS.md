# Current validation status

The old production recommendations and comparative accuracy claims are
withdrawn. They are preserved as historical records in
[docs/archive/FINDINGS.md](docs/archive/FINDINGS.md), not as support for the
rebuilt API. [AUDIT.md](AUDIT.md) describes the problems motivating the reset.

The new classical workflow has tests for input validation, exact pair-specific
localization correction, known mean-curve recovery, nonlinear fitting against
an independent profile-grid calculation, and Brownian recovery under varying
localization precision. These verify implementation properties; they do not
establish precise per-track inference from five observations.

A reproducible short-track study is in `scripts/validate_classic.py` and its
recorded output is in [audit/classic_validation.json](audit/classic_validation.json).
It varies length (5, 10, 20), K (0.01, 0.05), and alpha (0.5, 1, 1.5), with
100 independent tracks per cell. Position errors vary by frame and axis.
Its position-space simulator is separate from all fitting algorithms.
Reported summaries retain boundary estimates and give status counts; null
estimates have explicit counts. This study assumes the localization SDs are
known exactly, and does not include blur, linking mistakes, drift or confinement.

In the recorded run (seed 20260918), five-frame Brownian tracks at
D=0.01 µm²/s had 18 negative D estimates out of 100, all retained and flagged.
The alpha fits for that cell had 38 boundary fits and 18 unidentified fits.
At 20 frames and D=0.05, the alpha RMSE was 0.318, with no boundary or
unidentified results in that 100-track cell. These counts illustrate why
numerical status alone does not establish precision. The 1,800-track study
completed without a reported optimizer failure; this is not a guarantee
for arbitrary data or settings.

An end-to-end smoke run on `mobile_beads_1to200.csv` (0.1043 µm/pixel,
0.033 s/frame) returned all 539 tracks: 1,078 model-fit rows and 1,617 MSD
rows. It flagged one negative D, 31 alpha boundary fits and one unidentified
alpha. The dataset has no ground-truth parameters, so this verifies the
workflow, not the scientific correctness of those estimates.

No classical confidence intervals are estimated or calibrated. The default
three-lag window is a transparent baseline, not an optimized recommendation.
Bayesian priors, transformed MAP semantics, Laplace accuracy and diagnostics
remain for the next revision. Anisotropy is outside the current scope.

## Brownian MLE and non-Brownian score (2026-09-18)

Study: `scripts/validate_brownian_mle.py`, with recorded output in
[audit/brownian_mle_validation.json](audit/brownian_mle_validation.json)
(seed 20260918, 400 tracks per cell, 500 bootstrap replicates, dt = 0.033 s,
per-frame localization SDs of 10–40 nm, independent simulators). The SDs are
known exactly unless stated otherwise.

**Null calibration.** For Brownian tracks at N = 5–20 and D = 0.01–0.2 µm²/s,
with and without a full-frame exposure, the calibrated z had SD 0.93–1.07,
|z| > 1.96 in 2.8–7.7% of tracks, and |mean| ≤ 0.09. The Spearman
correlation with log D_hat was within ±0.15. The uncalibrated asymptotic z
was narrower (SD 0.82–1.05). For pure localization noise (true D = 0), about
61% of tracks gave D_hat = 0, and `p_motion < 0.05` occurred in 2.0–4.5%.
Noise-only tracks that did resolve had z correlated with log D_hat
(ρ ≈ -0.3).

**D.** MLE bias was ≤ 6% in every cell. Its relative RMSE was lower than the
3-lag MSD fit everywhere, for example 0.71 vs 0.92 (N = 5, D = 0.05) and
0.28 vs 0.33 (N = 20).

**Detection per track is weak at short lengths; population shifts are
not.** At N = 5, the score's rejection rate at 5% was 4–8% for fBm
alpha = 0.5–1.5, OU confinement and drift, so single tracks cannot be
classified. The MSD alpha fit, thresholded at its own null 95% range, never
rejected: its null spread fills [0, 2]. The mean z nevertheless shifted by
-0.71 (alpha = 0.5), -0.67 (strong OU) and +0.42 (alpha = 1.5), which is
visible in a population of about 100 tracks. At N = 20, the score detected
alpha = 0.5 in 50% of tracks, strong OU in 65% and drift in 37%. The
windowed MSD alpha detected those in 31%, 50% and 15%, but was better for
superdiffusive fBm (59% vs 43% at alpha = 1.5).

**Calibration inputs dominate small shifts.** Reporting SDs 20% too small or
too large shifted the mean z by about -0.2 and +0.1 to +0.26, and biased D
by +7% to +15% and -7% to -15%. Ignoring a full-frame exposure shifted the
mean z by +0.3 to +0.7 and biased D by -25%. These are as large as the
effect of alpha = 0.75. Population z shifts are therefore interpretable
only once exposure and localization-SD calibration are fixed.

**Bead control.** `mobile_beads_1to200.csv` was acquired with a 35 ms frame
interval and 20 ms exposure (earlier notes recorded 33 ms). Settings: 0.1043
µm/px, `Acquisition(dt_s=.035, exposure_s=.020)`. All 539 tracks resolved
motion. The mean z was -0.08 ± 0.05, the SD 1.07, and |z| > 1.96 occurred
in 8.0% of tracks. The median D was 0.052 µm²/s. Omitting the exposure
instead gave a mean z of +0.56 ± 0.05 and a median D of 0.044, which is the
blur artifact predicted above. Residual departures are small: 5–9-frame
tracks had a mean z of -0.22 ± 0.08, and the highest-D quintile -0.34 ±
0.11. Both are consistent with a few linking errors (jump and return) or
SDs that are about 10% too small. Scaling SDs by 1.1 brings the overall
mean z to -0.01. This dataset does not separate those two causes.
