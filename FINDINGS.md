# Current validation status

The old production recommendations and comparative accuracy claims from
before the classical rebuild are withdrawn; they are not support for the
current API.

The classical workflow has tests for input validation, exact pair-specific
localization correction, known mean-curve recovery, nonlinear fitting against
an independent profile-grid calculation, and Brownian recovery under varying
localization precision. These verify implementation properties; they do not
establish precise per-track inference from five observations.

The distribution of D across tracks (`gridpost.deconvolve_tracks`) has a
coverage study in `scripts/validate_deconvolve.py`, recorded in
[audit/deconvolve_validation.json](audit/deconvolve_validation.json): five
simulated populations, 40 datasets of 1000 tracks each, on the default grid.
On the 1e-5..10 grid, its 68%/95% bands covered the population CDF in
69-73%/89-98% of datasets, and no meaningful mass appeared past the tracks
(97.5% quantile above 1 um^2/s at most 0.003). See docs/gridpost.md for the table and limits.

Drift (`diffusionkit.drift`) is tested in `tests/test_drift.py`:
- a simulated rigid path is recovered to < 4 nm rms;
- with no drift the estimate stays within its reported SE;
- the neighbour check separates shared motion from each particle's own.

On three C. elegans hypodermis GEM movies, uncorrected drift (800 nm in anc-1) moved the whole slow population
(D ~ 1e-3 um^2/s) to 0.01-0.05. A local flow in one movie (~1 nm/frame) changed neither the D population nor the
step-memory summaries, consistent with its ~2e-5 um^2/s bias on D (docs/drift.md).

The per-track alpha posterior, the joint (alpha, D) posterior and its deconvolution were removed from `gridpost`
after commit 540ba4a. Per-track alpha was weakly identified on short tracks, leaned low for immobile tracks, and
confounded localization-SD errors with caging (docs/gridpost.md, "What is reported").

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
