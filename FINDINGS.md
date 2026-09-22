# Current validation status

The old production recommendations and comparative accuracy claims from
before the classical rebuild are withdrawn; they are not support for the
current API.

The classical workflow has tests for input validation, exact pair-specific
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
