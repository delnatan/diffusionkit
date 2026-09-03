"""Classic pipeline: fit a model to the mean-squared-displacement curve.

`fit_population` is the one-call entry point; everything it composes is
exposed individually below. Plotting lives in `diffusionkit.classic.viz` and
is deliberately not re-exported here, so importing this package does not pull
in matplotlib.
"""
from .api import PopulationFit, fit_population
from .fitting import (
    AnomalousDiffusionFit,
    NormalDiffusionFit,
    fit_all_tracks,
    fit_anomalous_diffusion,
    fit_normal_diffusion,
    localization_offset_by_track,
    n_fit_points,
    weighted_expected_offset,
)
from .io import AcquisitionParams, assert_contiguous_tracks, load_tracks
from .msd import compute_all_tamsd, ensemble_average_msd
from .simulate import simulate_brownian_tracks
