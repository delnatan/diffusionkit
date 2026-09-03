"""Diffusion-coefficient and anisotropy analysis for single-particle tracking.

Two pipelines over the same localization table, imported separately:

    from diffusionkit import classic    # MSD-curve fitting
    from diffusionkit import bayes      # exact-likelihood Bayesian inference

Only the loader is re-exported here, since both pipelines take the same
`load_tracks` DataFrame. Importing `diffusionkit` therefore costs polars and
nothing else -- `diffusionkit.bayes` pulls in jax/numpyro, and each pipeline's
`viz` submodule pulls in matplotlib, only when you ask for them.
"""
from .classic.io import AcquisitionParams, assert_contiguous_tracks, load_tracks

__version__ = "0.1.0"
