"""Compatibility import for archived code; use diffusionkit.classic.analyze_tracks."""
from ..legacy.classic.covariance import *  # noqa: F403
from ..legacy.classic import covariance as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
