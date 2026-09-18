"""Compatibility import for archived code; use diffusionkit.classic.analyze_tracks."""
from ..legacy.classic.fitting import *  # noqa: F403
from ..legacy.classic import fitting as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
