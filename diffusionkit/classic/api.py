"""Compatibility import for archived code; use diffusionkit.classic.analyze_tracks."""
from ..legacy.classic.api import *  # noqa: F403
from ..legacy.classic import api as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
