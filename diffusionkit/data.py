"""Acquisition metadata shared by track-analysis algorithms."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Acquisition:
    dt_s: float
    exposure_s: float = 0.0
