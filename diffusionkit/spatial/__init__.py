"""Spatial classes of D: tracks grouped by a hidden Potts model on their neighbour graph, read as populations.

Classes come from each track's Brownian D likelihood (`gridpost`) and its neighbours'. A class is a group of tracks
(`SpatialAnalysis.groups`, by the class the neighbours give each track); its D and alpha are that group's
populations (`gridpost.deconvolve_tracks`, `fbmpost.deconvolve_tracks`). Nothing is refitted per track.

`smooth_map` asks a different question with no classes: how ln D varies over the cell (`smooth`, a BYM map on the
same graph, with its smoothness chosen by evidence).
"""
from . import potts
from . import smooth
from .data import ShuffleNull, SpatialAnalysis, SpatialOptions
from .smooth import MapNull, SmoothMap, SmoothMapOptions, smooth_map, smooth_map_likelihoods
from .workflow import analyze, analyze_likelihoods, centroids, restore

__all__ = ["potts", "smooth", "MapNull", "ShuffleNull", "SmoothMap", "SmoothMapOptions", "SpatialAnalysis",
           "SpatialOptions", "analyze", "analyze_likelihoods", "centroids", "restore", "smooth_map",
           "smooth_map_likelihoods"]
