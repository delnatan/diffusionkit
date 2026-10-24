"""Spatial classes of D: tracks grouped by a hidden Potts model on their neighbour graph, read as populations.

Classes come from each track's Brownian D likelihood (`gridpost`) and its neighbours'. A class is a group of tracks
(`SpatialAnalysis.groups`, by the class the neighbours give each track); its D and alpha are that group's
populations (`gridpost.deconvolve_tracks`, `fbmpost.deconvolve_tracks`). Nothing is refitted per track.
"""
from . import potts
from .data import ShuffleNull, SpatialAnalysis, SpatialOptions
from .workflow import analyze, analyze_likelihoods, centroids, restore

__all__ = ["potts", "ShuffleNull", "SpatialAnalysis", "SpatialOptions", "analyze", "analyze_likelihoods", "centroids",
           "restore"]
