"""Fallback implementations for C++ components of the M3C2 algorithms """

from ._py4dgeo import DistanceUncertainty
from py4dgeo.epoch import Epoch
from py4dgeo.m3c2 import M3C2

import numpy as np


def radius_workingset_finder(
    epoch: Epoch,
    radius: float,
    corepoint: np.ndarray,
    direction: np.ndarray,
    max_cylinder_length: float,
    core_idx: int,
) -> np.ndarray:
    indices = epoch.kdtree.precomputed_radius_search(core_idx, radius)
    return epoch.cloud[indices, :]


def cylinder_workingset_finder(
    epoch: Epoch,
    radius: float,
    corepoint: np.ndarray,
    direction: np.ndarray,
    max_cylinder_length: float,
    core_idx: int,
) -> np.ndarray:
    # The search radius is the maximum of cylinder length and radius
    search_radius = max(radius, max_cylinder_length)

    # Find the points in the radius of max_cylinder_length
    indices = epoch.kdtree.precomputed_radius_search(core_idx, search_radius)
    superset = epoch.cloud[indices, :]

    # If max_cylinder_length is sufficiently small, we are done
    if max_cylinder_length <= radius:
        return superset

    crossprod = np.cross(superset - corepoint[0, :], direction[0, :])
    distances = np.sum(crossprod * crossprod, axis=1)

    return superset[distances < radius * radius, :]


def no_uncertainty(
    set1: np.ndarray, set2: np.ndarray, direction: np.ndarray
) -> DistanceUncertainty:
    return DistanceUncertainty()


def standard_deviation_uncertainty(
    set1: np.ndarray, set2: np.ndarray, direction: np.ndarray
) -> DistanceUncertainty:
    # Calculate variances
    variance1 = direction @ np.cov(set1.T) @ direction.T
    variance2 = direction @ np.cov(set2.T) @ direction.T

    # The structured array that describes the full uncertainty
    return DistanceUncertainty(
        lodetection=1.96
        * np.sqrt(variance1 / set1.shape[0] + variance2 / set2.shape[0]),
        stddev1=np.sqrt(variance1),
        num_samples1=set1.shape[0],
        stddev2=np.sqrt(variance2),
        num_samples2=set2.shape[0],
    )


class PythonFallbackM3C2(M3C2):
    """An implementation of M3C2 that makes use of Python fallback implementations"""

    @property
    def name(self):
        return "M3C2 (Python Fallback)"

    def callback_workingset_finder(self):
        return radius_workingset_finder

    def callback_uncertainty_calculation(self):
        return standard_deviation_uncertainty
