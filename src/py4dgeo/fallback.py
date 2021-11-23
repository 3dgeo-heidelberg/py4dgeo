"""Fallback implementations for C++ components of the M3C2 algorithms """

from py4dgeo.epoch import Epoch
from py4dgeo.m3c2 import M3C2

import numpy as np
import py4dgeo._py4dgeo as _py4dgeo


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
    # Cut the cylinder into N segments, perform radius searches around the
    # segment midpoints and create the union of indices
    N = 1
    if max_cylinder_length >= radius:
        N = int(np.ceil(max_cylinder_length / radius))
    else:
        max_cylinder_length = radius

    r_cyl = np.sqrt(
        radius * radius + max_cylinder_length * max_cylinder_length / (N * N)
    )
    indices = np.unique(
        np.concatenate(
            tuple(
                epoch.kdtree.radius_search(
                    corepoint[0, :]
                    + ((2 * i - N + 1) / N) * max_cylinder_length * direction[0, :],
                    r_cyl,
                )
                for i in range(N)
            )
        )
    )

    # Gather the points from the point cloud
    superset = epoch.cloud[indices, :]

    # And cut away those points that are too far away from the cylinder axis
    crossprod = np.cross(superset - corepoint[0, :], direction[0, :])
    distances = np.sum(crossprod * crossprod, axis=1)
    return superset[distances < radius * radius, :]


def no_uncertainty(
    set1: np.ndarray, set2: np.ndarray, direction: np.ndarray
) -> _py4dgeo.DistanceUncertainty:
    return _py4dgeo.DistanceUncertainty()


def standard_deviation_uncertainty(
    set1: np.ndarray, set2: np.ndarray, direction: np.ndarray
) -> _py4dgeo.DistanceUncertainty:
    # Calculate variances
    variance1 = direction @ np.cov(set1.T) @ direction.T
    variance2 = direction @ np.cov(set2.T) @ direction.T

    # The structured array that describes the full uncertainty
    return _py4dgeo.DistanceUncertainty(
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
        return cylinder_workingset_finder

    def callback_uncertainty_calculation(self):
        return standard_deviation_uncertainty
