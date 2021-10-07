"""Fallback implementations for C++ components of the M3C2 algorithms """

from py4dgeo.kdtree import KDTree
from py4dgeo.m3c2 import M3C2

import numpy as np


def radius_workingset_finder(
    cloud: np.ndarray,
    kdtree: KDTree,
    radius: float,
    corepoint: np.ndarray,
    direction: np.ndarray,
    max_cylinder_length: float,
    core_idx: int,
) -> np.ndarray:
    indices = kdtree.precomputed_radius_search(core_idx, radius)
    return cloud[indices, :]


def cylinder_workingset_finder(
    cloud: np.ndarray,
    kdtree: KDTree,
    radius: float,
    corepoint: np.ndarray,
    direction: np.ndarray,
    max_cylinder_length: float,
    core_idx: int,
) -> np.ndarray:
    raise NotImplementedError


class PythonFallbackM3C2(M3C2):
    """An implementation of M3C2 that makes use of Python fallback implementations"""

    @property
    def name(self):
        raise "M3C2 (Python Fallback)"

    def callback_workingset_finder(self):
        return radius_workingset_finder
