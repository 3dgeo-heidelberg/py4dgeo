from py4dgeo.util import Py4DGeoError

import abc
import dataclasses
import numpy as np
import typing


@dataclasses.dataclass(frozen=True)
class Direction(abc.ABC):
    @property
    def num_dirs(self):
        raise NotImplementedError

    def precompute(self, epoch=None, corepoints=None):
        pass

    def get(self, core_idx=None) -> np.ndarray:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class MultiConstantDirection(Direction):
    directions: np.ndarray = None

    @property
    def num_dirs(self):
        return self.directions.shape[0]

    def get(self, core_idx=None) -> np.ndarray:
        return self.directions


class ConstantDirection(MultiConstantDirection):
    def __init__(self, direction=None):
        super(ConstantDirection, self).__init__(directions=direction[np.newaxis])


@dataclasses.dataclass(frozen=True)
class CorePointDirection(Direction):
    directions: np.ndarray = None

    @property
    def num_dirs(self):
        return self.directions.shape[1]

    def get(self, core_idx=None) -> np.ndarray:
        return self.directions[core_idx, :, :]


@dataclasses.dataclass(frozen=True)
class PrecomputedDirection(Direction):
    _precomputation: list = dataclasses.field(default_factory=lambda: [], init=False)


@dataclasses.dataclass(frozen=True)
class MultiScaleDirection(PrecomputedDirection):
    scales: typing.List[float] = None

    def __post_init__(self):
        # Check the validity of the scales parameter
        if self.scales is None or len(self.scales) == 0:
            raise Py4DGeoError(
                f"{self.name} requires at least one scale radius to be given"
            )

    @property
    def num_dirs(self):
        return 1

    def precompute(self, epoch=None, corepoints=None, radius_searcher=None):
        # This is a Python placeholder for a C++ implementation of the multiscale
        # direction implementation. Some notes already gathered:
        # * https://eigen.tuxfamily.org/dox/group__TutorialSlicingIndexing.html (see Array of Indices)
        # * https://stackoverflow.com/a/15142446
        # * Radii too small to produce a good covariance matrix need to be detected
        if epoch is None or corepoints is None:
            raise ValueError("epoch and corepoints need to be provided to precompute")

        if radius_searcher is None:
            radius_searcher = lambda idx, r: epoch.kdtree.radius_search(
                corepoints[idx, :], r
            )

        # Reset precomputation results
        self._precomputation.clear()

        # Results to update iteratively
        result = np.zeros(shape=(corepoints.shape[0], 1, corepoints.shape[1]))

        for core_idx in range(corepoints.shape[0]):
            highest_planarity = 0.0
            for scale in self.scales:
                points_idx = radius_searcher(core_idx, scale)
                points_subs = epoch.cloud[points_idx, :]
                cxx = np.cov(points_subs.T)
                eigval, eigvec = np.linalg.eigh(cxx)
                planarity = (eigval[1] - eigval[0]) / eigval[2]

                if planarity > highest_planarity:
                    highest_planarity = planarity
                    result[core_idx, 0, :] = eigvec[:, 2]

        # Store the result
        self._precomputation.append(result)

    def get(self, core_idx=None) -> np.ndarray:
        return self._precomputation[0][core_idx, :]
