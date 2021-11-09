from ._py4dgeo import compute_multiscale_directions
from py4dgeo.util import Py4DGeoError, memory_policy_is_minimum, MemoryPolicy

import abc
import numpy as np
import typing


def normalize_directions(dir: np.ndarray):
    """Normalize the given directions"""
    assert len(dir.shape) == 2
    assert dir.shape[1] == 3
    return dir / dir.sum(axis=1)[:, np.newaxis]


class Direction(abc.ABC):
    @property
    def num_dirs(self) -> int:
        raise NotImplementedError

    def precompute(self, epoch=None, corepoints=None) -> None:
        pass

    def get(self, core_idx=None, dir_idx=None) -> np.ndarray:
        raise NotImplementedError


class MultiConstantDirection(Direction):
    def __init__(self, directions: np.ndarray = None):
        self.directions = normalize_directions(directions)

    @property
    def num_dirs(self):
        return self.directions.shape[0]

    def get(self, core_idx=None, dir_idx=None):
        return self.directions[dir_idx, :]


class ConstantDirection(MultiConstantDirection):
    def __init__(self, direction: np.ndarray = None):
        super(ConstantDirection, self).__init__(directions=direction[np.newaxis])


class CorePointDirection(Direction):
    def __init__(self, directions: np.ndarray = None):
        self.directions = directions
        for i in range(directions.shape[0]):
            self.directions[i, :, :] = normalize_directions(self.directions[i, :, :])

    @property
    def num_dirs(self):
        return self.directions.shape[1]

    def get(self, core_idx=None, dir_idx=None):
        return self.directions[core_idx, dir_idx, :]


class MultiScaleDirection(Direction):
    def __init__(
        self,
        scales: typing.List[float] = None,
        orientation_vector: np.array = np.array([0.0, 0.0, 1.0]),
    ):
        self.scales = scales
        self.orientation_vector = orientation_vector
        self.directions = None

        # This is currently only implemented as a precomputation
        if not memory_policy_is_minimum(MemoryPolicy.COREPOINTS):
            raise NotImplementedError(
                "M3C2 normal direction not implemented for your memory policy"
            )

        # Check the validity of the scales parameter
        if self.scales is None or len(self.scales) == 0:
            raise Py4DGeoError(
                f"{self.name} requires at least one scale radius to be given"
            )

    @property
    def num_dirs(self):
        return 1

    def precompute(self, epoch=None, corepoints=None):
        self.directions = np.empty(corepoints.shape)
        compute_multiscale_directions(
            epoch, corepoints, self.scales, self.orientation_vector, self.directions
        )

    def get(self, core_idx=None, dir_idx=None):
        return self.directions[core_idx, :]
