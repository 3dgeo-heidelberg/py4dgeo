from py4dgeo.util import Py4DGeoError, memory_policy_is_minimum, MemoryPolicy

import abc
import numpy as np
import typing

import _py4dgeo


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
        self.directions = directions

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

    @property
    def num_dirs(self):
        return self.directions.shape[1]

    def get(self, core_idx=None, dir_idx=None):
        return self.directions[core_idx, dir_idx, :]


class MultiScaleDirection(Direction):
    def __init__(self, scales: typing.List[float] = None):
        self.scales = scales
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
        _py4dgeo.compute_multiscale_directions(
            epoch, corepoints, self.scales, self.directions
        )

    def get(self, core_idx=None, dir_idx=None):
        return self.directions[core_idx, :]
