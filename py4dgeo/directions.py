from py4dgeo.util import Py4DGeoError

import abc
import dataclasses
import numpy as np
import typing

import _py4dgeo


@dataclasses.dataclass(frozen=True)
class Direction(abc.ABC):
    @property
    def num_dirs(self):
        raise NotImplementedError

    def precompute(self, epoch=None, corepoints=None):
        pass

    def get(self, core_idx=None, dir_idx=None) -> np.ndarray:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class MultiConstantDirection(Direction):
    directions: np.ndarray = None

    @property
    def num_dirs(self):
        return self.directions.shape[0]

    def get(self, core_idx=None, dir_idx=None) -> np.ndarray:
        return self.directions[dir_idx, :]


class ConstantDirection(MultiConstantDirection):
    def __init__(self, direction=None):
        super(ConstantDirection, self).__init__(directions=direction[np.newaxis])


@dataclasses.dataclass(frozen=True)
class CorePointDirection(Direction):
    directions: np.ndarray = None

    @property
    def num_dirs(self):
        return self.directions.shape[1]

    def get(self, core_idx=None, dir_idx=None) -> np.ndarray:
        return self.directions[core_idx, dir_idx, :]


@dataclasses.dataclass(frozen=True)
class PrecomputedDirection(Direction):
    _precomputation: list = dataclasses.field(default_factory=lambda: [], init=False)

    @property
    def directions(self):
        return self._precomputation[0]


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

    def precompute(self, epoch=None, corepoints=None):
        self._precomputation.clear()

        result = np.empty(corepoints.shape)
        _py4dgeo.compute_multiscale_directions(
            epoch.cloud, corepoints, self.scales, epoch.kdtree, result
        )

        self._precomputation.append(result)

    def get(self, core_idx=None, dir_idx=None) -> np.ndarray:
        return self._precomputation[0][core_idx, :]
