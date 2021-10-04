import _py4dgeo

from py4dgeo.directions import Direction, MultiScaleDirection
from py4dgeo.epoch import Epoch
from py4dgeo.util import Py4DGeoError

import abc
import dataclasses
import numpy as np
import typing


@dataclasses.dataclass
class M3C2LikeAlgorithm(abc.ABC):
    epochs: typing.Tuple[Epoch, ...] = None
    corepoints: np.ndarray = None
    radii: typing.List[float] = None
    directions: Direction = None

    def __post_init__(self):
        # Check the given array shapes
        if len(self.corepoints.shape) != 2 or self.corepoints.shape[1] != 3:
            raise Py4DGeoError("Corepoints need to be given as an array of shape nx3")

        # Check the given radii
        if self.radii is None or len(self.radii) == 0:
            raise Py4DGeoError(f"{self.name} requires at least one radius to be given")

        # Check the given number of epochs
        self.check_number_of_epochs()

        # Run setup code defined by the algorithm
        self.setup()

        # Calculate the directions if they were not given
        if self.directions is None:
            self.directions = self.calculate_directions()

    @property
    def name(self):
        raise NotImplementedError

    def setup(self):
        pass

    def calculate_directions(self):
        raise NotImplementedError

    def check_number_of_epochs(self):
        if len(self.epochs) != 2:
            raise Py4DGeoError(
                f"{self.name} only operates on exactly 2 epochs, {len(self.epochs)} given!"
            )

    def radius_search_around_corepoint(self, epoch_idx, core_idx, radius):
        """Perform a radius search around a core point

        By default, this will use the KDTree stored with the epoch. Alternatively,
        algorithm classes inheriting from this class can override this to provide a
        specialized search tree that e.g. implements a suitable caching strategy.
        """
        return self.epochs[epoch_idx].kdtree.radius_search(
            self.corepoints[core_idx, :], radius
        )

    def run(self):
        # Make sure to precompute the directions
        self.directions.precompute(epoch=self.epochs[0], corepoints=self.corepoints)

        assert len(self.radii) == 1

        # Allocate the result array
        result = np.empty((len(self.corepoints),))

        _py4dgeo.compute_distances(
            self.corepoints,
            self.radii[0],
            self.epochs[0].cloud,
            self.epochs[0].kdtree,
            self.epochs[1].cloud,
            self.epochs[1].kdtree,
            self.directions._precomputation[0],
            result,
        )

        return result


@dataclasses.dataclass
class M3C2(M3C2LikeAlgorithm):
    scales: typing.List[float] = None

    @property
    def name(self):
        return "M3C2"

    def setup(self):
        # Cache KDTree evaluations
        radius_candidates = []
        if self.scales is not None:
            radius_candidates.extend(list(self.scales))
        if self.radii is not None:
            radius_candidates.extend(list(self.radii))
        maxradius = max(radius_candidates)

        for epoch in self.epochs:
            epoch.kdtree.precompute(self.corepoints, maxradius)

    def calculate_directions(self):
        return MultiScaleDirection(scales=self.scales)
