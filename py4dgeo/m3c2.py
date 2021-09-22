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
        self.directions.precompute(
            epoch=self.epochs[0],
            corepoints=self.corepoints,
            radius_searcher=lambda idx, r: self.radius_search_around_corepoint(
                0, idx, r
            ),
        )

        # Correctly shape the distance array
        distances = np.empty(
            (
                self.corepoints.shape[0],
                len(self.radii),
                self.directions.num_dirs,
                len(self.epochs) - 1,
            )
        )

        # TODO: Decisions necessary how to
        # * Order these loops
        # * Allow more "bulk" interfaces
        # * Decide which parts to implement in C++
        for rix, radius in enumerate(self.radii):
            for cix in range(self.corepoints.shape[0]):
                ref_points = self.corepoint_vicinity(
                    epoch_idx=0, corepoint_index=cix, radius=radius
                )
                directions = self.directions.get(cix)
                for eix in range(1, len(self.epochs)):
                    diff_points = self.corepoint_vicinity(
                        epoch_idx=eix, corepoint_index=cix, radius=radius
                    )
                    for dix, direction in enumerate(directions):
                        distances[cix, rix, dix, eix - 1], _ = m3c2_distance(
                            p1=ref_points,
                            p2=diff_points,
                            corepoint=self.corepoints[cix, :],
                            direction=direction,
                        )

        return distances

    def corepoint_vicinity(self, epoch_idx=None, corepoint_index=None, radius=None):
        # TODO: Check with 3DGeo: Is the difference between "in radius of core point"
        #       and "in cylinder around core point" important?
        indices = self.radius_search_around_corepoint(
            epoch_idx, corepoint_index, radius
        )
        return self.epochs[epoch_idx].cloud[indices, :]


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

    def radius_search_around_corepoint(self, epoch_idx, core_idx, radius):
        return self.epochs[epoch_idx].kdtree.precomputed_radius_search(core_idx, radius)


def m3c2_distance(
    p1: np.ndarray, p2: np.ndarray, corepoint: np.ndarray, direction: np.ndarray
) -> tuple:
    """Calculates M3C2 distance between two point clouds P1 and P2 in direction.

    :param p1: Point cloud of the first epoch (n x 3) array
    :param p2: Point cloud of the second epoch (m x 3) array
    :param direction: vector showing the direction of the change to be investigated - (1 x 3) array
    :return: The distance and its uncertainty
    :rtype: tuple
    """
    return abs(np.inner(np.mean(p1, axis=0) - np.mean(p2, axis=0), direction)), 0.0
