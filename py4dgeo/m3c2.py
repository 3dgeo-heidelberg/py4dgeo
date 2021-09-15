from py4dgeo.directions import Direction, MultiScaleDirection
from py4dgeo.epoch import Epoch
from py4dgeo.util import Py4DGeoError

import abc
import dataclasses
import numpy as np


@dataclasses.dataclass
class M3C2LikeAlgorithm(abc.ABC):
    epochs: tuple[Epoch] = None
    corepoints: np.ndarray = None
    radii: list[float] = None
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

        # Calculate the directions if they were not given
        if self.directions is None:
            self.directions = self.calculate_directions()

    @property
    def name(self):
        raise NotImplementedError

    def calculate_directions(self):
        raise NotImplementedError

    def check_number_of_epochs(self):
        if len(self.epochs) != 2:
            raise Py4DGeoError(
                f"{self.name} only operates on exactly 2 epochs, {len(self.epochs)} given!"
            )

    def run(self):
        # Make sure to precompute the directions
        self.directions.precompute(epoch=self.epochs[0], corepoints=self.corepoints)

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
                    epoch=self.epochs[0], corepoint_index=cix, radius=radius
                )
                directions = self.directions.get(cix)
                for eix, epoch in enumerate(self.epochs[1:]):
                    diff_points = self.corepoint_vicinity(
                        epoch=epoch, corepoint_index=cix, radius=radius
                    )
                    for dix, direction in enumerate(directions):
                        distances[cix, rix, dix, eix], _ = m3c2_distance(
                            p1=ref_points,
                            p2=diff_points,
                            corepoint=self.corepoints[cix, :],
                            direction=direction,
                        )

        return distances

    def corepoint_vicinity(self, epoch=None, corepoint_index=None, radius=None):
        # TODO: Check with 3DGeo: Is the difference between "in radius of core point"
        #       and "in cylinder around core point" important?
        indices, _ = epoch.kdtree.radius_search(
            self.corepoints[corepoint_index, :], radius
        )
        points = epoch.cloud[indices, :]
        return points


@dataclasses.dataclass
class M3C2(M3C2LikeAlgorithm):
    scales: list[float] = None

    @property
    def name(self):
        return "M3C2"

    def calculate_directions(self):
        return MultiScaleDirection(scales=self.scales)


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
    p1_dist = np.inner(p1 - corepoint, direction)
    p2_dist = np.inner(p2 - corepoint, direction)

    return abs(np.mean(p1_dist) - np.mean(p2_dist)), 0.0
