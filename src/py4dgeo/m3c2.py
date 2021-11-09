from ._py4dgeo import (
    compute_distances,
    no_uncertainty,
    radius_workingset_finder,
    standard_deviation_uncertainty,
)
from py4dgeo.directions import Direction, MultiScaleDirection
from py4dgeo.epoch import Epoch, as_epoch
from py4dgeo.util import (
    as_double_precision,
    MemoryPolicy,
    Py4DGeoError,
    make_contiguous,
    memory_policy_is_minimum,
)

import abc
import numpy as np
import typing


class M3C2LikeAlgorithm(abc.ABC):
    def __init__(
        self,
        epochs: typing.Tuple[Epoch, ...],
        corepoints: np.ndarray = None,
        radii: typing.List[float] = None,
        max_cylinder_length: float = 0.0,
        directions: Direction = None,
        calculate_uncertainty: bool = True,
    ):
        self.epochs = epochs
        self.corepoints = as_double_precision(make_contiguous(corepoints))
        self.radii = radii
        self.max_cylinder_length = max_cylinder_length
        self.directions = directions
        self.calculate_uncertainty = calculate_uncertainty

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

    def calculate_distances(self, epoch1, epoch2):
        """Calculate the distances between two epochs"""

        # Find the correct epoch to use for normal calculation
        normals_epoch = self.cloud_for_normals
        if normals_epoch is None:
            normals_epoch = epoch1
        normals_epoch = as_epoch(normals_epoch)

        # Make sure to precompute the directions
        self.directions.precompute(epoch=normals_epoch, corepoints=self.corepoints)

        assert len(self.radii) == 1

        # Extract the uncertainty callback
        uncertainty_callback = self.callback_uncertainty_calculation()
        if not self.calculate_uncertainty:
            uncertainty_callback = no_uncertainty

        distances, uncertainties = compute_distances(
            self.corepoints,
            self.radii[0],
            epoch1,
            epoch2,
            self.directions.directions,
            self.max_cylinder_length,
            self.callback_workingset_finder(),
            uncertainty_callback,
        )

        return distances, uncertainties

    def run(self):
        """Main entry point for running the algorithm"""
        return self.calculate_distances(self.epochs[0], self.epochs[1])

    def callback_workingset_finder(self):
        """The callback used to determine the point cloud subset around a corepoint"""
        if memory_policy_is_minimum(MemoryPolicy.COREPOINTS):
            return radius_workingset_finder
        else:
            raise NotImplementedError(
                "No implementation of workingset_finder for your memory policy yet"
            )

    def callback_uncertainty_calculation(self):
        """The callback used to calculate the uncertainty"""
        return standard_deviation_uncertainty


class M3C2(M3C2LikeAlgorithm):
    def __init__(
        self,
        scales: typing.List[float] = None,
        orientation_vector: np.ndarray = np.array([0, 0, 1]),
        cloud_for_normals: Epoch = None,
        **kwargs,
    ):
        self.scales = scales
        self.orientation_vector = as_double_precision(
            make_contiguous(orientation_vector)
        )
        self.cloud_for_normals = cloud_for_normals
        super().__init__(**kwargs)

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
        radius_candidates.append(self.max_cylinder_length)
        maxradius = max(radius_candidates)

        for epoch in self.epochs:
            epoch.kdtree.precompute(self.corepoints, maxradius)

    def calculate_directions(self):
        return MultiScaleDirection(
            scales=self.scales, orientation_vector=self.orientation_vector
        )
