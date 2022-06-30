from py4dgeo.epoch import Epoch, as_epoch
from py4dgeo.util import (
    as_double_precision,
    MemoryPolicy,
    Py4DGeoError,
    make_contiguous,
    memory_policy_is_minimum,
)

import abc
import logging
import numpy as np
import typing

import py4dgeo._py4dgeo as _py4dgeo


logger = logging.getLogger("py4dgeo")


class M3C2LikeAlgorithm(abc.ABC):
    def __init__(
        self,
        epochs: typing.Tuple[Epoch, ...] = None,
        corepoints: np.ndarray = None,
        cyl_radii: typing.List[float] = None,
        max_distance: float = 0.0,
        registration_error: float = 0.0,
        robust_aggr: bool = False,
    ):
        self.epochs = epochs
        self.corepoints = corepoints
        self.cyl_radii = cyl_radii
        self.max_distance = max_distance
        self.registration_error = registration_error
        self.robust_aggr = robust_aggr

    @property
    def corepoints(self):
        return self._corepoints

    @corepoints.setter
    def corepoints(self, _corepoints):
        if _corepoints is None:
            self._corepoints = None
        else:
            if len(_corepoints.shape) != 2 or _corepoints.shape[1] != 3:
                raise Py4DGeoError(
                    "Corepoints need to be given as an array of shape nx3"
                )
            self._corepoints = as_double_precision(make_contiguous(_corepoints))

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, _epochs):
        if _epochs is not None and len(_epochs) != 2:
            raise Py4DGeoError("Exactly two epochs need to be given!")
        self._epochs = _epochs

    @property
    def name(self):
        raise NotImplementedError

    def directions(self):
        """The normal direction(s) to use for this algorithm."""
        raise NotImplementedError

    def calculate_distances(self, epoch1, epoch2):
        """Calculate the distances between two epochs"""

        if self.cyl_radii is None or len(self.cyl_radii) != 1:
            raise Py4DGeoError(
                f"{self.name} requires exactly one cylinder radius to be given"
            )

        # Ensure that the KDTree data structures have been built. This is no-op
        # if it has already been triggered before - e.g. by a user with a custom
        # leaf cutoff parameter.
        epoch1.build_kdtree()
        epoch2.build_kdtree()

        distances, uncertainties = _py4dgeo.compute_distances(
            self.corepoints,
            self.cyl_radii[0],
            epoch1,
            epoch2,
            self.directions(),
            self.max_distance,
            self.registration_error,
            self.callback_workingset_finder(),
            self.callback_distance_calculation(),
        )

        return distances, uncertainties

    def run(self):
        """Main entry point for running the algorithm"""
        return self.calculate_distances(self.epochs[0], self.epochs[1])

    def callback_workingset_finder(self):
        """The callback used to determine the point cloud subset around a corepoint"""
        return _py4dgeo.cylinder_workingset_finder

    def callback_distance_calculation(self):
        """The callback used to calculate the distance between two point clouds"""
        if self.robust_aggr:
            return _py4dgeo.median_iqr_distance
        else:
            return _py4dgeo.mean_stddev_distance


class M3C2(M3C2LikeAlgorithm):
    def __init__(
        self,
        normal_radii: typing.List[float] = None,
        orientation_vector: np.ndarray = np.array([0, 0, 1]),
        corepoint_normals: np.ndarray = None,
        cloud_for_normals: Epoch = None,
        **kwargs,
    ):
        self.normal_radii = normal_radii
        self.orientation_vector = as_double_precision(
            make_contiguous(orientation_vector), policy_check=False
        )
        self.cloud_for_normals = cloud_for_normals
        self.corepoint_normals = corepoint_normals
        super().__init__(**kwargs)

    def directions(self):
        # If we already have normals, we return them. This happens e.g. if the user
        # explicitly provided them or if we already computed them in a previous run.
        if self.corepoint_normals is not None:
            # Make sure that the normals use double precision
            self.corepoint_normals = as_double_precision(self.corepoint_normals)

            # Assert that the normal array has the correct shape
            if (
                len(self.corepoint_normals.shape) != 2
                or self.corepoint_normals.shape[0] not in (1, self.corepoints.shape[0])
                or self.corepoint_normals.shape[1] != 3
            ):
                raise Py4DGeoError(
                    f"Incompative size of corepoint normal array {self.corepoint_normals.shape}, expected {self.corepoints.shape} or (1, 3)!"
                )

            return self.corepoint_normals

        # This does not work in STRICT mode
        if not memory_policy_is_minimum(MemoryPolicy.MINIMAL):
            raise Py4DGeoError(
                "M3C2 requires at least the MINIMUM memory policy level to compute multiscale normals"
            )

        # Allocate the storage for the computed normals
        self.corepoint_normals = np.empty(self.corepoints.shape, dtype=np.float64)

        # Find the correct epoch to use for normal calculation
        normals_epoch = self.cloud_for_normals
        if normals_epoch is None:
            normals_epoch = self.epochs[0]
        normals_epoch = as_epoch(normals_epoch)

        # Trigger the precomputation
        _py4dgeo.compute_multiscale_directions(
            normals_epoch,
            self.corepoints,
            self.normal_radii,
            self.orientation_vector,
            self.corepoint_normals,
        )

        return self.corepoint_normals

    @property
    def name(self):
        return "M3C2"
