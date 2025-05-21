from ast import List
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
import laspy

import _py4dgeo


logger = logging.getLogger("py4dgeo")


class M3C2LikeAlgorithm(abc.ABC):
    def __init__(
        self,
        epochs: typing.Optional[typing.Tuple[Epoch, ...]] = None,
        corepoints: typing.Optional[np.ndarray] = None,
        cyl_radii: typing.Optional[List] = None,
        cyl_radius: typing.Optional[float] = None,
        max_distance: float = 0.0,
        registration_error: float = 0.0,
        robust_aggr: bool = False,
    ):
        self.epochs = epochs
        self.corepoints = corepoints
        self.cyl_radii = cyl_radii
        self.cyl_radius = cyl_radius
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

    def calculate_distances(
        self, epoch1, epoch2, searchtree: typing.Optional[str] = None
    ):
        """Calculate the distances between two epochs"""

        if isinstance(self.cyl_radii, typing.Iterable):
            logger.warning(
                "DEPRECATION: use cyl_radius instead of cyl_radii. In a future version, cyl_radii will be removed!"
            )
            if len(self.cyl_radii) != 1:
                raise Py4DGeoError(
                    "cyl_radii must be a list containing a single float!"
                )
            elif self.cyl_radius is None:
                self.cyl_radius = self.cyl_radii[0]
            self.cyl_radii = None

        if self.cyl_radius is None:
            raise Py4DGeoError(
                f"{self.name} requires exactly one cylinder radius to be given as a float."
            )

        # Ensure appropriate trees are built
        epoch1._validate_search_tree()
        epoch2._validate_search_tree()

        distances, uncertainties = _py4dgeo.compute_distances(
            self.corepoints,
            self.cyl_radius,
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
        self._directions_radii = None
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

        # Find the correct epoch to use for normal calculation
        normals_epoch = self.cloud_for_normals
        if normals_epoch is None:
            normals_epoch = self.epochs[0]
        normals_epoch = as_epoch(normals_epoch)

        # Ensure appropriate tree structures have been built
        normals_epoch._validate_search_tree()

        # Trigger the precomputation
        self.corepoint_normals, self._directions_radii = (
            _py4dgeo.compute_multiscale_directions(
                normals_epoch,
                self.corepoints,
                self.normal_radii,
                self.orientation_vector,
            )
        )

        return self.corepoint_normals

    def directions_radii(self):
        if self._directions_radii is None:
            raise ValueError(
                "Radii are only available after calculating directions with py4dgeo."
            )

        return self._directions_radii

    @property
    def name(self):
        return "M3C2"


def write_m3c2_results_to_las(
    outfilepath: str, m3c2: M3C2LikeAlgorithm, attribute_dict: dict = {}
):
    """Save the corepoints, distances and other attributes to a given las filename

    :param outfilepath:
        The las file path to save the corepoints, distances and other attributes.
    :type outfilepath: str
    :param m3c2:
        The M3C2LikeAlgorithm object.
    :type m3c2: M3C2LikeAlgorithm
    :param attribute_dict:
        The dictionary of attributes which will be saved together with corepoints.
    :type attribute_dict: dict
    """
    # Will utilize Epoch.save(), by creating epoch from m3c2 corepoints and attribute_dict
    # to be the epoch additional_dimensions, write this epoch to las not a zip file.
    outpoints = m3c2.corepoints
    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.x_scale = 0.00025
    hdr.y_scale = 0.00025
    hdr.z_scale = 0.00025
    mean_extent = np.mean(outpoints, axis=0)
    hdr.x_offset = int(mean_extent[0])
    hdr.y_offset = int(mean_extent[1])
    hdr.z_offset = int(mean_extent[2])

    las = laspy.LasData(hdr)

    las.x = outpoints[:, 0]
    las.y = outpoints[:, 1]
    las.z = outpoints[:, 2]
    for key, vals in attribute_dict.items():
        try:
            las[key] = vals
        except:
            las.add_extra_dim(laspy.ExtraBytesParams(name=key, type=type(vals[0])))
            las[key] = vals

    las.write(outfilepath)
