from py4dgeo.epoch import Epoch
from py4dgeo.util import (
    Py4DGeoError,
    as_double_precision,
    make_contiguous,
)

import numpy as np
import typing
import laspy

from scipy.spatial import cKDTree


class C2C:
    """Cloud-to-cloud distance based on nearest neighbors.

    Parameters
    ----------
    epochs
        Pair of point clouds or Epoch objects.
    corepoints
        Optional source points. If given, distances are computed for these points
        instead of points from ``epochs[0]``.
    max_distance
        Maximum nearest-neighbor search distance. Distances beyond this threshold
        are returned as ``np.nan``.
    correspondence_filter
        Optional correspondence filter.
        ``"none"`` keeps all valid nearest-neighbor matches (default).
        ``"mutual_nearest_neighbors"`` accepts a match only if nearest-neighbor
        relation is mutual.
    """

    _VALID_CORRESPONDENCE_FILTERS = ("none", "mutual_nearest_neighbors")

    def __init__(
        self,
        epochs: typing.Optional[typing.Tuple[Epoch, Epoch]] = None,
        corepoints: typing.Optional[np.ndarray] = None,
        max_distance: float = np.inf,
        correspondence_filter: str = "none",
    ):
        self.epochs = epochs
        self.corepoints = corepoints
        self.max_distance = max_distance
        self.correspondence_filter = correspondence_filter

    @property
    def corepoints(self):
        return self._corepoints

    @corepoints.setter
    def corepoints(self, _corepoints):
        if _corepoints is None:
            self._corepoints = None
            return

        if len(_corepoints.shape) != 2 or _corepoints.shape[1] != 3:
            raise Py4DGeoError("Corepoints need to be given as an array of shape nx3")

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
        return "C2C"

    @property
    def correspondence_filter(self):
        return self._correspondence_filter

    @correspondence_filter.setter
    def correspondence_filter(self, filter_name):
        if filter_name not in self._VALID_CORRESPONDENCE_FILTERS:
            raise Py4DGeoError(
                "Invalid correspondence_filter. "
                "Use one of: 'none', 'mutual_nearest_neighbors'."
            )
        self._correspondence_filter = filter_name

    def calculate_distances(
        self,
        epoch1: typing.Union[np.ndarray, Epoch],
        epoch2: typing.Union[np.ndarray, Epoch],
    ):
        source = self.corepoints if self.corepoints is not None else epoch1

        pts_1 = source.cloud if isinstance(source, Epoch) else np.asarray(source)
        pts_2 = epoch2.cloud if isinstance(epoch2, Epoch) else np.asarray(epoch2)

        if pts_1.shape[0] == 0:
            return np.empty(0, dtype=float)
        if pts_2.shape[0] == 0:
            return np.full(pts_1.shape[0], np.nan, dtype=float)

        tree = cKDTree(pts_2)
        distances, forward_indices = tree.query(
            pts_1,
            k=1,
            distance_upper_bound=self.max_distance,
            workers=-1,
        )

        # cKDTree returns inf when no neighbor is found within max_distance
        distances = distances.astype(float, copy=False)
        valid_forward = np.isfinite(distances)

        if self.correspondence_filter == "mutual_nearest_neighbors":
            reverse_tree = cKDTree(pts_1)
            _, reverse_indices = reverse_tree.query(pts_2, k=1, workers=-1)

            source_indices = np.arange(pts_1.shape[0], dtype=np.int64)
            mutual_mask = np.zeros(pts_1.shape[0], dtype=bool)
            mutual_mask[valid_forward] = (
                reverse_indices[forward_indices[valid_forward]]
                == source_indices[valid_forward]
            )

            valid_forward &= mutual_mask

        distances[~valid_forward] = np.nan
        return distances

    def run(self):
        """Main entry point for running the algorithm."""
        if self.epochs is None:
            raise Py4DGeoError("Exactly two epochs need to be given!")
        return self.calculate_distances(self.epochs[0], self.epochs[1])


def write_c2c_results_to_las(outfilepath: str, c2c: C2C, attribute_dict: dict = {}):
    """Save the C2C output points and attributes to a LAS file.

    :param outfilepath:
        The las file path to save the corepoints and other attributes.
    :type outfilepath: str
    :param c2c:
        The C2C object.
    :type c2c: C2C
    :param attribute_dict:
        The dictionary of attributes which will be saved together with points.
    :type attribute_dict: dict
    """
    if c2c.corepoints is not None:
        outpoints = c2c.corepoints
    elif c2c.epochs is not None:
        epoch1 = c2c.epochs[0]
        outpoints = epoch1.cloud if isinstance(epoch1, Epoch) else np.asarray(epoch1)
    else:
        raise Py4DGeoError(
            "Cannot determine output points. Please provide corepoints or epochs."
        )

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
