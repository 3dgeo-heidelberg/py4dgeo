from py4dgeo.epoch import Epoch
from py4dgeo.base import PairwiseEpochCorepointsBase
from py4dgeo.util import (
    Py4DGeoError,
)

import numpy as np
import typing
import laspy

from scipy.spatial import cKDTree


class C2C(PairwiseEpochCorepointsBase):
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
        super().__init__(epochs=epochs, corepoints=corepoints)
        self.max_distance = max_distance
        self.correspondence_filter = correspondence_filter

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

    def _nearest_neighbor_query(
        self,
        reference: typing.Union[np.ndarray, Epoch],
        query: np.ndarray,
    ):
        if isinstance(reference, Epoch):
            reference._validate_search_tree()

            # Taken from registration 89-93 and 247-249
            neighbor_arrays = np.asarray(reference.kdtree.nearest_neighbors(query, 1))
            indices, distances = np.split(neighbor_arrays, 2, axis=0)

            nearest_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
            # Distances returned by py4dgeo KDTree are squared, see test_kdtree
            distances = np.sqrt(np.asarray(distances, dtype=float).reshape(-1))

            return distances, nearest_indices

        tree = cKDTree(np.asarray(reference))
        distances, nearest_indices = tree.query(query, k=1, workers=-1)
        distances = np.asarray(distances, dtype=float).reshape(-1)
        nearest_indices = np.asarray(nearest_indices, dtype=np.int64).reshape(-1)

        return distances, nearest_indices

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

        distances, forward_indices = self._nearest_neighbor_query(epoch2, pts_1)
        distances = distances.astype(float, copy=False)
        valid_forward = distances <= self.max_distance

        if self.correspondence_filter == "mutual_nearest_neighbors":
            _, reverse_indices = self._nearest_neighbor_query(source, pts_2)

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
