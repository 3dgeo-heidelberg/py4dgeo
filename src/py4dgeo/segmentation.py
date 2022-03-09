from py4dgeo.util import Py4DGeoError

import numpy as np


# This integer controls the versioning of the segmentation file format. Whenever the
# format is changed, this version should be increased, so that py4dgeo can warn
# about incompatibilities of py4dgeo with loaded data. This version is intentionally
# different from py4dgeo's version, because not all releases of py4dgeo necessarily
# change the segmentation file format and we want to be as compatible as possible.
PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION = 0


class SpatiotemporalSegmentation:
    def __init__(self, reference_epoch=None, epochs=None, m3c2=None):
        """Construct a spatiotemporal segmentation object

        This is the basic data structure for the 4D objects by change algorithm
        and its derived variants. It allows to store M3C2 distances for a time
        series of epochs. The original point clouds themselved are not needed after
        initial distance calculation and additional epochs can be added to
        existing segmentations. The class allows saving and loading to a custom
        file format.

        :param reference_epoch:
            The reference epoch that is used to calculate distances against. This
            is a required parameter
        :type reference_epoch: Epoch
        :param m3c2:
            The M3C2 algorithm instance to calculate distances. This is a required
            paramter.
        :type m3c2: M3C2LikeAlgorithm
        """

        # Store parameters as internals
        self._reference_epoch = check_epoch_timestamp(reference_epoch)
        self._m3c2 = m3c2

        # This is the data structure that holds the distances
        self.distances = np.empty((0, self._m3c2.corepoints.shape[0]), dtype=np.float64)
        self.uncertainties = np.empty(
            (0, self._m3c2.corepoints.shape[0]),
            dtype=np.dtype(
                [
                    ("lodetection", "<f8"),
                    ("spread1", "<f8"),
                    ("num_samples1", "<i8"),
                    ("spread2", "<f8"),
                    ("num_samples2", "<i8"),
                ]
            ),
        )

    def save(self, filename):
        """Save segmentation to a file"""

        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        """Load a segmentation object from a file"""

        raise NotImplementedError

    def add_epoch(self, epoch):
        """Adds an epoch to the existing segmentation"""

        # Calculate the M3C2 distances
        d, u = self._m3c2.calculate_distances(
            self._reference_epoch, check_epoch_timestamp(epoch)
        )

        # Append them to our existing infrastructure
        self.distances = np.vstack((self.distances, np.expand_dims(d, axis=0)))
        self.uncertainties = np.vstack((self.uncertainties, np.expand_dims(u, axis=0)))


def check_epoch_timestamp(epoch):
    """Validate an epoch to be used with SpatiotemporalSegmnetation"""
    if epoch.timestamp is None:
        raise Py4DGeoError(
            "Epochs need to define a timestamp to be usable in SpatiotemporalSegmentation"
        )

    return epoch
