from py4dgeo.epoch import Epoch, as_epoch
from py4dgeo.util import Py4DGeoError, append_file_extension

import datetime
import json
import numpy as np
import os
import tempfile
import zipfile

import py4dgeo._py4dgeo as _py4dgeo


# This integer controls the versioning of the segmentation file format. Whenever the
# format is changed, this version should be increased, so that py4dgeo can warn
# about incompatibilities of py4dgeo with loaded data. This version is intentionally
# different from py4dgeo's version, because not all releases of py4dgeo necessarily
# change the segmentation file format and we want to be as compatible as possible.
PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION = 0


class SpatiotemporalSegmentation:
    def __init__(self, reference_epoch=None, m3c2=None):
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
        self.timedeltas = []
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

        # Ensure that we have a file extension
        filename = append_file_extension(filename, "zip")

        # Use a temporary directory when creating files
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create the final archive
            with zipfile.ZipFile(
                filename, mode="w", compression=zipfile.ZIP_BZIP2
            ) as zf:
                # Write the epoch file format version number
                zf.writestr(
                    "SEGMENTATION_FILE_FORMAT",
                    str(PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION),
                )

                # Write the metadata dictionary into a json file
                timestampsfile = os.path.join(tmp_dir, "timestamps.json")
                with open(timestampsfile, "w") as f:
                    json.dump(
                        [
                            {
                                "days": td.days,
                                "seconds": td.seconds,
                                "microseconds": td.microseconds,
                            }
                            for td in self.timedeltas
                        ],
                        f,
                    )
                zf.write(timestampsfile, arcname="timestamps.json")

                # Write distances and uncertainties
                distance_file = os.path.join(tmp_dir, "distances.npy")
                np.save(distance_file, self.distances)
                zf.write(distance_file, arcname="distances.npy")

                uncertainty_file = os.path.join(tmp_dir, "uncertainty.npy")
                np.save(uncertainty_file, self.uncertainties)
                zf.write(uncertainty_file, arcname="uncertainty.npy")

                # Write reference epoch
                refepoch_file = os.path.join(tmp_dir, "reference_epoch.zip")
                self._reference_epoch.save(refepoch_file)
                zf.write(refepoch_file, arcname="reference_epoch.zip")

    @classmethod
    def load(cls, filename, m3c2):
        """Load a segmentation object from a file"""

        # Ensure that we have a file extension
        filename = append_file_extension(filename, "zip")

        # Use temporary directory for extraction of files
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Open the ZIP archive
            with zipfile.ZipFile(filename, mode="r") as zf:

                # Read the segmentation file version number and compare to current
                version = int(zf.read("SEGMENTATION_FILE_FORMAT").decode())
                if version != PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION:
                    raise Py4DGeoError("Segmentation file format is out of date!")

                # Read the reference epoch
                ref_epochfile = zf.extract("reference_epoch.zip", path=tmp_dir)
                refepoch = Epoch.load(ref_epochfile)

                # Create the segmentation object
                segmentation = cls(reference_epoch=refepoch, m3c2=m3c2)

                # Read the distances and uncertainties
                distancefile = zf.extract("distances.npy", path=tmp_dir)
                segmentation.distances = np.load(distancefile)
                uncertaintyfile = zf.extract("uncertainty.npy", path=tmp_dir)
                segmentation.uncertainties = np.load(uncertaintyfile)

                # Read timedeltas
                timestampsfile = zf.extract("timestamps.json")
                with open(timestampsfile) as f:
                    timedeltas = json.load(f)
                segmentation.timedeltas = [
                    datetime.timedelta(**data) for data in timedeltas
                ]

        return segmentation

    def add_epoch(self, epoch):
        """Adds an epoch to the existing segmentation"""

        # Calculate the M3C2 distances
        d, u = self._m3c2.calculate_distances(
            self._reference_epoch, check_epoch_timestamp(epoch)
        )

        # Append them to our existing infrastructure
        self.distances = np.vstack((self.distances, np.expand_dims(d, axis=0)))
        self.uncertainties = np.vstack((self.uncertainties, np.expand_dims(u, axis=0)))
        self.timedeltas.append(epoch.timestamp - self._reference_epoch.timestamp)


class RegionGrowingAlgorithm:
    def __init__(self):
        """Construct a spatiotemporal segmentation algorithm.

        This class can be derived from to customize the algorithm behaviour.
        """
        pass

    def temporal_averaging(self, distances):
        """Smoothen a space-time array of distance change"""

        # We use no-op smooting as the default implementation here
        return distances

    def distance_measure(self):
        """Distance measure between two time series

        Expected to return a function that accepts two time series and returns
        the distance.
        """

        return _py4dgeo.dtw_distance

    def construct_sorted_seedpoints(self):
        """Calculate seedpoints for the region growing algorithm

        They are expected to be sorted by priority.
        """

        return [_py4dgeo.RegionGrowingSeed(0, 0, 1)]

    def run(self, segmentation):
        """Calculate the segmentation"""

        # Smooth the distance array
        smoothed = self.temporal_averaging(segmentation.distances)

        # Get corepoints from M3C2 class and build a KDTree on them
        corepoints = as_epoch(segmentation.m3c2.corepoints)
        corepoints.build_kdtree()

        # Calculate the list of seed points
        seeds = self.construct_sorted_seedpoints()
        objects = []

        # Iterate over the seeds to maybe turn them into objects
        for seed in seeds:
            # Check all already calculated objects whether they overlap with this seed.
            found = False
            (seed_index,) = seed.indices
            for obj in objects:
                if seed_index in obj.indices and (
                    obj.end_epoch > seed.start_epoch or seed.end_epoch > obj.start_epoch
                ):
                    found = True
                    break

            # If we found an overlap, we skip this seed
            if found:
                break

            data = _py4dgeo.RegionGrowingAlgorithmData(
                smoothed, corepoints, 2.0, seed, [0.5]
            )

            # Perform the region growing
            objects.append(_py4dgeo.region_growing(data))

        return objects


def check_epoch_timestamp(epoch):
    """Validate an epoch to be used with SpatiotemporalSegmentation"""
    if epoch.timestamp is None:
        raise Py4DGeoError(
            "Epochs need to define a timestamp to be usable in SpatiotemporalSegmentation"
        )

    return epoch
