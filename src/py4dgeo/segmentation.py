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


class SpatiotemporalAnalysis:
    def __init__(self, filename):
        """Construct a spatiotemporal segmentation object

        This is the basic data structure for the 4D objects by change algorithm
        and its derived variants. It manages storage of M3C2 distances and other
        intermediate results for a time series of epochs. The original point clouds
        themselves are not needed after initial distance calculation and additional
        epochs can be added to an existing analysis. The class uses a disk backend
        to store information and allows lazy loading of additional data like e.g.
        M3C2 uncertainty values for postprocessing.

        :param filename:
            The filename used for this analysis. If it does not exist on the file
            system, a new analysis is created. Otherwise, the data is loaded.
        :type filename: str
        """

        # Store the filename
        self.filename = append_file_extension(filename, "zip")

        # Instantiate some properties used later on
        self._m3c2 = None

        # If the filename does not already exist, we create a new archive
        if not os.path.exists(self.filename):
            with zipfile.ZipFile(
                self.filename, mode="w", compression=zipfile.ZIP_BZIP2
            ) as zf:
                # Write the epoch file format version number
                zf.writestr(
                    "SEGMENTATION_FILE_FORMAT",
                    str(PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION),
                )

        # Assert that the segmentation file format is still valid
        with zipfile.ZipFile(
            self.filename, mode="r", compression=zipfile.ZIP_BZIP2
        ) as zf:
            # Read the segmentation file version number and compare to current
            version = int(zf.read("SEGMENTATION_FILE_FORMAT").decode())
            if version != PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION:
                raise Py4DGeoError("Segmentation file format is out of date!")

    @property
    def reference_epoch(self):
        """Access the reference epoch of this analysis"""
        with zipfile.ZipFile(
            self.filename, mode="a", compression=zipfile.ZIP_BZIP2
        ) as zf:
            # Double check that the reference has already been set
            if "reference_epoch.zip" not in zf.namelist():
                raise Py4DGeoError("Reference epoch for analysis not yet set")

            # Extract it from the archive
            with tempfile.TemporaryDirectory() as tmp_dir:
                ref_epochfile = zf.extract("reference_epoch.zip", path=tmp_dir)
                return Epoch.load(ref_epochfile)

    @reference_epoch.setter
    def reference_epoch(self, epoch):
        """Set the reference epoch of this analysis (only possible once)"""
        with zipfile.ZipFile(
            self.filename, mode="a", compression=zipfile.ZIP_BZIP2
        ) as zf:
            # If we already have a reference epoch, the user should start a
            # new analysis instead
            if "reference_epoch.zip" in zf.namelist():
                raise Py4DGeoError(
                    "Reference epoch cannot be changed - please start a new analysis"
                )

            # Ensure that we do have a timestamp on the epoch
            epoch = check_epoch_timestamp(epoch)

            # Ensure that the KDTree is built - no-op if triggered by the user
            epoch.build_kdtree()

            # Write the reference epoch into the archive
            with tempfile.TemporaryDirectory() as tmp_dir:
                epochfilename = os.path.join(tmp_dir, "reference_epoch.zip")
                epoch.save(epochfilename)
                zf.write(epochfilename, arcname="reference_epoch.zip")

    @property
    def m3c2(self):
        """Access the M3C2 algorithm of this analysis"""
        # If M3C2 has not been set,
        if self._m3c2 is None:
            raise Py4DGeoError("M3C2 algorithm has not been set")
        return self._m3c2

    @m3c2.setter
    def m3c2(self, _m3c2):
        """Set the M3C2 algorithm of this analysis"""
        self._m3c2 = _m3c2

    @property
    def timedeltas(self):
        """Access the sequence of time stamp deltas for the time series"""
        with zipfile.ZipFile(
            self.filename, mode="r", compression=zipfile.ZIP_BZIP2
        ) as zf:
            if "timestamps.json" not in zf.namelist():
                return []

            # Read timedeltas
            with tempfile.TemporaryDirectory() as tmp_dir:
                timestampsfile = zf.extract("timestamps.json", path=tmp_dir)
                with open(timestampsfile) as f:
                    timedeltas = json.load(f)

                # Convert the serialized deltas to datetime.timedelta
                return [datetime.timedelta(**data) for data in timedeltas]

    @property
    def distances(self):
        """Access the M3C2 distances of this analysis"""
        with zipfile.ZipFile(
            self.filename, mode="r", compression=zipfile.ZIP_BZIP2
        ) as zf:
            if "distances.npy" not in zf.namelist():
                return np.empty((0, self.m3c2.corepoints.shape[0]), dtype=np.float64)

            with tempfile.TemporaryDirectory() as tmp_dir:
                distancefile = zf.extract("distances.npy", path=tmp_dir)
                return np.load(distancefile)

    @property
    def uncertainties(self):
        """Access the M3C2 uncertainties of this analysis"""
        with zipfile.ZipFile(
            self.filename, mode="r", compression=zipfile.ZIP_BZIP2
        ) as zf:
            if "uncertainties.npy" not in zf.namelist():
                return np.empty(
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

            with tempfile.TemporaryDirectory() as tmp_dir:
                uncertaintyfile = zf.extract("uncertainties.npy", path=tmp_dir)
                return np.load(uncertaintyfile)

    def add_epochs(self, *epochs):
        """Add a numbers of epochs to the existing analysis"""

        # Assert that all epochs have a timestamp
        for epoch in epochs:
            check_epoch_timestamp(epoch)

        # Lazily fetch required data
        reference_epoch = self.reference_epoch
        timedeltas = self.timedeltas

        # Collect the calculated results to only add them once to the archive
        new_distances = []
        new_uncertainties = []

        # Iterate over the given epochs
        for epoch in sorted(epochs, key=lambda e: e.timestamp):
            # Calculate the M3C2 distances
            d, u = self.m3c2.calculate_distances(reference_epoch, epoch)
            new_distances.append(d)
            new_uncertainties.append(u)
            timedeltas.append(epoch.timestamp - reference_epoch.timestamp)

        # Prepare all archive data in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write a new timestamps file
            timestampsfile = os.path.join(tmp_dir, "timestamps.json")
            with open(timestampsfile, "w") as f:
                json.dump(
                    [
                        {
                            "days": td.days,
                            "seconds": td.seconds,
                            "microseconds": td.microseconds,
                        }
                        for td in timedeltas
                    ],
                    f,
                )

            # We do not need the reference_epoch at this point
            del reference_epoch

            # Load the distance array and append new data
            distance_file = os.path.join(tmp_dir, "distances.npy")
            np.save(
                distance_file,
                np.vstack(
                    (self.distances,)
                    + tuple(np.expand_dims(d, axis=0) for d in new_distances)
                ),
            )

            # Load the uncertainty array and append new data
            uncertainty_file = os.path.join(tmp_dir, "uncertainties.npy")
            np.save(
                uncertainty_file,
                np.vstack(
                    (self.uncertainties,)
                    + tuple(np.expand_dims(u, axis=0) for u in new_uncertainties)
                ),
            )

            # Dump the updated files into the archive
            with zipfile.ZipFile(
                self.filename, mode="a", compression=zipfile.ZIP_BZIP2
            ) as zf:
                zf.write(timestampsfile, arcname="timestamps.json")
                zf.write(distance_file, arcname="distances.npy")
                zf.write(uncertainty_file, arcname="uncertainties.npy")


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
