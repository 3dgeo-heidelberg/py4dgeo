from py4dgeo.epoch import Epoch, as_epoch
from py4dgeo.util import Py4DGeoError, append_file_extension
from py4dgeo.zipfile import UpdateableZipFile

import datetime
import json
import numpy as np
import os
import ruptures
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
    def __init__(self, filename, compress=True):
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
        :param compress:
            Whether to compress the stored data. This is a tradeoff decision between
            disk space and runtime. Especially appending new epochs to an existing
            analysis is an operation whose runtime can easily be dominated by
            decompression/compression of data.
        :type compress: bool
        """

        # Store the given parameters
        self.filename = append_file_extension(filename, "zip")
        self.compress = compress

        # Instantiate some properties used later on
        self._m3c2 = None
        self._corepoints = None

        # If the filename does not already exist, we create a new archive
        if not os.path.exists(self.filename):
            with zipfile.ZipFile(self.filename, mode="w") as zf:
                # Write the segmentation file format version number
                zf.writestr(
                    "SEGMENTATION_FILE_FORMAT",
                    str(PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION),
                )

                # Write the compression algorithm used for all suboperations
                zf.writestr("USE_COMPRESSION", str(self.compress))

        # Assert that the segmentation file format is still valid
        with zipfile.ZipFile(self.filename, mode="r") as zf:
            # Read the segmentation file version number and compare to current
            version = int(zf.read("SEGMENTATION_FILE_FORMAT").decode())
            if version != PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION:
                raise Py4DGeoError("Segmentation file format is out of date!")

            # Read the compression algorithm
            self.compress = eval(zf.read("USE_COMPRESSION").decode())

    @property
    def reference_epoch(self):
        """Access the reference epoch of this analysis"""
        with zipfile.ZipFile(self.filename, mode="r") as zf:
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
        with zipfile.ZipFile(self.filename, mode="a") as zf:
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
    def corepoints(self):
        """Access the corepoints of this analysis"""
        if self._corepoints is None:
            with zipfile.ZipFile(self.filename, mode="r") as zf:
                # Double check that the reference has already been set
                if "corepoints.zip" not in zf.namelist():
                    raise Py4DGeoError("Corepoints for analysis not yet set")

                # Extract it from the archive
                with tempfile.TemporaryDirectory() as tmp_dir:
                    cpfile = zf.extract("corepoints.zip", path=tmp_dir)
                    return Epoch.load(cpfile)

        return self._corepoints

    @corepoints.setter
    def corepoints(self, _corepoints):
        """Set the corepoints for this analysis (only possible once)"""
        with zipfile.ZipFile(self.filename, mode="a") as zf:
            # If we already have corepoints in the archive, the user should start a
            # new analysis instead
            if "corepoints.zip" in zf.namelist():
                raise Py4DGeoError(
                    "Corepoints cannot be changed - please start a new analysis"
                )

            # Ensure that the corepoints are stored as an epoch and build its KDTree
            self._corepoints = as_epoch(_corepoints)
            self._corepoints.build_kdtree()

            # Write the corepoints into the archive
            with tempfile.TemporaryDirectory() as tmp_dir:
                cpfilename = os.path.join(tmp_dir, "corepoints.zip")
                self._corepoints.save(cpfilename)
                zf.write(cpfilename, arcname="corepoints.zip")

    @property
    def m3c2(self):
        """Access the M3C2 algorithm of this analysis"""
        # If M3C2 has not been set, we use a default constructed one
        return self._m3c2

    @m3c2.setter
    def m3c2(self, _m3c2):
        """Set the M3C2 algorithm of this analysis"""
        self._m3c2 = _m3c2

    @property
    def timedeltas(self):
        """Access the sequence of time stamp deltas for the time series"""
        with zipfile.ZipFile(self.filename, mode="r") as zf:
            if "timestamps.json" not in zf.namelist():
                return []

            # Read timedeltas
            with tempfile.TemporaryDirectory() as tmp_dir:
                timestampsfile = zf.extract("timestamps.json", path=tmp_dir)
                with open(timestampsfile) as f:
                    timedeltas = json.load(f)

                # Convert the serialized deltas to datetime.timedelta
                return [datetime.timedelta(**data) for data in timedeltas]

    @timedeltas.setter
    def timedeltas(self, _timedeltas):
        """Set the timedeltas manually

        This is only possible exactly once and mutually exclusive with adding
        epochs via the :ref:`add_epochs` method.
        """
        with zipfile.ZipFile(self.filename, mode="a") as zf:
            # If we already have timestamps in the archive, this is not possible
            if "timestamps.json" in zf.namelist():
                raise Py4DGeoError(
                    "Timestamps can only be set on freshly created analysis instances"
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                timestampsfile = os.path.join(tmp_dir, "timestamps.json")
                with open(timestampsfile, "w") as f:
                    json.dump(
                        [
                            {
                                "days": td.days,
                                "seconds": td.seconds,
                                "microseconds": td.microseconds,
                            }
                            for td in _timedeltas
                        ],
                        f,
                    )
                zf.write(timestampsfile, arcname="timestamps.json")

    @property
    def distances(self):
        """Access the M3C2 distances of this analysis"""
        with zipfile.ZipFile(self.filename, mode="r") as zf:
            filename = "distances.npz" if self.compress else "distances.npy"
            if filename not in zf.namelist():
                return np.empty((self.corepoints.cloud.shape[0], 0), dtype=np.float64)

            with tempfile.TemporaryDirectory() as tmp_dir:
                distancefile = zf.extract(filename, path=tmp_dir)
                read_func = (
                    (lambda f: np.load(f)["arr_0"]) if self.compress else np.load
                )
                return read_func(distancefile)

    @distances.setter
    def distances(self, _distances):
        """Set the distances manually

        This is only possible exactly once and mutually exclusive with adding
        epochs via the :ref:`add_epochs` method.
        """
        with zipfile.ZipFile(self.filename, mode="a") as zf:
            filename = "distances.npz" if self.compress else "distances.npy"
            write_func = np.savez_compressed if self.compress else np.save

            # If we already have distacces in the archive, this is not possible
            if filename in zf.namelist():
                raise Py4DGeoError(
                    "Distances can only be set on freshly created analysis instances, use add_epochs instead."
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                distancesfile = os.path.join(tmp_dir, filename)
                write_func(distancesfile, _distances)
                zf.write(distancesfile, arcname=filename)

    @property
    def uncertainties(self):
        """Access the M3C2 uncertainties of this analysis"""
        with zipfile.ZipFile(self.filename, mode="r") as zf:
            filename = "uncertainties.npz" if self.compress else "uncertainties.npy"
            if filename not in zf.namelist():
                return np.empty(
                    (self.corepoints.cloud.shape[0], 0),
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
                uncertaintyfile = zf.extract(filename, path=tmp_dir)
                read_func = (
                    (lambda f: np.load(f)["arr_0"]) if self.compress else np.load
                )
                return read_func(uncertaintyfile)

    @uncertainties.setter
    def uncertainties(self, _uncertainties):
        """Set the uncertainties manually

        This is only possible exactly once and mutually exclusive with adding
        epochs via the :ref:`add_epochs` method.
        """
        with zipfile.ZipFile(self.filename, mode="a") as zf:
            filename = "uncertainties.npz" if self.compress else "uncertainties.npy"
            write_func = np.savez_compressed if self.compress else np.save

            # If we already have distacces in the archive, this is not possible
            if filename in zf.namelist():
                raise Py4DGeoError(
                    "Uncertainties can only be set on freshly created analysis instances, use add_epochs instead."
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                uncertaintiesfile = os.path.join(tmp_dir, filename)
                write_func(uncertaintiesfile, _uncertainties)
                zf.write(uncertaintiesfile, arcname=filename)

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
            # Prepare the M3C2 instance
            self.m3c2.corepoints = self.corepoints.cloud
            self.m3c2.epochs = (reference_epoch, epoch)

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

            # Depending on whether we compress, we use different numpy functionality
            write_func = np.savez_compressed if self.compress else np.save
            distance_filename = "distances.npz" if self.compress else "distances.npy"
            uncertainty_filename = (
                "uncertainties.npz" if self.compress else "uncertainties.npy"
            )

            # Load the distance array and append new data
            distance_file = os.path.join(tmp_dir, distance_filename)
            write_func(
                distance_file,
                np.concatenate(
                    (self.distances, np.column_stack(tuple(new_distances))), axis=1
                ),
            )

            # Load the uncertainty array and append new data
            uncertainty_file = os.path.join(tmp_dir, uncertainty_filename)
            write_func(
                uncertainty_file,
                np.concatenate(
                    (self.uncertainties, np.column_stack(tuple(new_uncertainties))),
                    axis=1,
                ),
            )

            # Dump the updated files into the archive
            with UpdateableZipFile(self.filename, mode="a") as zf:
                if "timestamps.json" in zf.namelist():
                    zf.remove("timestamps.json")
                zf.write(timestampsfile, arcname="timestamps.json")
                if distance_filename in zf.namelist():
                    zf.remove(distance_filename)
                zf.write(distance_file, arcname=distance_filename)
                if uncertainty_filename in zf.namelist():
                    zf.remove(uncertainty_filename)
                zf.write(uncertainty_file, arcname=uncertainty_filename)


class RegionGrowingAlgorithm:
    def __init__(
        self,
        smoothing_window=24,
        neighborhood_radius=1.0,
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
    ):
        """Construct a spatiotemporal segmentation algorithm.

        This class can be derived from to customize the algorithm behaviour.

        :param neighborhood_radius:
            The size of the neighborhood of a corepoint. All corepoints within
            this radius are considered adjacent and are therefore considered as
            candidates for inclusion in the region growing algorithm.
        :type neighborhood_radius: float
        :param thresholds:
            A list of thresholds to use as candidates in 4D-OBC's adaptive
            thresholding procedure.
        :type thresholds: list
        """

        # Store the given parameters
        self.smoothing_window = smoothing_window
        self.neighborhood_radius = neighborhood_radius
        self.thresholds = thresholds

    def temporal_averaging(self, distances):
        """Smoothen a space-time array of distance change"""

        smoothed = np.empty_like(distances)
        eps = self.smoothing_window // 2

        for i in range(distances.shape[1]):
            smoothed[i, :] = np.nanmedian(
                distances[max(0, i - eps) : min(distances.shape[1] - 1, i + eps)],
                axis=0,
            )

        # We use no-op smooting as the default implementation here
        return smoothed

    def distance_measure(self):
        """Distance measure between two time series

        Expected to return a function that accepts two time series and returns
        the distance.
        """

        return _py4dgeo.normalized_dtw_distance

    def find_seedpoints(self, distances):
        """Calculate seedpoints for the region growing algorithm"""

        algo = ruptures.Window(width=24, model="l1", min_size=12, jump=1)
        seeds = []

        # Iterate over all time series to analyse their change points
        for i in range(distances.shape[0]):
            # Run detection of change points
            changepoints = algo.fit_predict(distances[i, :], pen=1.0)

            # Iterate over the start/end pairs only covering signals that
            # have both a start and end point.
            for start, end in zip(changepoints[::2], changepoints[1::2]):
                seeds.append(RegionGrowingSeed(i, start, end))

        return seeds

    def sort_seedpoints(self, seeds):
        """Sort seed points by priority"""

        # Here, we simply sort by length of the change event
        return list(reversed(sorted(seeds, key=lambda x: x.end_epoch - x.start_epoch)))

    def run(self, analysis):
        """Calculate the segmentation"""

        # Smooth the distance array
        smoothed = self.temporal_averaging(analysis.distances)

        # Get corepoints from M3C2 class and build a KDTree on them
        corepoints = as_epoch(analysis.corepoints)
        corepoints.build_kdtree()

        # Calculate the list of seed points
        seeds = self.find_seedpoints(smoothed)
        seeds = self.sort_seedpoints(seeds)
        objects = []

        # Iterate over the seeds to maybe turn them into objects
        for seed in seeds:
            # Check all already calculated objects whether they overlap with this seed.
            found = False
            for obj in objects:
                if seed.index in obj.indices and (
                    obj.end_epoch > seed.start_epoch or seed.end_epoch > obj.start_epoch
                ):
                    found = True
                    break

            # If we found an overlap, we skip this seed
            if found:
                continue

            data = _py4dgeo.RegionGrowingAlgorithmData(
                smoothed,
                corepoints,
                self.neighborhood_radius,
                seed._seed,
                self.thresholds,
            )

            # Perform the region growing
            objdata = _py4dgeo.region_growing(data, self.distance_measure())
            objects.append(ObjectByChange(objdata))

        return objects


class RegionGrowingSeed:
    def __init__(self, index, start_epoch, end_epoch):
        self._seed = _py4dgeo.RegionGrowingSeed(index, start_epoch, end_epoch)

    @property
    def index(self):
        return self._seed.index

    @property
    def start_epoch(self):
        return self._seed.start_epoch

    @property
    def end_epoch(self):
        return self._seed.end_epoch


class ObjectByChange:
    """Representation a change object in the spatiotemporal domain"""

    def __init__(self, data):
        self._data = data

    @property
    def indices(self):
        """The set of corepoint indices that compose the object by change"""
        return self._data.indices

    @property
    def start_epoch(self):
        """The index of the start epoch of the change object"""
        return self._data.start_epoch

    @property
    def end_epoch(self):
        """The index of the end epoch of the change object"""
        return self._data.end_epoch

    @property
    def threshold(self):
        """The distance threshold that produced this object"""
        return self._data.threshold


def check_epoch_timestamp(epoch):
    """Validate an epoch to be used with SpatiotemporalSegmentation"""
    if epoch.timestamp is None:
        raise Py4DGeoError(
            "Epochs need to define a timestamp to be usable in SpatiotemporalSegmentation"
        )

    return epoch
