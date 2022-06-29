from py4dgeo.epoch import Epoch, as_epoch
from py4dgeo.logger import logger_context
from py4dgeo.util import Py4DGeoError, find_file
from py4dgeo.zipfile import UpdateableZipFile

import datetime
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import ruptures
import seaborn
import tempfile
import zipfile

import py4dgeo._py4dgeo as _py4dgeo


# Get the py4dgeo logger instance
logger = logging.getLogger("py4dgeo")


# This integer controls the versioning of the segmentation file format. Whenever the
# format is changed, this version should be increased, so that py4dgeo can warn
# about incompatibilities of py4dgeo with loaded data. This version is intentionally
# different from py4dgeo's version, because not all releases of py4dgeo necessarily
# change the segmentation file format and we want to be as compatible as possible.
PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION = 0


class SpatiotemporalAnalysis:
    def __init__(self, filename, compress=True, allow_pickle=True, force=False):
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
        :param allow_pickle:
            Whether py4dgeo is allowed to use the pickle module to store some data
            in the file representation of the analysis. If set to false, some data
            may not be stored and needs to be recomputed instead.
        :type allow_pickle: bool
        :param force:
            Force creation of a new analysis object, even if a file of this name
            already exists.
        """

        # Store the given parameters
        self.filename = find_file(filename, fatal=False)
        self.compress = compress
        self.allow_pickle = allow_pickle

        # Instantiate some properties used later on
        self._m3c2 = None

        # This is the cache for lazily loaded data
        self._corepoints = None
        self._distances = None
        self._smoothed_distances = None
        self._uncertainties = None
        self._reference_epoch = None

        # If the filename does not already exist, we create a new archive
        if force or not os.path.exists(self.filename):
            logger.info(f"Creating analysis file {self.filename}")
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

        if self._reference_epoch is None:
            with zipfile.ZipFile(self.filename, mode="r") as zf:
                # Double check that the reference has already been set
                if "reference_epoch.zip" not in zf.namelist():
                    raise Py4DGeoError("Reference epoch for analysis not yet set")

                # Extract it from the archive
                with tempfile.TemporaryDirectory() as tmp_dir:
                    ref_epochfile = zf.extract("reference_epoch.zip", path=tmp_dir)
                    self._reference_epoch = Epoch.load(ref_epochfile)

        return self._reference_epoch

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

        # Also cache it directly
        self._reference_epoch = epoch

    @reference_epoch.deleter
    def reference_epoch(self):
        self._reference_epoch = None

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
                    self._corepoints = Epoch.load(cpfile)

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

    @corepoints.deleter
    def corepoints(self):
        self._corepoints = None

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

        if self._distances is None:
            with zipfile.ZipFile(self.filename, mode="r") as zf:
                filename = self._numpy_filename("distances")
                if filename not in zf.namelist():
                    self.distances = np.empty(
                        (self.corepoints.cloud.shape[0], 0), dtype=np.float64
                    )
                    return self._distances

                with tempfile.TemporaryDirectory() as tmp_dir:
                    distancefile = zf.extract(filename, path=tmp_dir)
                    read_func = (
                        (lambda f: np.load(f)["arr_0"]) if self.compress else np.load
                    )
                    self._distances = read_func(distancefile)

        return self._distances

    @distances.setter
    def distances(self, _distances):
        """Set the distances manually

        This is only possible exactly once and mutually exclusive with adding
        epochs via the :ref:`add_epochs` method.
        """
        with zipfile.ZipFile(self.filename, mode="a") as zf:
            filename = self._numpy_filename("distances")
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

        self._distances = _distances

    @distances.deleter
    def distances(self):
        self._distances = None

    @property
    def smoothed_distances(self):
        if self._smoothed_distances is None:
            with zipfile.ZipFile(self.filename, mode="r") as zf:
                filename = self._numpy_filename("smoothed_distances")
                if filename in zf.namelist():
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        smoothedfile = zf.extract(filename, path=tmp_dir)
                        read_func = (
                            (lambda f: np.load(f)["arr_0"])
                            if self.compress
                            else np.load
                        )
                        self._smoothed_distances = read_func(smoothedfile)

        return self._smoothed_distances

    @smoothed_distances.setter
    def smoothed_distances(self, _smoothed_distances):
        with zipfile.ZipFile(self.filename, mode="a") as zf:
            filename = self._numpy_filename("smoothed_distances")
            write_func = np.savez_compressed if self.compress else np.save

            with tempfile.TemporaryDirectory() as tmp_dir:
                smoothedfile = os.path.join(tmp_dir, filename)
                write_func(smoothedfile, _smoothed_distances)
                zf.write(smoothedfile, arcname=filename)

        self._smoothed_distances = _smoothed_distances

    @smoothed_distances.deleter
    def smoothed_distances(self):
        self._smoothed_distances = None

    @property
    def uncertainties(self):
        """Access the M3C2 uncertainties of this analysis"""

        if self._uncertainties is None:
            with zipfile.ZipFile(self.filename, mode="r") as zf:
                filename = self._numpy_filename("uncertainties")
                if filename not in zf.namelist():
                    self.uncertainties = np.empty(
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
                    return self._uncertainties

                with tempfile.TemporaryDirectory() as tmp_dir:
                    uncertaintyfile = zf.extract(filename, path=tmp_dir)
                    read_func = (
                        (lambda f: np.load(f)["arr_0"]) if self.compress else np.load
                    )
                    self._uncertainties = read_func(uncertaintyfile)

        return self._uncertainties

    @uncertainties.setter
    def uncertainties(self, _uncertainties):
        """Set the uncertainties manually

        This is only possible exactly once and mutually exclusive with adding
        epochs via the :ref:`add_epochs` method.
        """
        with zipfile.ZipFile(self.filename, mode="a") as zf:
            filename = self._numpy_filename("uncertainties")
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

        self._uncertainties = _uncertainties

    @uncertainties.deleter
    def uncertainties(self):
        self._uncertainties = None

    def add_epochs(self, *epochs):
        """Add a numbers of epochs to the existing analysis"""

        # Remove intermediate results from the archive
        self.invalidate_results()

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
        for i, epoch in enumerate(sorted(epochs, key=lambda e: e.timestamp)):
            with logger_context(f"Adding epoch {i+1}/{len(epochs)} to analysis object"):
                # Prepare the M3C2 instance
                self.m3c2.corepoints = self.corepoints.cloud
                self.m3c2.epochs = (reference_epoch, epoch)

                # Calculate the M3C2 distances
                d, u = self.m3c2.calculate_distances(reference_epoch, epoch)
                new_distances.append(d)
                new_uncertainties.append(u)
                timedeltas.append(epoch.timestamp - reference_epoch.timestamp)

        # We do not need the reference_epoch at this point
        del self.reference_epoch

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

            # Depending on whether we compress, we use different numpy functionality
            write_func = np.savez_compressed if self.compress else np.save
            distance_filename = self._numpy_filename("distances")
            uncertainty_filename = self._numpy_filename("uncertainties")

            with logger_context("Rearranging space-time array in memory"):
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

            # Invalidate potential caches for distances/uncertainties
            self._distances = None
            self._uncertainties = None

            # Dump the updated files into the archive
            with logger_context("Updating disk-based analysis archive with new epochs"):
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

        # (Potentially) remove caches
        del self.distances
        del self.uncertainties

    @property
    def seeds(self):
        """The list of seed candidates for this analysis"""

        with zipfile.ZipFile(self.filename, mode="r") as zf:
            if "seeds.pickle" not in zf.namelist():
                return None

            with tempfile.TemporaryDirectory() as tmp_dir:
                zf.extract("seeds.pickle", path=tmp_dir)
                with open(os.path.join(tmp_dir, "seeds.pickle"), "rb") as f:
                    return pickle.load(f)

    @seeds.setter
    def seeds(self, _seeds):
        # Assert that we received the correct type
        for seed in _seeds:
            if not isinstance(seed, RegionGrowingSeed):
                raise Py4DGeoError(
                    "Seeds are expected to inherit from RegionGrowingSeed"
                )

        if not self.allow_pickle:
            return

        with UpdateableZipFile(self.filename, mode="a") as zf:
            if "seeds.pickle" in zf.namelist():
                zf.remove("seeds.pickle")

            with tempfile.TemporaryDirectory() as tmp_dir:
                seedsfile = os.path.join(tmp_dir, "seeds.pickle")
                with open(seedsfile, "wb") as f:
                    pickle.dump(_seeds, f)

                zf.write(seedsfile, arcname="seeds.pickle")

    @property
    def objects(self):
        """The list of objects by change for this analysis"""

        with zipfile.ZipFile(self.filename, mode="r") as zf:
            if "objects.pickle" not in zf.namelist():
                return None

            with tempfile.TemporaryDirectory() as tmp_dir:
                zf.extract("objects.pickle", path=tmp_dir)
                with open(os.path.join(tmp_dir, "objects.pickle"), "rb") as f:
                    return pickle.load(f)

    @objects.setter
    def objects(self, _objects):
        # Assert that we received the correct type
        for seed in _objects:
            if not isinstance(seed, ObjectByChange):
                raise Py4DGeoError(
                    "Objects are expected to inherit from ObjectByChange"
                )

        if not self.allow_pickle:
            return

        with UpdateableZipFile(self.filename, mode="a") as zf:
            if "objects.pickle" in zf.namelist():
                zf.remove("objects.pickle")

            with tempfile.TemporaryDirectory() as tmp_dir:
                objectsfile = os.path.join(tmp_dir, "objects.pickle")
                with open(objectsfile, "wb") as f:
                    pickle.dump(_objects, f)

                zf.write(objectsfile, arcname="objects.pickle")

    def invalidate_results(self, seeds=True, objects=True, smoothed_distances=True):
        """Invalidate (and remove) calculated results

        This is automatically called when new epochs are added or when
        an algorithm sets the :code:`force` option.
        """

        logger.info(
            f"Removing intermediate results from the analysis file {self.filename}"
        )
        with UpdateableZipFile(self.filename, mode="a") as zf:
            if seeds and "seeds.pickle" in zf.namelist():
                zf.remove("seeds.pickle")

            if objects and "objects.pickle" in zf.namelist():
                zf.remove("objects.pickle")

            smoothed_file = self._numpy_filename("smoothed_distances")
            if smoothed_distances and smoothed_file in zf.namelist():
                zf.remove(smoothed_file)

    def _numpy_filename(self, name):
        extension = "npz" if self.compress else "npy"
        return f"{name}.{extension}"

    @property
    def distances_for_compute(self):
        """Retrieve the distance array used for computation

        This might be the raw data or smoothed data, based on whether
        a smoothing was provided by the user.
        """
        distances = self.smoothed_distances
        if distances is None:
            distances = self.distances
        return distances


class RegionGrowingAlgorithmBase:
    def __init__(
        self,
        neighborhood_radius=1.0,
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        min_segments=20,
        max_segments=None,
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
        :param min_segments:
            The minimum number of core points in an object by change. Defaults to
            20.
        :type min_segments: int
        :param max_segments:
            The maximum number of core points in an object by change. This is mainly
            used to bound the runtime of expensive region growing. By default, no
            maximum is applied.
        :type max_segments: int
        """

        self.neighborhood_radius = neighborhood_radius
        self.thresholds = thresholds
        self.min_segments = min_segments
        self.max_segments = max_segments

        self._analysis = None

    def distance_measure(self):
        """Distance measure between two time series

        Expected to return a function that accepts two time series and returns
        the distance.
        """

        return _py4dgeo.normalized_dtw_distance

    def find_seedpoints(self):
        """Calculate seedpoints for the region growing algorithm"""

        raise NotImplementedError

    def seed_sorting_scorefunction(self):
        """A function that computes a score for a seed candidate

        This function is used to prioritize seed candidates.
        """

        # The base class does not perform sorting.
        return lambda seed: 0.0

    def filter_objects(self, obj):
        """A filter for objects produced by the region growing algorithm

        Objects are discarded if this method returns False.
        """

        # The base class does not perform filtering
        return True

    @property
    def analysis(self):
        """Access the analysis object that the algorithm operates on

        This is only available after :ref:`run` has been called.
        """
        if self._analysis is None:
            raise Py4DGeoError(
                "Analysis object is only available when the algorithm is run"
            )
        return self._analysis

    def run(self, analysis, force=False):
        """Calculate the segmentation

        :param analysis:
            The analysis object we are working with
        :type analysis: py4dgeo.segmentation.SpatiotemporalAnalysis
        :param force:
            Force recalculation of results. If false, some intermediate results will be
            restored from the analysis object instead of being recalculated
        """

        # Make the analysis object known to all members
        self._analysis = analysis

        # Enforce the removal of intermediate results
        if force:
            analysis.invalidate_results()

        # Return pre-calculated objects if they are available
        precalculated = analysis.objects
        if precalculated is not None:
            logger.info("Reusing objects by change stored in analysis object")
            return precalculated

        # Get corepoints from M3C2 class and build a KDTree on them
        corepoints = as_epoch(analysis.corepoints)
        corepoints.build_kdtree()

        # Calculate the list of seed points
        seeds = analysis.seeds
        if seeds is None:
            with logger_context("Find seed candidates in time series"):
                seeds = self.find_seedpoints()
                analysis.seeds = seeds
        else:
            logger.info("Reusing seed candidates stored in analysis object")

        # Sort the seed points
        with logger_context("Sort seed candidates by priority"):
            seeds = list(sorted(seeds, key=self.seed_sorting_scorefunction()))

        objects = []

        # Iterate over the seeds to maybe turn them into objects
        for i, seed in enumerate(seeds):
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

            # Apply a numeric default to the max_segments parameter
            max_segments = self.max_segments
            if max_segments is None:
                max_segments = corepoints.cloud.shape[0] + 1

            data = _py4dgeo.RegionGrowingAlgorithmData(
                analysis.distances_for_compute,
                corepoints,
                self.neighborhood_radius,
                seed._seed,
                self.thresholds,
                self.min_segments,
                max_segments,
            )

            # Perform the region growing
            with logger_context(
                f"Performing region growing on seed candidate {i+1}/{len(seeds)}"
            ):
                objdata = _py4dgeo.region_growing(data, self.distance_measure())

                # If the returned object has 0 indices, the min_segments threshold was violated
                if objdata.indices_distances:
                    obj = ObjectByChange(objdata, seed, analysis)
                    if self.filter_objects(obj):
                        objects.append(obj)

                # If the returned object is larger than max_segments we issue a warning
                if len(objdata.indices_distances) >= max_segments:
                    logger.warning(
                        f"An object by change exceeded the given maximum size of {max_segments}"
                    )

        # Store the results in the analysis object
        analysis.objects = objects

        # Potentially remove objects from memory
        del analysis.smoothed_distances
        del analysis.distances

        return objects


class RegionGrowingAlgorithm(RegionGrowingAlgorithmBase):
    def __init__(self, seed_subsampling=1, **kwargs):
        """Construct the 4D-OBC algorithm.

        :param seed_subsampling:
            A subsampling factor for the set of corepoints for the generation
            of region growing seed candidates. This can be used to speed up
            the generation of seeds. The default of 1 does not perform any
            subsampling, a value of e.g. 10 would only consider every 10th
            corepoint for adding seeds.
        :type seed_subsampling: int
        """

        # Initialize base class
        super().__init__(**kwargs)

        # Store the given parameters
        self.seed_subsampling = seed_subsampling

    def find_seedpoints(self):
        """Calculate seedpoints for the region growing algorithm"""

        # These are some arguments used below that we might consider
        # exposing to the user
        window_width = 24
        window_costmodel = "l1"
        window_min_size = 12
        window_jump = 1
        window_penalty = 1.0
        minperiod = 24
        height_threshold = 0.0

        # The chang point detection algorithm we use
        algo = ruptures.Window(
            width=window_width,
            model=window_costmodel,
            min_size=window_min_size,
            jump=window_jump,
        )

        # The list of generated seeds
        seeds = []

        # Iterate over all time series to analyse their change points
        for i in range(
            0, self.analysis.distances_for_compute.shape[0], self.seed_subsampling
        ):
            # Extract the time series and interpolate its nan values
            timeseries = self.analysis.distances_for_compute[i, :]
            bad_indices = np.isnan(timeseries)
            num_nans = np.count_nonzero(bad_indices)

            # If we too many nans, this timeseries does not make sense
            if num_nans > timeseries.shape[0] - 3:
                continue

            # If there are nan values, we try fixing things by interpolation
            if num_nans > 0:
                good_indices = np.logical_nor(bad_indices)
                timeseries[bad_indices] = np.interp(
                    bad_indices.nonzero()[0],
                    good_indices.nonzero()[0],
                    timeseries[good_indices],
                )

            # Run detection of change points
            changepoints = algo.fit_predict(timeseries, pen=window_penalty)[:-1]

            # Shift the time series to positive values
            timeseries = timeseries + abs(np.nanmin(timeseries) + 0.1)
            timeseries_flipped = timeseries * -1.0 + abs(np.nanmax(timeseries)) + 0.1

            # Create seeds for this timeseries
            corepoint_seeds = []
            for start_idx in changepoints:
                # Skip this changepoint if it was included into a previous seed
                if corepoint_seeds and start_idx <= corepoint_seeds[-1].end_epoch:
                    continue

                # Skip this changepoint if this to close to the end
                if start_idx >= timeseries.shape[0] - minperiod:
                    break

                # Decide whether we need use the flipped timeseries
                used_timeseries = timeseries
                if timeseries[start_idx] >= timeseries[start_idx + minperiod]:
                    used_timeseries = timeseries_flipped

                previous_volume = -999.9

                for target_idx in range(start_idx + 1, timeseries.shape[0] - minperiod):

                    # Calculate the change volume
                    height = used_timeseries[start_idx] + height_threshold
                    volume = np.nansum(
                        used_timeseries[start_idx : target_idx + 1] - height
                    )
                    if volume < 0.0:
                        height = used_timeseries[start_idx]
                        volume = np.nansum(
                            used_timeseries[start_idx : target_idx + 1] - height
                        )

                    # Check whether the volume started decreasing
                    # TODO: Didn't we explicitly enforce positivity of the series?
                    if previous_volume > volume:
                        corepoint_seeds.append(
                            RegionGrowingSeed(i, start_idx, target_idx)
                        )
                        break
                    else:
                        previous_volume = volume

                # We reached the present and add a seed based on it
                corepoint_seeds.append(
                    RegionGrowingSeed(i, start_idx, timeseries.shape[0] - 1)
                )

            # Add all the seeds found for this corepoint to the full list
            seeds.extend(corepoint_seeds)

        return seeds

    def seed_sorting_scorefunction(self):
        """Neighborhood similarity sorting function"""

        # The 4D-OBC algorithm sorts by similarity in the neighborhood
        # of the seed.
        def neighborhood_similarity(seed):
            neighbors = self.analysis.corepoints.kdtree.radius_search(
                self.analysis.corepoints.cloud[seed.index, :], self.neighborhood_radius
            )
            similarities = []
            for n in neighbors:
                data = _py4dgeo.TimeseriesDistanceFunctionData(
                    self.analysis.distances_for_compute[
                        seed.index, seed.start_epoch : seed.end_epoch + 1
                    ],
                    self.analysis.distances_for_compute[
                        n, seed.start_epoch : seed.end_epoch + 1
                    ],
                )
                similarities.append(self.distance_measure()(data))

            return sum(similarities, 0.0) / (len(neighbors) - 1)

        return neighborhood_similarity

    def filter_objects(self, obj):
        """A filter for objects produced by the region growing algorithm"""

        # Filter based on coefficient of variation
        distarray = np.fromiter(obj._data.indices_distances.values(), np.float64)
        cv = np.std(distarray) / np.mean(distarray)

        # TODO: Make this threshold configurable?
        return cv <= 0.8


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

    def __init__(self, data, seed, analysis=None):
        self._data = data
        self._analysis = analysis
        self.seed = seed

    @property
    def indices(self):
        """The set of corepoint indices that compose the object by change"""
        return list(self._data.indices_distances.keys())

    def distance(self, index):
        return self._data.indices_distances[index]

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

    def plot(self, filename=None):
        """Create an informative visualization of the Object By Change

        :param filename:
            The filename to use to store the plot. Can be omitted to only show
            plot in a Jupyter notebook session.
        :type filename: str
        """

        # Extract DTW distances from this object
        indexarray = np.fromiter(self.indices, np.int32)
        distarray = np.fromiter((self.distance(i) for i in indexarray), np.float64)

        # Intitialize the figure and all of its subfigures
        fig = plt.figure(figsize=plt.figaspect(0.3))
        tsax = fig.add_subplot(1, 3, 1)
        histax = fig.add_subplot(1, 3, 2)
        mapax = fig.add_subplot(1, 3, 3)

        # The first plot (tsax) prints all time series of chosen corepoints
        # and colors them according to distance.
        tsax.set_ylabel("Height change [m]")
        tsax.set_xlabel("Time [h]")

        # We pad the time series visualization with a number of data
        # points on both sides. TODO: Expose as argument to plot?
        timeseries_padding = 10
        start_epoch = max(self.start_epoch - timeseries_padding, 0)
        end_epoch = min(
            self.end_epoch + timeseries_padding,
            self._analysis.distances_for_compute.shape[1],
        )

        # We use the seed's timeseries to set good axis limits
        seed_ts = self._analysis.distances_for_compute[
            self.seed.index, start_epoch:end_epoch
        ]
        tsax.set_ylim(np.nanmin(seed_ts) * 0.5, np.nanmax(seed_ts) * 1.5)

        # Create a colormap with distance for this object
        cmap = plt.cm.get_cmap("viridis")
        maxdist = np.nanmax(distarray)

        # Plot each time series individually
        for index in self.indices:
            tsax.plot(
                self._analysis.distances_for_compute[index, start_epoch:end_epoch],
                linewidth=0.7,
                alpha=0.3,
                color=cmap(self.distance(index) / maxdist),
            )

        # Plot the seed timeseries again, but with a thicker line
        tsax.plot(seed_ts, linewidth=2.0, zorder=10, color="blue")

        # Next, we add a histogram plot with the distance values (using seaborn)
        seaborn.histplot(distarray, ax=histax, kde=True, color="r")

        # Add labels to the histogram plot
        histax.set_title(f"Segment size: {distarray.shape[0]}")
        histax.set_xlabel("DTW distance")

        # Create a 2D view of the segment
        locations = self._analysis.corepoints.cloud[indexarray, 0:2]
        mapax.scatter(locations[:, 0], locations[:, 1], c=distarray)

        # Some global settings of the generated figure
        fig.tight_layout()

        # Maybe save to file
        if filename is not None:
            plt.savefig(filename)


def check_epoch_timestamp(epoch):
    """Validate an epoch to be used with SpatiotemporalSegmentation"""
    if epoch.timestamp is None:
        raise Py4DGeoError(
            "Epochs need to define a timestamp to be usable in SpatiotemporalSegmentation"
        )

    return epoch


def regular_corepoint_grid(lowerleft, upperright, num_points, zval=0.0):
    """A helper function to create a regularly spaced grid for the analysis

    :param lowerleft:
        The lower left corner of the grid. Given as a 2D coordinate.
    :type lowerleft: np.ndarray
    :param upperright:
        The upper right corner of the grid. Given as a 2D coordinate.
    :type upperright: nd.ndarray
    :param num_points:
        A tuple with two entries denoting the number of points to be used in
        x and y direction
    :type num_points: tuple
    :param zval:
        The value to fill for the z-direction.
    :type zval: double
    """
    xspace = np.linspace(
        lowerleft[0], upperright[0], num=num_points[0], dtype=np.float64
    )
    yspace = np.linspace(
        lowerleft[1], upperright[1], num=num_points[1], dtype=np.float64
    )

    grid = np.empty(shape=(num_points[0] * num_points[1], 3), dtype=np.float64)
    for i, x in enumerate(xspace):
        for j, y in enumerate(yspace):
            grid[i * num_points[0] + j, 0] = x
            grid[i * num_points[0] + j, 1] = y
            grid[i * num_points[0] + j, 2] = zval

    return grid


def temporal_averaging(distances, smoothing_window=24):
    """Smoothen a space-time array of distance change using a sliding window approach

    :param distances:
        The raw data to smoothen.
    :type distances: np.ndarray
    :param smoothing_window:
        The size of the sliding window used in smoothing the data. The
        default value of 0 does not perform any smooting.
    :type smooting_window: int
    """

    with logger_context("Smoothing temporal data"):
        smoothed = np.empty_like(distances)
        eps = smoothing_window // 2

        for i in range(distances.shape[1]):
            smoothed[:, i] = np.nanmedian(
                distances[
                    :,
                    max(0, i - eps) : min(distances.shape[1] - 1, i + eps),
                ],
                axis=1,
            )

        # We use no-op smooting as the default implementation here
        return smoothed
