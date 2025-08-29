from nbclient.client import timestamp
from py4dgeo.epoch import Epoch, as_epoch
from py4dgeo.logger import logger_context
from py4dgeo.util import Py4DGeoError, find_file
from py4dgeo.UpdateableZipFile import UpdateableZipFile
from py4dgeo.segmentation import RegionGrowingSeed

import datetime
import json
import logging
import matplotlib
import numpy as np
import os
import pickle
import seaborn
import tempfile
import zipfile
import py4dgeo
import matplotlib.pyplot as plt

import rdp
from sklearn.linear_model import LinearRegression

import _py4dgeo


# Get the py4dgeo logger instance
logger = logging.getLogger("py4dgeo")


# This integer controls the versioning of the _segmentation file format. Whenever the
# format is changed, this version should be increased, so that py4dgeo can warn
# about incompatibilities of py4dgeo with loaded data. This version is intentionally
# different from py4dgeo's version, because not all releases of py4dgeo necessarily
# change the _segmentation file format and we want to be as compatible as possible.
PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION = 0


class SpatiotemporalAnalysis:
    def __init__(self, filename, compress=True, allow_pickle=True, force=False):
        """Construct a spatiotemporal _segmentation object

        This is the basic data structure for the 4D objects by change algorithm
        and its derived variants. It manages storage of M3C2 distances and other
        intermediate results for a time series of epochs. The original point clouds
        themselves are not needed after initial distance calculation and additional
        epochs can be added to an existing analysis. The class uses a disk backend
        to store information and allows lazy loading of additional data like e.g.
        M3C2 uncertainty values for postprocessing.

        :param filename:
            The filename used for this analysis. If it does not exist on the file
            system, a new analysis is created. Otherwise, the data is loaded from the existent file.
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
                # Write the _segmentation file format version number
                zf.writestr(
                    "SEGMENTATION_FILE_FORMAT",
                    str(PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION),
                )

                # Write the compression algorithm used for all suboperations
                zf.writestr("USE_COMPRESSION", str(self.compress))

        # Assert that the _segmentation file format is still valid
        with zipfile.ZipFile(self.filename, mode="r") as zf:
            # Read the _segmentation file version number and compare to current
            version = int(zf.read("SEGMENTATION_FILE_FORMAT").decode())
            if version != PY4DGEO_SEGMENTATION_FILE_FORMAT_VERSION:
                raise Py4DGeoError("_segmentation file format is out of date!")

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

            # Ensure that the tearch tree is built - no-op if triggered by the user
            epoch._validate_search_tree()

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

            # Ensure that the corepoints are stored as an epoch and its search trees are built
            self._corepoints = as_epoch(_corepoints)
            self._corepoints._validate_search_tree()

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
       # for seed in _objects:
        #    if not isinstance(seed, ObjectByChange):
         #       raise Py4DGeoError(
          #          "Objects are expected to inherit from ObjectByChange"
           #     )

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

    def invalidate_results(self, seeds=True, objects=True, smoothed_distances=False):
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

    def merge(self, time_threshold=0.7, spatial_threshold=0.1, smoothing=False, smoothing_window=5):
        distance = self.distances_for_compute
        if distance is None:
            raise ValueError
        if smoothing:
            distance = temporal_averaging(self._distances, smoothing_window)

        change = []
        for obj in self.objects:
            if distance[obj.seed.index, obj.end_epoch] - distance[obj.seed.index, obj.start_epoch] < 0:
                change.append('negative')
            else:
                change.append('positive')

        merging = [[]] * len(self.objects)
        values = []
        for idx_act, obj_act in enumerate(self.objects):
            print(idx_act)
            for idx_it, obj_it in enumerate(self.objects):

                # identical 4D-OBC
                if idx_it == idx_act:
                    continue

                # temporal overlap
                IoAct_time = (min(obj_act.end_epoch, obj_it.end_epoch) - max(obj_act.start_epoch,
                                                                             obj_it.start_epoch)) / (
                                     obj_act.end_epoch - obj_act.start_epoch)
                IoIt_time = (min(obj_act.end_epoch, obj_it.end_epoch) - max(obj_act.start_epoch,
                                                                            obj_it.start_epoch)) / (
                                    obj_it.end_epoch - obj_it.start_epoch)

                # spatial overlap
                ident_cp = np.intersect1d(obj_act.indices, obj_it.indices)
                IoAct = len(ident_cp) / len(obj_act.indices)
                IoIt = len(ident_cp) / len(obj_it.indices)

                max_time = max(IoAct_time, IoIt_time)
                max_spatial = max(IoAct, IoIt)

                # if calculated overlap exceeds defined thresholds and change direction is equal -> store link between objects
                if max_time > time_threshold and max_spatial > spatial_threshold and change[idx_act] == change[
                    idx_it]:
                    merging[idx_act] = merging[idx_act] + [idx_it]
                    values.append([max_time, max_spatial])

        values = np.array(values)

        import copy
        visited = [False] * len(self.objects)
        merged_idxs = []

        for idx_act, lst in enumerate(merging):

            if visited[idx_act] == True:
                continue
            else:
                visited[idx_act] = True
                merged_idxs.append([idx_act])

                to_visit = copy.deepcopy(lst)
                while to_visit:
                    idx_next = to_visit.pop(0)
                    if visited[idx_next] == True:
                        continue
                    else:
                        visited[idx_next] = True
                        merged_idxs[-1] = merged_idxs[-1] + [idx_next]
                        to_visit = to_visit + merging[idx_next]

        from functools import reduce
        merged_4dobcs = []
        for idx in merged_idxs:
            indices = [self.objects[i].indices for i in idx]
            start_epochs = [self.objects[i].start_epoch for i in idx]
            end_epochs = [self.objects[i].end_epoch for i in idx]

            indices_merge = reduce(np.union1d, indices)
            start_epoch_merge = min(start_epochs)
            end_epoch_merge = max(end_epochs)

            merged_4dobcs.append(
                MergedObjectsOfChange(indices_merge, start_epoch_merge, end_epoch_merge, [i for i in idx], self,
                                      distance))
        return merged_4dobcs

    def extract(self, method: str, smoothing_window=5, seed_subsampling=1, max_change_period = 200, data_gap: int | None = None):
        print(f"Method received: {method}")
        if method == 'RDP':
            timestamps = [t + self.reference_epoch.timestamp for t in self.timedeltas]
            lod = self.uncertainties['lodetection']
            mean_lod = np.nanmean(lod)
            smoothed = np.empty_like(self.distances)
            eps = smoothing_window // 2
            timedelta_max = 5
            time_day = np.array([(t - self.reference_epoch.timestamp).total_seconds() / (3600 * 24) for t in timestamps])

            if np.isnan(self.distances[0]).all():
                self.distances[0] = 0

            for i in range(self.distances.shape[1]):
                day_act = time_day[i]
                day_limit = [day_act - timedelta_max, day_act + timedelta_max]
                idx_limit = [np.where(time_day <= day_limit[0])[0], np.where(time_day >= day_limit[1])[0]]

                if idx_limit[0].size != 0:
                    idx_limit[0] = idx_limit[0][-1]
                else:
                    idx_limit[0] = 0

                if idx_limit[1].size != 0:
                    idx_limit[1] = idx_limit[1][0]
                else:
                    idx_limit[1] = self.distances.shape[1]

                # if idx_limit[0].size > 0:
                #     idx_limit[0] = idx_limit[0][-1]
                # else:
                #     idx_limit[0] = 0  # or another default value
                #
                # if idx_limit[1].size > 0:
                #     idx_limit[1] = idx_limit[1][0]
                # else:
                #     idx_limit[1] = self.distances.shape[1]
                smoothed[:, i] = np.nanmedian(
                    self.distances[:,
                    max(0, i - eps, idx_limit[0]): min(self.distances.shape[1] - 1, i + eps, idx_limit[1])], axis=1,)

            self.smoothed_distances = smoothed
            algo = LinearChangeSeeds_rdp(neighborhood_radius=0.2,
                                            min_segments=50,
                                            max_segments=10000,
                                            thresholds=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                            epsilon = mean_lod,
                                            height_threshold = 0.20,
                                         seed_subsampling=seed_subsampling,
                                         max_change_period = max_change_period,
                                         data_gap = data_gap)

            print('Algorithm started')
            algo.run(self)


        elif method == 'DTR':
            timestamps = [t + self.reference_epoch.timestamp for t in self.timedeltas]
            corepoints = self.corepoints.cloud
            lod = self.uncertainties['lodetection']
            median_lod = np.nanmean(lod)
            smoothed = np.empty_like(self.distances)
            eps = smoothing_window // 2
            if np.isnan(self.distances[0]).all():
                self.distances[0] = 0

            timedelta_max = 5
            time_day = np.array([(t - self.reference_epoch.timestamp).total_seconds() / (3600 * 24) for t in timestamps])

            for i in range(self.distances.shape[1]):
                print(i)
                day_act = time_day[i]
                day_limit = [day_act - timedelta_max, day_act + timedelta_max]

                idx_limit = [np.where(time_day <= day_limit[0])[0], np.where(time_day >= day_limit[1])[0]]

                if idx_limit[0].size != 0:
                    idx_limit[0] = idx_limit[0][-1]
                else:
                    idx_limit[0] = 0

                if idx_limit[1].size != 0:
                    idx_limit[1] = idx_limit[1][0]
                else:
                    idx_limit[1] = self.distances.shape[1]

                # if idx_limit[0].size > 0:
                #     idx_limit[0] = idx_limit[0][-1]
                # else:
                #     idx_limit[0] = 0  # or another default value
                #
                # if idx_limit[1].size > 0:
                #     idx_limit[1] = idx_limit[1][0]
                # else:
                #     idx_limit[1] = self.distances.shape[1]  # max index + 1 for slicing
                smoothed[:, i] = np.nanmedian(
                    self.distances[:, max(0, i - eps, idx_limit[0]): min(self.distances.shape[1] - 1, i + eps, idx_limit[1])],
                    axis=1,
                )

            self.smoothed_distances = smoothed

            algo = LinearChangeSeeds_dtr(neighborhood_radius=0.2,
                                         min_segments=50,
                                         max_segments=10000,
                                         thresholds=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                         epsilon=median_lod,
                                         height_threshold=0.20,
                                         seed_subsampling=seed_subsampling,
                                         max_change_period=max_change_period,
                                         data_gap=data_gap)

            algo.run(self)

        else:
            raise ValueError('Method must be RDP or DTR')

class RegionGrowingAlgorithmBase:
    def __init__(
        self,
        neighborhood_radius=1.0,
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        min_segments=20,
        max_segments=None,
    ):
        """Construct a spatiotemporal _segmentation algorithm.

        This class can be derived from to customize the algorithm behaviour.

        :param neighborhood_radius:
            The size of the neighborhood of a core point. All core points within
            this radius are considered adjacent and are therefore considered as
            candidates for inclusion in the region growing algorithm.
        :type neighborhood_radius: float
        :param thresholds:
            A list of thresholds to use as candidates in 4D-OBC's adaptive
            thresholding procedure.
        :type thresholds: list
        :param min_segments:
            The minimum number of core points in an object-by-change. Defaults to
            20.
        :type min_segments: int
        :param max_segments:
            The maximum number of core points in an object-by-change. This is mainly
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
        _py4dgeo.Epoch.set_default_radius_search_tree("octree")
        """Calculate the _segmentation

        :param analysis:
            The analysis object we are working with.
        :type analysis: py4dgeo.segmentation.SpatiotemporalAnalysis
        :param force:
            Force recalculation of results. If false, some intermediate results will be
            restored from the analysis object instead of being recalculated.
        """

        # Make the analysis object known to all members
        self._analysis = analysis

        # Enforce the removal of intermediate results
        if force:
            analysis.invalidate_results()

        # Return pre-calculated objects if they are available
        # precalculated = analysis.objects
        # if precalculated is not None:
        #     logger.info("Reusing objects by change stored in analysis object")
        #     return precalculated

        # Check if there are pre-calculated objects.
        # If so, create objects list from these and continue growing objects, taking into consideration objects that are already grown.
        # if not initiate new empty objects list
        precalculated = analysis.objects  # TODO: do not assign to new object
        if precalculated is not None:
            logger.info("Reusing objects by change stored in analysis object")
            objects = (
                precalculated.copy()
            )  # test if .copy() solves memory problem, or deepcopy?
        else:
            objects = (
                []
            )  # TODO: test initializing this in the analysis class, see if it crashes instantly

        # Get corepoints from M3C2 class and build a search tree on them
        corepoints = as_epoch(analysis.corepoints)
        corepoints._validate_search_tree()

        # Calculate the list of seed points and sort them
        seeds = analysis.seeds
        if seeds is None:
            with logger_context("Find seed candidates in time series"):
                seeds = self.find_seedpoints()

            # Sort the seed points
            with logger_context("Sort seed candidates by priority"):
                seeds = list(sorted(seeds, key=self.seed_sorting_scorefunction()))

            # Store the seeds
            analysis.seeds = seeds
        else:
            logger.info("Reusing seed candidates stored in analysis object")
        # write the number of seeds to a separate text file if self.write_nr_seeds is True
        if self.write_nr_seeds:
            with open("number_of_seeds.txt", "w") as f:
                f.write(str(len(seeds)))

        # Iterate over the seeds to maybe turn them into objects
        for i, seed in enumerate(
            seeds
        ):  # [self.resume_from_seed-1:]): # starting seed ranked at the `resume_from_seed` variable (representing 1 for index 0)
            # or to keep within the same index range when resuming from seed:
            if i < (
                self.resume_from_seed - 1
            ):  # resume from index 0 when `resume_from_seed` == 1
                continue
            if i >= (self.stop_at_seed - 1):  # stop at index 0 when `stop_at_seed` == 1
                break

            # save objects to analysis object when at index `intermediate_saving`
            if (
                (self.intermediate_saving)
                and ((i % self.intermediate_saving) == 0)
                and (i != 0)
            ):
                with logger_context(
                    f"Intermediate saving of first {len(objects)} objects, grown from first {i+1}/{len(seeds)} seeds"
                ):
                    analysis.objects = objects  # This assigns itself to itself

            # Check all already calculated objects whether they overlap with this seed.
            found = False
            for obj in objects:
                if seed.index in obj.indices and (
                    obj.end_epoch > seed.start_epoch
                    and seed.end_epoch > obj.start_epoch
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
                    obj = ObjectByChange(
                        objdata, seed, analysis
                    )  # TODO: check, does it copy the whole analysis object when initializing
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
    def __init__(
        self,
        seed_subsampling=1,
        seed_candidates=None,
        window_width=24,
        window_min_size=12,
        window_jump=1,
        window_penalty=1.0,
        minperiod=24,
        height_threshold=0.0,
        use_unfinished=True,
        intermediate_saving=0,
        resume_from_seed=0,
        stop_at_seed=np.inf,
        write_nr_seeds=False,
        **kwargs,
    ):
        """Construct the 4D-OBC algorithm.

        :param seed_subsampling:
            A subsampling factor for the set of core points for the generation
            of _segmentation seed candidates. This can be used to speed up
            the generation of seeds. The default of 1 does not perform any
            subsampling, a value of, e.g., 10 would only consider every 10th
            corepoint for adding seeds.
        :type seed_subsampling: int
        :param seed_candidates:
            A set of indices specifying which core points should be used for seed detection. This can be used to perform _segmentation for selected locations. The default of None does not perform any selection and uses all corepoints. The subsampling parameter is applied additionally.
        :type seed_candidates: list
        :param window_width:
            The width of the sliding temporal window for change point detection. The sliding window
            moves along the signal and determines the discrepancy between the first and the second
            half of the window (i.e. subsequent time series segments within the window width). The
            default value is 24, corresponding to one day in case of hourly data.
        :type window_width: int
        :param window_min_size:
            The minimum temporal distance needed between two seed candidates, for the second one to be considered.
            The default value is 1, such that all detected seeds candidates are considered.
        :type window_min_size: int
        :param window_jump:
            The interval on which the sliding temporal window moves and checks for seed candidates.
            The default value is 1, corresponding to a check for every epoch in the time series.
        :type window_jump: int
        :param window_penalty:
            A complexity penalty that determines how strict the change point detection is.
            A higher penalty results in stricter change point detection (i.e, fewer points are detected), while a low
            value results in a large amount of detected change points. The default value is 1.0.
        :type window_penalty: float
        :param minperiod:
            The minimum period of a detected change to be considered as seed candidate for subsequent
            _segmentation. The default is 24, corresponding to one day for hourly data.
        :type minperiod: int
        :param height_threshold:
            The height threshold represents the required magnitude of a detected change to be considered
            as seed candidate for subsequent _segmentation. The magnitude of a detected change is derived
            as unsigned difference between magnitude (i.e. distance) at start epoch and peak magnitude.
            The default is 0.0, in which case all detected changes are used as seed candidates.
        :type height_threshold: float
        :param use_unfinished:
            If False, seed candidates that are not finished by the end of the time series are not considered in further
            analysis. The default is True, in which case unfinished seed_candidates are regarded as seeds region growing.
        :type use_unfinished: bool
        :param intermediate_saving:
            Parameter that determines after how many considered seeds, the resulting list of 4D-OBCs is saved to the SpatiotemporalAnalysis object.
            This is to ensure that if the algorithm is terminated unexpectedly not all results are lost. If set to 0 no intermediate saving is done.
        :type intermediate_saving: int
        :param resume_from_seed:
            Parameter specifying from which seed index the region growing algorithm must resume. If zero all seeds are considered, starting from the highest ranked seed.
            Default is 0.
        :type resume_from_seed: int
        :param stop_at_seed:
            Parameter specifying at which seed to stop region growing and terminate the run function.
            Default is np.inf, meaning all seeds are considered.
        :type stop_at_seed: int
        :param write_nr_seeds:
            If True, after seed detection, a text file is written in the working directory containing the total number of detected seeds.
            This can be used to split up the consecutive 4D-OBC segmentation into different subsets.
            Default is False, meaning no txt file is written.
        :type write_nr_seeds: bool
        """

        # Initialize base class
        super().__init__(**kwargs)

        # Store the given parameters
        self.seed_subsampling = seed_subsampling
        self.seed_candidates = seed_candidates
        self.window_width = window_width
        self.window_min_size = window_min_size
        self.window_jump = window_jump
        self.window_penalty = window_penalty
        self.minperiod = minperiod
        self.height_threshold = height_threshold
        self.use_unfinished = use_unfinished
        self.intermediate_saving = intermediate_saving
        self.resume_from_seed = resume_from_seed
        self.stop_at_seed = stop_at_seed
        self.write_nr_seeds = write_nr_seeds

    def find_seedpoints(self):
        """Calculate seedpoints for the region growing algorithm"""

        # These are some arguments used below that we might consider
        # exposing to the user in the future. For now, they are considered
        # internal, but they are still defined here for readability.
        window_costmodel = "l1"
        # window_min_size = 12
        # window_jump = 1
        # window_penalty = 1.0

        # Before starting the process, we check if the user has set a reasonable window width parameter
        if self.window_width >= self.analysis.distances_for_compute.shape[1]:
            raise Py4DGeoError(
                "Window width cannot be larger than the length of the time series - please adapt parameter"
            )

        # The list of generated seeds
        seeds = []

        # The list of core point indices to check as seeds
        if self.seed_candidates is None:
            if self.seed_subsampling == 0:
                raise Py4DGeoError(
                    "Subsampling factor cannot be 0, use 1 or any integer larger than 1"
                )
            # Use all corepoints if no selection specified, considering subsampling
            seed_candidates_curr = range(
                0, self.analysis.distances_for_compute.shape[0], self.seed_subsampling
            )
        else:
            # Use the specified corepoint indices, but consider subsampling
            seed_candidates_curr = self.seed_candidates  # [::self.seed_subsampling]

        # Iterate over all time series to analyse their change points
        for i in seed_candidates_curr:
            # Extract the time series and interpolate its nan values
            timeseries = self.analysis.distances_for_compute[i, :]
            bad_indices = np.isnan(timeseries)
            num_nans = np.count_nonzero(bad_indices)

            # If we too many nans, this timeseries does not make sense
            if num_nans > timeseries.shape[0] - 3:
                continue

            # If there are nan values, we try fixing things by interpolation
            if num_nans > 0:
                good_indices = np.logical_not(bad_indices)
                timeseries[bad_indices] = np.interp(
                    bad_indices.nonzero()[0],
                    good_indices.nonzero()[0],
                    timeseries[good_indices],
                )

            # Run detection of change points
            cpdata = _py4dgeo.ChangePointDetectionData(
                ts=timeseries,
                window_size=self.window_width,
                min_size=self.window_min_size,
                jump=self.window_jump,
                penalty=self.window_penalty,
            )
            changepoints = _py4dgeo.change_point_detection(cpdata)[:-1]

            # Shift the time series to positive values
            timeseries = timeseries + abs(np.nanmin(timeseries) + 0.1)
            # create a flipped version for negative change volumes
            timeseries_flipped = timeseries * -1.0 + abs(np.nanmax(timeseries)) + 0.1

            # Create seeds for this timeseries
            corepoint_seeds = []
            for start_idx in changepoints:
                # Skip this changepoint if it was included into a previous seed
                if corepoint_seeds and start_idx <= corepoint_seeds[-1].end_epoch:
                    continue

                # Skip this changepoint if this to close to the end
                if start_idx >= timeseries.shape[0] - self.minperiod:
                    break

                # Decide whether we need use the flipped timeseries
                used_timeseries = timeseries
                if timeseries[start_idx] >= timeseries[start_idx + self.minperiod]:
                    used_timeseries = timeseries_flipped

                previous_volume = -999.9
                for target_idx in range(start_idx + 1, timeseries.shape[0]):
                    # Calculate the change volume
                    height = used_timeseries[start_idx]
                    volume = np.nansum(
                        used_timeseries[start_idx : target_idx + 1] - height
                    )

                    # Check whether the volume started decreasing
                    if previous_volume > volume:
                        # Only add seed if larger than the minimum period and height of the change form larger than threshold
                        if (target_idx - start_idx >= self.minperiod) and (
                            np.abs(
                                np.max(used_timeseries[start_idx : target_idx + 1])
                                - np.min(used_timeseries[start_idx : target_idx + 1])
                            )
                            >= self.height_threshold
                        ):
                            corepoint_seeds.append(
                                RegionGrowingSeed(i, start_idx, target_idx)
                            )
                        break
                    else:
                        previous_volume = volume

                    # This causes a seed to always be detected if the volume doesn't decrease before present
                    #  Useful when used in an online setting, can be filtered before region growing
                    # Only if the last epoch is reached we use the segment as seed
                    if (target_idx == timeseries.shape[0] - 1) and self.use_unfinished:
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
            self.analysis.corepoints._validate_search_tree()
            neighbors = self.analysis.corepoints._radius_search(
                self.analysis.corepoints.cloud[seed.index, :], self.neighborhood_radius
            )
            # if no neighbors are found make sure the algorithm continues its search but with a large dissimilarity
            if len(neighbors) < 2:
                return 9999999.0  # return very large number? or delete the seed point, but then also delete from the seeds list

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

        # Check if mean is 0.0, if so, set to very small value to avoid division by 0
        mean_distarray = np.mean(distarray)
        if mean_distarray == 0.0:
            mean_distarray = 10**-10

        # Calculate coefficient of variation
        cv = np.std(distarray) / mean_distarray

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
        cmap = matplotlib.colormaps.get_cmap("viridis")
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
        plt.show()

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
    :type upperright: np.ndarray
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

# class Merge:
#     def __init__(self, objects, smoothed_distances):
#         self.objects = objects
#         self.smoothed_distances = smoothed_distances
#         self.analysis = analysis
#
#     def change_direction(self):
#         change = []
#         for obj in self.objects:
#             if self.smoothed_distances[obj.seed.index, obj.end_epoch] - self.smoothed_distances[
#                 obj.seed.index, obj.start_epoch] < 0:
#                 change.append('negative')
#             else:
#                 change.append('positive')
#         return change
#
#     def overlap(self, change, time_threshold, spatial_threshold):
#         merging = [[]] * len(self.objects)
#         values = []
#         for idx_act, obj_act in enumerate(self.objects):
#             print(idx_act)
#             for idx_it, obj_it in enumerate(self.objects):
#
#                 # identical 4D-OBC
#                 if idx_it == idx_act:
#                     continue
#
#                 # temporal overlap
#                 IoAct_time = (min(obj_act.end_epoch, obj_it.end_epoch) - max(obj_act.start_epoch,
#                                                                              obj_it.start_epoch)) / (
#                                          obj_act.end_epoch - obj_act.start_epoch)
#                 IoIt_time = (min(obj_act.end_epoch, obj_it.end_epoch) - max(obj_act.start_epoch,
#                                                                             obj_it.start_epoch)) / (
#                                         obj_it.end_epoch - obj_it.start_epoch)
#
#                 # spatial overlap
#                 ident_cp = np.intersect1d(obj_act.indices, obj_it.indices)
#                 IoAct = len(ident_cp) / len(obj_act.indices)
#                 IoIt = len(ident_cp) / len(obj_it.indices)
#
#                 max_time = max(IoAct_time, IoIt_time)
#                 max_spatial = max(IoAct, IoIt)
#
#                 # if calculated overlap exceeds defined thresholds and change direction is equal -> store link between objects
#                 if max_time > time_threshold and max_spatial > spatial_threshold and change[idx_act] == change[idx_it]:
#                     merging[idx_act] = merging[idx_act] + [idx_it]
#                     values.append([max_time, max_spatial])
#
#         values = np.array(values)
#         return merging
#
#
#     def connecting_indexes(self, merging):
#         import copy
#         visited = [False] * len(self.objects)
#         merged_idxs = []
#
#         for idx_act, lst in enumerate(merging):
#
#             if visited[idx_act] == True:
#                 continue
#             else:
#                 visited[idx_act] = True
#                 merged_idxs.append([idx_act])
#
#                 to_visit = copy.deepcopy(lst)
#                 while to_visit:
#                     idx_next = to_visit.pop(0)
#                     if visited[idx_next] == True:
#                         continue
#                     else:
#                         visited[idx_next] = True
#                         merged_idxs[-1] = merged_idxs[-1] + [idx_next]
#                         to_visit = to_visit + merging[idx_next]
#
#         return merged_idxs
#
#     def merging(self, merged_idxs):
#         from functools import reduce
#         merged_4dobcs = []
#         for idx in merged_idxs:
#             indices = [self.objects[i].indices for i in idx]
#             start_epochs = [self.objects[i].start_epoch for i in idx]
#             end_epochs = [self.objects[i].end_epoch for i in idx]
#
#             indices_merge = reduce(np.union1d, indices)
#             start_epoch_merge = min(start_epochs)
#             end_epoch_merge = max(end_epochs)
#
#             merged_4dobcs.append(MergedObjectsOfChange(indices_merge, start_epoch_merge, end_epoch_merge, [i for i in idx], analysis, smoothed_distances))
#         return merged_4dobcs
#
#     def run(self, time_threshold=0.7, spatial_threshold=0.1):
#         change = self.change_direction()
#         overlap = self.overlap(change, time_threshold, spatial_threshold)
#         merged_idxs = self.connecting_indexes(overlap)
#         merged_4dobcs = self.merging(merged_idxs)
#         return merged_4dobcs

class MergedObjectsOfChange:
    def __init__(self, indices, start_epoch, end_epoch, obj_4dobc, analysis, smoothed_distances):
        self.analysis = analysis
        self.indices = indices
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.obj_4dobc = obj_4dobc
        self.smoothed_distances = smoothed_distances

    def distance(self, index):
        return np.nanmean(self.smoothed_distances[index, self.start_epoch: self.end_epoch + 1])

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
            self.analysis.distances_for_compute.shape[1],
        )

        # We use the seed's timeseries to set good axis limits
       # seed_ts = self.analysis.distances_for_compute[
        #    self.seed.index, start_epoch:end_epoch
        #]
        #tsax.set_ylim(np.nanmin(seed_ts) * 0.5, np.nanmax(seed_ts) * 1.5)

        # Create a colormap with distance for this object
        cmap = matplotlib.colormaps.get_cmap("viridis")
        maxdist = np.nanmax(distarray)

        # Plot each time series individually
        for index in self.indices:
            tsax.plot(
                self.analysis.distances_for_compute[index, start_epoch:end_epoch],
                linewidth=0.7,
                alpha=0.3,
                color=cmap(self.distance(index) / maxdist),
            )

        # Plot the seed timeseries again, but with a thicker line
        tsax.plot( linewidth=2.0, zorder=10, color="blue")#seed_ts

        # Next, we add a histogram plot with the distance values (using seaborn)
        seaborn.histplot(distarray, ax=histax, kde=True, color="r")

        # Add labels to the histogram plot
        histax.set_title(f"Segment size: {distarray.shape[0]}")
        histax.set_xlabel("DTW distance")

        # Create a 2D view of the segment
        locations = self.analysis.corepoints.cloud[indexarray, 0:2]
        mapax.scatter(locations[:, 0], locations[:, 1], c=distarray)

        # Some global settings of the generated figure
        fig.tight_layout()

        # Maybe save to file
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

class LinearChangeSeeds_rdp(py4dgeo.RegionGrowingAlgorithm):
    def __init__(self, epsilon, max_change_period, data_gap, **kwargs, ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.max_change_period = max_change_period
        self.data_gap = data_gap
        self.seed_subsampling = kwargs.pop("seed_subsampling")

    def find_seedpoints(self, seed_candidates=None):
        # list of generated seeds
        seeds = []

        # list of core point indices to check as seeds
        if self.seed_candidates is None:
            print('Seed Subsampling', self.seed_subsampling)
            # use all corepoints if no selection is specified, considering subsampling
            seed_candidates_curr = range(0, self.analysis.distances_for_compute.shape[0], self.seed_subsampling)
            print(len(seed_candidates_curr))
        else:
            # use the specified corepoint indices
            seed_candidates_curr = self.seed_candidates

        # iterate over all time series to identify linear changes
        print('Iterating over seedpoints')
        for cp_idx in seed_candidates_curr:
            print(f'Seedpoint: {cp_idx}')
            timeseries = self.analysis.distances_for_compute[cp_idx, :]
            timestamps = [t + self.analysis.reference_epoch.timestamp for t in self.analysis.timedeltas]
            time_day = np.array([(t - self.analysis.reference_epoch.timestamp).total_seconds() / (3600 * 24) for t in timestamps])

            # polygon approximation using the Ramer-Douglas-Peucker algorithm
            poly_aprx = rdp.rdp(np.column_stack([time_day, timeseries]), epsilon=self.epsilon, return_mask=True)
            idxs_keypoints = np.where(poly_aprx)[0]
            idxs_keypoints_both = [[idxs_keypoints[i], idxs_keypoints[i + 1]] for i in range(len(idxs_keypoints) - 1)]

            # segment-wise linear regression for each polygon interval
            for idx in idxs_keypoints_both:
                time_day_fit = time_day[idx[0]:idx[-1] + 1]
                timeseries_fit = timeseries[idx[0]:idx[-1] + 1]

                # delete nan values
                idx_nan = np.isnan(timeseries_fit)
                time_day_fit = time_day_fit[~idx_nan]
                timeseries_fit = timeseries_fit[~idx_nan]

                # estimate a straight line by linear regression
                lin_reg = LinearRegression()
                lin_reg.fit(time_day_fit.reshape(-1, 1), timeseries_fit.reshape(-1, 1))
                y_lr = lin_reg.predict(time_day[idx[0]:idx[-1] + 1].reshape(-1, 1)).flatten()

                startp = np.max([idx[0] - 1, 0])
                stopp = np.min([idx[-1] + 1, len(timeseries) - 1])

                # consider minimal change amplitude
                if abs(np.max(y_lr) - np.min(y_lr)) < self.height_threshold:
                    continue

                # consider maximum change period
                elif stopp - startp > self.max_change_period:
                    continue

                # cosider data gap of between 09.09.2021 and 30.09.2021
                #elif stopp >= 828 and startp <= 827:
                elif stopp >= self.data_gap and startp <= self.data_gap-1:
                    continue

                # add current seed to list of seed candidates
                else:
                    curr_seed = RegionGrowingSeed(cp_idx, startp, stopp)
                    seeds.append(curr_seed)

        return seeds

    # sort the seeds according to their change amplitude in descending order
    def seed_sorting_scorefunction(self):
        def magnitude_sort(seed):
            magn = abs(
                self.analysis.distances_for_compute[seed.index, seed.start_epoch] - self.analysis.distances_for_compute[
                    seed.index, seed.end_epoch])
            return magn * (-1)

        return magnitude_sort

class LinearChangeSeeds_dtr(py4dgeo.RegionGrowingAlgorithm):
    def __init__(self, epsilon,max_change_period, data_gap,**kwargs): #seed_subsampling,
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.seed_subsampling = kwargs.pop("seed_subsampling")
        self.max_change_period = max_change_period
        self.data_gap = data_gap

    def find_seedpoints(self, seed_candidates = None):
        from sklearn.tree import DecisionTreeRegressor
        # list of generated seeds
        seeds = []

        # list of core point indices to check as seeds
        if self.seed_candidates is None:
            # use all corepoints if no selection is specified, considering subsampling
            seed_candidates_curr = range(0, self.analysis.distances_for_compute.shape[0], self.seed_subsampling)
            print(f'Number of seed_candidates:{len(seed_candidates_curr)}')
        else:
            # use the specified corepoint indices
            seed_candidates_curr = self.seed_candidates


        # iterate over all time series to identify linear changes
        for cp_idx in seed_candidates_curr:
            print(f'Seedpoint: {cp_idx}')
            timeseries = self.analysis.distances_for_compute[cp_idx, :]
            timestamps = [t+self.analysis.reference_epoch.timestamp for t in self.analysis.timedeltas]
            time_day = np.array([(t-self.analysis.reference_epoch.timestamp).total_seconds()/(3600*24) for t in timestamps])

            # delete nan values and calculate the gradient of each epoch

            idx_nan = np.isnan(timeseries)
            dys = np.gradient(timeseries[~idx_nan], time_day[~idx_nan])


            # Initialisation of the DTR
            # Training with data and prediction of the gradient for all epochs
            rgr = DecisionTreeRegressor(max_depth=10, min_samples_leaf=1, min_samples_split=2, max_leaf_nodes=50, max_features=None)
            rgr.fit(time_day[~idx_nan].reshape(-1, 1), dys.reshape(-1, 1))
            dys_dt = rgr.predict(time_day.reshape(-1, 1)).flatten()

            # group epochs with equal predicted gradient dys_dt into on interval
            ys_sl = np.ones_like(timeseries)
            for dy in np.unique(dys_dt):

                # segment-wise linear regression for each interval
                msk = dys_dt == dy
                msk_nan = msk[~idx_nan]
                lin_reg = LinearRegression()
                lin_reg.fit(time_day[~idx_nan][msk_nan].reshape(-1,1), timeseries[~idx_nan][msk_nan].reshape(-1,1))
                ys_sl[msk] = lin_reg.predict(time_day[msk].reshape(-1,1)).flatten()

                idx = np.where(msk == True)[0]

                startp = np.max([idx[0]-1, 0])
                stopp = np.min([idx[-1]+1, len(timeseries)-1])

                # consider minimal change amplitude
                if abs(np.max(ys_sl[msk]) - np.min(ys_sl[msk])) < self.height_threshold:
                    continue

                # consider maximum change period
                elif stopp - startp > self.max_change_period:
                    continue

                # cosider data gap of between 09.09.2021 and 30.09.2021
                elif stopp >= self.data_gap and startp <= self.data_gap-1:
                    continue

                # add current seed to list of seed candidates
                else:
                    curr_seed = RegionGrowingSeed(cp_idx, startp, stopp)
                    seeds.append(curr_seed)

        return seeds


    # sort the seeds according to their change amplitude in descending order
    def seed_sorting_scorefunction(self):
        def magnitude_sort(seed):
            magn = abs(self.analysis.distances_for_compute[seed.index, seed.start_epoch] - self.analysis.distances_for_compute[seed.index, seed.end_epoch])
            return magn * (-1) # achieve descending order
        return magnitude_sort
#%%
import numpy as np
import pickle
analysis = SpatiotemporalAnalysis('C:/Users/schar/OneDrive/Desktop/Working Student/Tasks/py4dgeo/Results/DTR_new/riverbank_4dobc_test.zip', force = False)

timestamps = [t+analysis.reference_epoch.timestamp for t in analysis.timedeltas]
corepoints = analysis.corepoints.cloud
smoothed_distances = analysis.smoothed_distances
objects = analysis.objects

#%%
merged = analysis.merge(time_threshold=0.7, spatial_threshold=0.1)
merged[0].plot()

# #%%
# analysis.distances[:,62] = 0
# analysis.distances[:,81] = 0
# analysis.distances[:4,:] = 0
# #%%
# analysis.distances[0,:] = 0

#%%
analysis.extract('DTR', 5, 1000, 200, 828)

#%%
analysis.invalidate_results(True, True, True)
