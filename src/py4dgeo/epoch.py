from py4dgeo.util import (
    Py4DGeoError,
    append_file_extension,
    as_double_precision,
    find_file,
    make_contiguous,
    is_iterable,
)

import dateparser
import datetime
import json
import laspy
import logging
import numpy as np
import os
import tempfile
import zipfile

import py4dgeo._py4dgeo as _py4dgeo


logger = logging.getLogger("py4dgeo")


# This integer controls the versioning of the epoch file format. Whenever the
# format is changed, this version should be increased, so that py4dgeo can warn
# about incompatibilities of py4dgeo with loaded data. This version is intentionally
# different from py4dgeo's version, because not all releases of py4dgeo necessarily
# change the epoch file format and we want to be as compatible as possible.
PY4DGEO_EPOCH_FILE_FORMAT_VERSION = 2


class Epoch(_py4dgeo.Epoch):
    def __init__(self, cloud: np.ndarray, timestamp=None):
        """

        :param cloud:
            The point cloud array of shape (n, 3).
        """
        # Check the given array shapes
        if len(cloud.shape) != 2 or cloud.shape[1] != 3:
            raise Py4DGeoError("Clouds need to be an array of shape nx3")

        # Make sure that cloud is double precision and contiguous in memory
        cloud = as_double_precision(cloud)
        cloud = make_contiguous(cloud)

        # Set metadata properties
        self.timestamp = timestamp

        # Call base class constructor
        super().__init__(cloud)

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        self._timestamp = normalize_timestamp(timestamp)

    @property
    def metadata(self):
        """Provide the metadata of this epoch as a Python dictionary

        The return value of this property only makes use of Python built-in
        data structures such that it can e.g. be serialized using the JSON
        module. Also, the returned values are understood by :ref:`Epoch.__init__`
        such that you can do :code:`Epoch(cloud, **other.metadata)`.
        """

        return {
            "timestamp": None if self.timestamp is None else str(self.timestamp),
        }

    def build_kdtree(self, leaf_size=10, force_rebuild=False):
        """Build the search tree index

        :param leaf_size:
            An internal optimization parameter of the search tree data structure.
            The algorithm uses a bruteforce search on subtrees of size below the
            given threshold. Increasing this value speeds up search tree build time,
            but slows down query times.
        :type leaf_size: int
        :param force_rebuild:
            Rebuild the search tree even if it was already built before.
        :type force_rebuild: bool
        """
        if self.kdtree.leaf_parameter() == 0 or force_rebuild:
            logger.info(f"Building KDTree structure with leaf parameter {leaf_size}")
            self.kdtree.build_tree(leaf_size)

    def save(self, filename):
        """Save this epoch to a file

        :param filename:
            The filename to save the epoch in.
        :type filename: str
        """

        # Ensure that we have a file extension
        filename = append_file_extension(filename, "zip")
        logger.info(f"Saving epoch to file '{filename}'")

        # Use a temporary directory when creating files
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create the final archive
            with zipfile.ZipFile(
                filename, mode="w", compression=zipfile.ZIP_BZIP2
            ) as zf:

                # Write the epoch file format version number
                zf.writestr("EPOCH_FILE_FORMAT", str(PY4DGEO_EPOCH_FILE_FORMAT_VERSION))

                # Write the metadata dictionary into a json file
                metadatafile = os.path.join(tmp_dir, "metadata.json")
                with open(metadatafile, "w") as f:
                    json.dump(self.metadata, f)
                zf.write(metadatafile, arcname="metadata.json")

                # Write the actual point cloud array using laspy - LAZ compression
                # is far better than any compression numpy + zipfile can do.
                cloudfile = os.path.join(tmp_dir, "cloud.laz")
                header = laspy.LasHeader(version="1.4", point_format=6)
                lasfile = laspy.LasData(header)
                lasfile.x = self.cloud[:, 0]
                lasfile.y = self.cloud[:, 1]
                lasfile.z = self.cloud[:, 2]
                lasfile.write(cloudfile)
                zf.write(cloudfile, arcname="cloud.laz")

                kdtreefile = os.path.join(tmp_dir, "kdtree")
                with open(kdtreefile, "w") as f:
                    self.kdtree.save_index(kdtreefile)
                zf.write(kdtreefile, arcname="kdtree")

    @staticmethod
    def load(filename):
        """Construct an Epoch instance by loading it from a file

        :param filename:
            The filename to load the epoch from.
        :type filename: str
        """

        # Ensure that we have a file extension
        filename = append_file_extension(filename, "zip")
        logger.info(f"Restoring epoch from file '{filename}'")

        # Use temporary directory for extraction of files
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Open the ZIP archive
            with zipfile.ZipFile(filename, mode="r") as zf:

                # Read the epoch file version number and compare to current
                version = int(zf.read("EPOCH_FILE_FORMAT").decode())
                if version != PY4DGEO_EPOCH_FILE_FORMAT_VERSION:
                    raise Py4DGeoError("Epoch file format is out of date!")

                # Read the metadata JSON file
                metadatafile = zf.extract("metadata.json", path=tmp_dir)
                with open(metadatafile, "r") as f:
                    metadata = json.load(f)

                # Restore the point cloud itself
                cloudfile = zf.extract("cloud.laz", path=tmp_dir)
                lasfile = laspy.read(cloudfile)
                cloud = np.vstack((lasfile.x, lasfile.y, lasfile.z)).transpose()

                # Construct the epoch object
                epoch = Epoch(cloud, **metadata)

                # Restore the KDTree object
                kdtreefile = zf.extract("kdtree", path=tmp_dir)
                epoch.kdtree.load_index(kdtreefile)

        return epoch

    def __getstate__(self):
        return (
            PY4DGEO_EPOCH_FILE_FORMAT_VERSION,
            self.metadata,
            _py4dgeo.Epoch.__getstate__(self),
        )

    def __setstate__(self, state):
        v, metadata, base = state

        if v != PY4DGEO_EPOCH_FILE_FORMAT_VERSION:
            raise Py4DGeoError("Epoch file format is out of date!")

        # Restore metadata
        for k, v in metadata.items():
            setattr(self, k, v)

        # Set the base class object
        _py4dgeo.Epoch.__setstate__(self, base)


def save_epoch(epoch, filename):
    """Save an epoch to a given filename

    :param epoch:
        The epoch that should be saved.
    :type epoch: Epoch
    :param filename:
        The filename where to save the epoch
    :type filename: str
    """
    return epoch.save(filename)


def load_epoch(filename):
    """Load an epoch from a given filename

    :param filename:
        The filename to load the epoch from.
    :type filename: str
    """
    return Epoch.load(filename)


def as_epoch(cloud):
    """Create an epoch from a cloud

    Idempotent operation to create an epoch from a cloud.
    """

    # If this is already an epoch, this is a no-op
    if isinstance(cloud, Epoch):
        return cloud

    # Initialize an epoch from the given cloud
    logger.info("Initializing Epoch object from given point cloud")
    return Epoch(cloud)


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)


def read_from_xyz(*filenames, other_epoch=None, **parse_opts):
    """Create an epoch from an xyz file

    :param filename:
        The filename to read from. Each line in the input file is expected
        to contain three space separated numbers.
    :type filename: str
    :param other_epoch:
        An existing epoch that we want to be compatible with.
    :type other_epoch: py4dgeo.Epoch
    :param parse_opts:
        Additional options forwarded to numpy.genfromtxt. This can be used
        to e.g. change the delimiter character, remove header_lines or manually
        specify which columns of the input contain the XYZ coordinates.
    :type parse_opts: dict
    """

    # Resolve the given path
    filename = find_file(filenames[0])

    # Read the first cloud
    try:
        logger.info(f"Reading point cloud from file '{filename}'")
        cloud = np.genfromtxt(filename, dtype=np.float64, **parse_opts)
    except ValueError:
        raise Py4DGeoError(
            "Malformed XYZ file - all rows are expected to have exactly three columns"
        )

    # Construct the new Epoch object
    new_epoch = Epoch(cloud=cloud)

    if len(filenames) == 1:
        # End recursion and return non-tuple to make the case that the user
        # called this with only one filename more intuitive
        return new_epoch
    else:
        # Go into recursion
        return (new_epoch,) + _as_tuple(
            read_from_xyz(*filenames[1:], other_epoch=new_epoch, **parse_opts)
        )


def read_from_las(*filenames, other_epoch=None):
    """Create an epoch from a LAS/LAZ file

    :param filename:
        The filename to read from. It is expected to be in LAS/LAZ format
        and will be processed using laspy.
    :type filename: str
    :param other_epoch:
        An existing epoch that we want to be compatible with.
    :type other_epoch: py4dgeo.Epoch
    """

    # Resolve the given path
    filename = find_file(filenames[0])

    # Read the lasfile using laspy
    logger.info(f"Reading point cloud from file '{filename}'")
    lasfile = laspy.read(filename)

    # Construct Epoch and go into recursion
    new_epoch = Epoch(
        np.vstack(
            (
                lasfile.x,
                lasfile.y,
                lasfile.z,
            )
        ).transpose(),
        timestamp=lasfile.header.creation_date,
    )

    if len(filenames) == 1:
        # End recursion and return non-tuple to make the case that the user
        # called this with only one filename more intuitive
        return new_epoch
    else:
        # Go into recursion
        return (new_epoch,) + _as_tuple(
            read_from_las(*filenames[1:], other_epoch=new_epoch)
        )


def normalize_timestamp(timestamp):
    """Bring a given timestamp into a standardized Python format"""

    # This might be normalized already or non-existing
    if isinstance(timestamp, datetime.datetime) or timestamp is None:
        return timestamp

    # This might be a date without time information e.g. from laspy
    if isinstance(timestamp, datetime.date):
        return datetime.datetime(timestamp.year, timestamp.month, timestamp.day)

    # If this is a tuple of (year, day of year) as used in the LAS
    # file header, we convert it.
    if is_iterable(timestamp):
        if len(timestamp) == 2:
            return datetime.datetime(timestamp[0], 1, 1) + datetime.timedelta(
                timestamp[1] - 1
            )

    # If this is a string we use the dateparser library that understands
    # all sorts of human-readable timestamps
    if isinstance(timestamp, str):
        parsed = dateparser.parse(timestamp)

        # dateparser returns None for anything it does not understand
        if parsed is not None:
            return parsed

    raise Py4DGeoError(f"The timestamp '{timestamp}' was not understood by py4dgeo.")
