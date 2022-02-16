from py4dgeo.util import (
    Py4DGeoError,
    append_file_extension,
    as_single_precision,
    make_contiguous,
)

import json
import laspy
import numpy as np
import os
import pickle
import tempfile
import zipfile

import py4dgeo._py4dgeo as _py4dgeo


# This integer controls the versioning of the epoch file format. Whenever the
# format is changed, this version should be increased, so that py4dgeo can warn
# about incompatibilities of py4dgeo with loaded data. This version is intentionally
# different from py4dgeo's version, because not all releases of py4dgeo necessarily
# change the epoch file format and we want to be as compatible as possible.
PY4DGEO_EPOCH_FILE_FORMAT_VERSION = 0


class Epoch(_py4dgeo.Epoch):
    def __init__(self, cloud: np.ndarray, geographic_offset=None):
        """

        :param cloud:
            The point cloud array of shape (n, 3).
        :param geographic_offset:
            The offset that needs to be applied to transform the given points
            into actual geographic coordinates.
        """
        # Check the given array shapes
        if len(cloud.shape) != 2 or cloud.shape[1] != 3:
            raise Py4DGeoError("Clouds need to be an array of shape nx3")

        # Make sure that cloud is single precision and contiguous in memory
        cloud = as_single_precision(cloud)
        cloud = make_contiguous(cloud)

        # Apply defaults to metadata
        if geographic_offset is None:
            geographic_offset = np.array([0, 0, 0], dtype=np.float32)
        self.geographic_offset = np.asarray(geographic_offset)

        # Call base class constructor
        super().__init__(cloud)

    @property
    def metadata(self):
        """Provide metadata of this epoch."""
        return {"geographic_offset": tuple(self.geographic_offset)}

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
            self.kdtree.build_tree(leaf_size)

    def save(self, filename):
        """Save this epoch to a file

        :param filename:
            The filename to save the epoch in.
        :type filename: str
        """

        # Ensure that we have a file extension
        filename = append_file_extension(filename, "zip")

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
                cloud = (
                    np.vstack((lasfile.x, lasfile.y, lasfile.z)).astype("f").transpose()
                )

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
        for k, v in metadata:
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
    return Epoch(cloud)


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)


def read_from_xyz(*filenames, other_epoch=None, delimiter=" ", header_lines=0):
    """Create an epoch from an xyz file

    :param filename:
        The filename to read from. Each line in the input file is expected
        to contain three space separated numbers.
    :type filename: str
    :param other_epoch:
        An existing epoch that we want to be compatible with.
    :type other_epoch: py4dgeo.Epoch
    :param delimiter:
        The delimiter used between x, y and z coordinates in the file (defaults to a space)
    :type delimited: str
    :param header_lines:
        The number of header lines in the XYZ files. These will be skipped
        and ignored when reading the file.
    :type header_lines: int
    """

    # Read the first cloud
    try:
        cloud = np.genfromtxt(
            filenames[0],
            dtype=np.float64,
            delimiter=delimiter,
            skip_header=header_lines,
        )
    except ValueError:
        raise Py4DGeoError(
            "Malformed XYZ file - all rows are expected to have exactly three columns"
        )

    # Determine the offset to use. If no epoch to be compatible with has been
    # given, we calculate one. Otherwise, we take the same offset to be
    # compatible.s
    if other_epoch is None:
        offset = cloud.mean(axis=0)
    else:
        offset = other_epoch.geographic_offset

    # Apply chosen offset
    cloud -= offset

    # Construct the new Epoch object
    new_epoch = Epoch(cloud=cloud.astype("f"), geographic_offset=offset)

    if len(filenames) == 1:
        # End recursion and return non-tuple to make the case that the user
        # called this with only one filename more intuitive
        return new_epoch
    else:
        # Go into recursion
        return (new_epoch,) + _as_tuple(
            read_from_xyz(*filenames[1:], other_epoch=new_epoch)
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

    # Read the lasfile using laspy
    lasfile = laspy.read(filenames[0])

    # Determine the offset to use. If no epoch to be compatible with has been
    # given, we calculate one. Otherwise, we take the same offset to be
    # compatible.s
    if other_epoch is None:
        geographic_offset = lasfile.header.mins
    else:
        geographic_offset = other_epoch.geographic_offset

    # Construct Epoch and go into recursion
    new_epoch = Epoch(
        np.vstack(
            (
                lasfile.x - geographic_offset[0],
                lasfile.y - geographic_offset[1],
                lasfile.z - geographic_offset[2],
            )
        )
        .astype("f")
        .transpose(),
        geographic_offset=geographic_offset,
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
