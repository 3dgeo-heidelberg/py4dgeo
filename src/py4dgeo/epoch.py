from py4dgeo.util import (
    Py4DGeoError,
    as_single_precision,
    make_contiguous,
)

import laspy
import numpy as np
import pickle
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
        self.geographic_offset = geographic_offset

        # Call base class constructor
        super().__init__(cloud)

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
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Construct an Epoch instance by loading it from a file

        :param filename:
            The filename to load the epoch from.
        :type filename: str
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def __getstate__(self):
        return (
            PY4DGEO_EPOCH_FILE_FORMAT_VERSION,
            self.__dict__,
            _py4dgeo.Epoch.__getstate__(self),
        )

    def __setstate__(self, state):
        v, metadata, base = state

        if v != PY4DGEO_EPOCH_FILE_FORMAT_VERSION:
            raise Py4DGeoError("Epoch file format is out of date!")

        # Restore metadata and base class object
        self.__dict__ = metadata
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


def read_from_xyz(*filenames, other_epoch=None):
    """Create an epoch from an xyz file

    :param filename:
        The filename to read from. Each line in the input file is expected
        to contain three space separated numbers.
    :type filename: str
    :param other_epoch:
        An existing epoch that we want to be compatible with.
    :type other_epoch: py4dgeo.Epoch
    """

    # Read the first cloud
    cloud = np.genfromtxt(filenames[0], dtype=np.float64)

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
        return (new_epoch,) + _as_tuple(read_from_xyz(*filenames[1:], other_epoch=new_epoch))


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
        return (new_epoch,) + _as_tuple(read_from_las(*filenames[1:], other_epoch=new_epoch))
