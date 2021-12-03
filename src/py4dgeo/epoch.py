from py4dgeo.util import (
    Py4DGeoError,
    as_single_precision,
    make_contiguous,
)

import laspy
import numpy as np
import py4dgeo._py4dgeo as _py4dgeo


class Epoch(_py4dgeo.Epoch):
    def __init__(self, cloud: np.ndarray):
        # Check the given array shapes
        if len(cloud.shape) != 2 or cloud.shape[1] != 3:
            raise Py4DGeoError("Clouds need to be an array of shape nx3")

        # Make sure that cloud is double precision and contiguous in memory
        cloud = as_single_precision(cloud)
        cloud = make_contiguous(cloud)

        # Call base class constructor
        super().__init__(cloud)

        # Build the KDTree index
        # TODO: When exactly should this be done and how do we want to
        #       the user interface to look like?
        self.kdtree.build_tree(10)


def as_epoch(cloud):
    """Create an epoch from a cloud

    Idempotent operation to create an epoch from a cloud.
    """

    # If this is already an epoch, this is a no-op
    if isinstance(cloud, Epoch):
        return cloud

    # Initialize an epoch from the given cloud
    return Epoch(cloud)


def read_from_xyz(filename):
    """Create an epoch from an xyz file

    :param filename:
        The filename to read from. Each line in the input file is expected
        to contain three space separated numbers.
    :type filename: str
    """
    cloud = np.genfromtxt(filename, dtype=np.float64)
    cloud -= cloud.min(axis=0)

    return Epoch(cloud=cloud.astype("f"))


def read_from_las(filename):
    """Create an epoch from a LAS/LAZ file

    :param filename:
        The filename to read from. It is expected to be in LAS/LAZ format
        and will be processed using laspy.
    :type filename: str
    """

    lasfile = laspy.read(filename)
    lasfile.header.offsets = np.array([0, 0, 0])

    return Epoch(np.vstack((lasfile.x, lasfile.y, lasfile.z)).astype("f").transpose())
