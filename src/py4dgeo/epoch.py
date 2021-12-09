from py4dgeo.util import (
    Py4DGeoError,
    as_single_precision,
    make_contiguous,
)

import laspy
import numpy as np
import py4dgeo._py4dgeo as _py4dgeo


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

    # End recursion
    if len(filenames) == 0:
        return ()

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

    # Construct Epoch and go into recursion
    new_epoch = Epoch(cloud=cloud.astype("f"), geographic_offset=offset)
    return (new_epoch,) + read_from_xyz(*filenames[1:], other_epoch=new_epoch)


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

    # End recursion
    if len(filenames) == 0:
        return ()

    # Read the lasfile using laspy
    lasfile = laspy.read(filenames[0])

    # Determine the offset to use. If no epoch to be compatible with has been
    # given, we calculate one. Otherwise, we take the same offset to be
    # compatible.s    
    if other_epoch is None:
        geographic_offset = lasfile.header.offsets
        lasfile.header.offsets = np.array([0, 0, 0])
    else:
        geographic_offset = other_epoch.geographic_offset
        lasfile.header.offsets = lasfile.header.offsets - other_epoch.geographic_offset

    # Construct Epoch and go into recursion
    new_epoch = Epoch(
        np.vstack((lasfile.x, lasfile.y, lasfile.z)).astype("f").transpose(),
        geographic_offset=geographic_offset,
    )

    return (new_epoch,) + read_from_las(*filenames[1:], other_epoch=new_epoch)
