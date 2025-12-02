from py4dgeo.logger import logger_context
from py4dgeo.registration import Transformation
from py4dgeo.util import (
    Py4DGeoError,
    append_file_extension,
    as_double_precision,
    find_file,
    make_contiguous,
    is_iterable,
)
from numpy.lib.recfunctions import append_fields

import dateparser
import datetime
import json
import laspy
import logging
import numpy as np
import os
import tempfile
import typing
import zipfile

import _py4dgeo

logger = logging.getLogger("py4dgeo")

# This integer controls the versioning of the epoch file format. Whenever the
# format is changed, this version should be increased, so that py4dgeo can warn
# about incompatibilities of py4dgeo with loaded data. This version is intentionally
# different from py4dgeo's version, because not all releases of py4dgeo necessarily
# change the epoch file format and we want to be as compatible as possible.
PY4DGEO_EPOCH_FILE_FORMAT_VERSION = 4


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Epoch(_py4dgeo.Epoch):
    def __init__(
        self,
        cloud: np.ndarray,
        normals: np.ndarray = None,
        additional_dimensions: np.ndarray = None,
        timestamp=None,
        scanpos_info: dict = None,
    ):
        """

        :param cloud:
            The point cloud array of shape (n, 3).

        :param normals:
            The point cloud normals of shape (n, 3) where n is the
            same as the number of points in the point cloud.

        :param additional_dimensions:
            A numpy array of additional, per-point data in the point cloud. The
            numpy data type is expected to be a structured dtype, so that the data
            columns are accessible by their name.

        :param timestamp:
            The point cloud timestamp, default is None.

        :param scanpos_info:
            The point scan positions information, default is None..
        """
        # Check the given array shapes
        if len(cloud.shape) != 2 or cloud.shape[1] != 3:
            raise Py4DGeoError("Clouds need to be an array of shape nx3")

        # Make sure that cloud is double precision and contiguous in memory
        cloud = as_double_precision(cloud)
        cloud = make_contiguous(cloud)

        # Set identity transformation
        self._transformations = []

        # Make sure that given normals are DP and contiguous as well
        if normals is not None:
            normals = make_contiguous(as_double_precision(normals))
        self._normals = normals

        # Set metadata properties
        self.timestamp = timestamp
        self.scanpos_info = scanpos_info

        # Set the additional information (e.g. segment ids, normals, etc)
        self.additional_dimensions = additional_dimensions

        # Call base class constructor
        super().__init__(cloud)

    @property
    def cloud(self):
        return self._cloud

    @cloud.setter
    def cloud(self, cloud):
        raise Py4DGeoError(
            "The Epoch point cloud cannot be changed after initialization. Please construct a new Epoch, e.g. by slicing an existing one."
        )

    @property
    def kdtree(self):
        return self._kdtree

    @kdtree.setter
    def kdtree(self, kdtree):
        raise Py4DGeoError(
            "The KDTree of an Epoch cannot be changed after initialization."
        )

    @property
    def octree(self):
        return self._octree

    @octree.setter
    def octree(self, octree):
        raise Py4DGeoError(
            "The Octree of an Epoch cannot be changed after initialization."
        )

    @property
    def normals(self):
        # Maybe calculate normals
        if self._normals is None:
            raise Py4DGeoError(
                "Normals for this Epoch have not been calculated! Please use Epoch.calculate_normals or load externally calculated normals."
            )

        return self._normals

    def calculate_normals(
        self, radius=1.0, orientation_vector: np.ndarray = np.array([0, 0, 1])
    ):
        """Calculate point cloud normals

        :param radius:
            The radius used to determine the neighborhood of a point.

        :param orientation_vector:
            A vector to determine orientation of the normals. It should point "up".
        """

        self._validate_search_tree()

        # Reuse the multiscale code with a single radius in order to
        # avoid code duplication.
        with logger_context("Calculating point cloud normals:"):
            self._normals, _ = _py4dgeo.compute_multiscale_directions(
                self,
                self.cloud,
                [radius],
                orientation_vector,
            )

        return self.normals

    def _validate_search_tree(self):
        """ "Check if the default search tree is built"""

        tree_type = self.get_default_radius_search_tree()

        if tree_type == _py4dgeo.SearchTree.KDTreeSearch:
            if self.kdtree.leaf_parameter() == 0:
                self.build_kdtree()
        else:
            if self.octree.get_number_of_points() == 0:
                self.build_octree()

    def normals_attachment(self, normals_array):
        """Attach normals to the epoch object

        :param normals:
            The point cloud normals of shape (n, 3) where n is the
            same as the number of points in the point cloud.
        """

        if normals_array.shape == self.cloud.shape:
            self._normals = normals_array
        else:
            raise ValueError("Normals cannot be added. Shape does not match.")

    def copy(self):
        """Copy the epoch object"""

        new_epoch = Epoch(
            self.cloud.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            additional_dimensions=(
                self.additional_dimensions.copy()
                if self.additional_dimensions is not None
                else None
            ),
            timestamp=self.timestamp,
            scanpos_info=(
                self.scanpos_info.copy() if self.scanpos_info is not None else None
            ),
        )

        return new_epoch

    def __getitem__(self, ind):
        """Slice the epoch in order to e.g. downsample it.

        Creates a copy of the epoch.
        """

        return Epoch(
            self.cloud[ind],
            normals=self.normals[ind] if self.normals is not None else None,
            additional_dimensions=(
                self.additional_dimensions[ind]
                if self.additional_dimensions is not None
                else None
            ),
            **self.metadata,
        )

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        self._timestamp = normalize_timestamp(timestamp)

    @property
    def scanpos_info(self):
        return self._scanpos_info

    @scanpos_info.setter
    def scanpos_info(self, scanpos_info):
        if isinstance(scanpos_info, list):
            self._scanpos_info = scanpos_info
        elif isinstance(scanpos_info, dict):
            self._scanpos_info = scan_positions_info_from_dict(scanpos_info)
        else:
            self._scanpos_info = None

    @property
    def scanpos_id(self):
        return (
            self.additional_dimensions["scanpos_id"]
            .reshape(self.cloud.shape[0])
            .astype(np.int32)
        )

    @scanpos_id.setter
    def scanpos_id(self, scanpos_id):
        if self.additional_dimensions is None:
            additional_columns = np.empty(
                shape=(self.cloud.shape[0], 1),
                dtype=np.dtype([("scanpos_id", "<i4")]),
            )
            additional_columns["scanpos_id"] = np.array(
                scanpos_id, dtype=np.int32
            ).reshape(-1, 1)
            self.additional_dimensions = additional_columns
        else:
            scanpos_id = np.array(scanpos_id, dtype=np.int32)
            new_additional_dimensions = append_fields(
                self.additional_dimensions, "scanpos_id", scanpos_id, usemask=False
            )

            self.additional_dimensions = new_additional_dimensions

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
            "scanpos_info": None if self.scanpos_info is None else self.scanpos_info,
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

    def build_octree(self):
        """Build the search octree index"""
        if self.octree.get_number_of_points() == 0:
            logger.info(f"Building Octree structure")
            self.octree.build_tree()

    def transform(
        self,
        transformation: typing.Optional[Transformation] = None,
        affine_transformation: typing.Optional[np.ndarray] = None,
        rotation: typing.Optional[np.ndarray] = None,
        translation: typing.Optional[np.ndarray] = None,
        reduction_point: typing.Optional[np.ndarray] = None,
    ):
        """Transform the epoch with an affine transformation

        :param transformation:
            A Transformation object that describes the transformation to apply.
            If this argument is given, the other arguments are ignored.
            This parameter is typically used if the transformation was calculated
            by py4dgeo itself.
        :type transformation: Transformation
        :param affine_transformation:
            A 4x4 or 3x4 matrix representing the affine transformation. Given
            as a numpy array. If this argument is given, the rotation and
            translation arguments are ignored.
        :type transformation: np.ndarray
        :param rotation:
            A 3x3 matrix specifying the rotation to apply
        :type rotation: np.ndarray
        :param translation:
            A vector specifying the translation to apply
        :type translation: np.ndarray
        :param reduction_point:
            A translation vector to apply before applying rotation and scaling.
            This is used to increase the numerical accuracy of transformation.
            If a transformation is given, this argument is ignored.
        :type reduction_point: np.ndarray
        """

        # Extract the affine transformation and reduction point from the given transformation
        if transformation is not None:
            assert isinstance(transformation, Transformation)
            affine_transformation = transformation.affine_transformation
            reduction_point = transformation.reduction_point

        # Build the transformation if it is not explicitly given
        if affine_transformation is None:
            trafo = np.identity(4, dtype=np.float64)
            trafo[:3, :3] = rotation
            trafo[:3, 3] = translation
        else:
            # If it was given, make a copy and potentially resize it
            trafo = affine_transformation.copy()
            if trafo.shape[0] == 3:
                trafo.resize((4, 4), refcheck=False)
                trafo[3, 3] = 1

        if reduction_point is None:
            reduction_point = np.array([0, 0, 0], dtype=np.float64)

        # Ensure contiguous DP memory
        trafo = as_double_precision(make_contiguous(trafo))

        # Invalidate the KDTree
        self.kdtree.invalidate()

        # Invalidate the Octree
        self.octree.invalidate()

        if self._normals is None:
            self._normals = np.empty((1, 3))  # dummy array to avoid error in C++ code
        # Apply the actual transformation as efficient C++
        _py4dgeo.transform_pointcloud_inplace(
            self.cloud, trafo, reduction_point, self._normals
        )

        # Store the transformation
        self._transformations.append(
            Transformation(affine_transformation=trafo, reduction_point=reduction_point)
        )

    @property
    def transformation(self):
        """Access the affine transformations that were applied to this epoch

        In order to set this property please use the transform method instead,
        which will make sure to also apply the transformation.

        :returns:
            Returns a list of applied transformations. These are given
            as a tuple of a 4x4 matrix defining the affine transformation
            and the reduction point used when applying it.
        """
        return self._transformations

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

                # Write the transformation into a file
                trafofile = os.path.join(tmp_dir, "trafo.json")
                with open(trafofile, "w") as f:
                    json.dump(
                        [t.__dict__ for t in self._transformations],
                        f,
                        cls=NumpyArrayEncoder,
                    )
                zf.write(trafofile, arcname="trafo.json")

                # Write the actual point cloud array using laspy - LAZ compression
                # is far better than any compression numpy + zipfile can do.
                cloudfile = os.path.join(tmp_dir, "cloud.laz")
                hdr = laspy.LasHeader(version="1.4", point_format=6)
                hdr.x_scale = 0.00025
                hdr.y_scale = 0.00025
                hdr.z_scale = 0.00025
                mean_extent = np.mean(self.cloud, axis=0)
                hdr.x_offset = int(mean_extent[0])
                hdr.y_offset = int(mean_extent[1])
                hdr.z_offset = int(mean_extent[2])
                lasfile = laspy.LasData(hdr)
                lasfile.x = self.cloud[:, 0]
                lasfile.y = self.cloud[:, 1]
                lasfile.z = self.cloud[:, 2]

                # define dimensions for normals below:
                if self._normals is not None:
                    lasfile.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name="NormalX", type="f8", description="X axis of normals"
                        )
                    )
                    lasfile.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name="NormalY", type="f8", description="Y axis of normals"
                        )
                    )
                    lasfile.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name="NormalZ", type="f8", description="Z axis of normals"
                        )
                    )
                    lasfile.NormalX = self.normals[:, 0]
                    lasfile.NormalY = self.normals[:, 1]
                    lasfile.NormalZ = self.normals[:, 2]
                else:
                    logger.info("Saving a file without normals.")

                lasfile.write(cloudfile)
                zf.write(cloudfile, arcname="cloud.laz")

                kdtreefile = os.path.join(tmp_dir, "kdtree")
                with open(kdtreefile, "w") as f:
                    self.kdtree.save_index(kdtreefile)
                zf.write(kdtreefile, arcname="kdtree")

                octreefile = os.path.join(tmp_dir, "octree")
                with open(octreefile, "w") as f:
                    self.octree.save_index(octreefile)
                zf.write(octreefile, arcname="octree")

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
                if version > PY4DGEO_EPOCH_FILE_FORMAT_VERSION:
                    raise Py4DGeoError(
                        "Epoch file format not known - please update py4dgeo!"
                    )

                # Read the metadata JSON file
                metadatafile = zf.extract("metadata.json", path=tmp_dir)
                with open(metadatafile, "r") as f:
                    metadata = json.load(f)

                # Restore the point cloud itself
                cloudfile = zf.extract("cloud.laz", path=tmp_dir)
                lasfile = laspy.read(cloudfile)
                cloud = np.vstack((lasfile.x, lasfile.y, lasfile.z)).transpose()
                try:
                    normals = np.vstack(
                        (lasfile.NormalX, lasfile.NormalY, lasfile.NormalZ)
                    ).transpose()
                except AttributeError:
                    normals = None
                # Construct the epoch object
                epoch = Epoch(cloud, normals=normals, **metadata)

                # Restore the KDTree object
                kdtreefile = zf.extract("kdtree", path=tmp_dir)
                epoch.kdtree.load_index(kdtreefile)

                # Restore the Octree object if present
                try:
                    octreefile = zf.extract("octree", path=tmp_dir)
                    epoch.octree.load_index(octreefile)
                except KeyError:
                    logger.warning(
                        "No octree found in the archive. Skipping octree loading."
                    )

                # Read the transformation if it exists
                if version >= 3:
                    trafofile = zf.extract("trafo.json", path=tmp_dir)
                    with open(trafofile, "r") as f:
                        trafo = json.load(f)
                    epoch._transformations = [Transformation(**t) for t in trafo]

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


def read_from_xyz(
    *filenames,
    xyz_columns=[0, 1, 2],
    normal_columns=[],
    additional_dimensions={},
    **parse_opts,
):
    """Create an epoch from an xyz file

    :param filename:
        The filename to read from. Each line in the input file is expected
        to contain three space separated numbers.
    :type filename: str
    :param xyz_columns:
        The column indices of X, Y and Z coordinates. Defaults to [0, 1, 2].
    :type xyz_columns: list
    :param normal_columns:
        The column indices of the normal vector components. Leave empty, if
        your data file does not contain normals, otherwise exactly three indices
        for the x, y and z components need to be given.
    :type normal_columns: list
    :param parse_opts:
        Additional options forwarded to numpy.genfromtxt. This can be used
        to e.g. change the delimiter character, remove header_lines or manually
        specify which columns of the input contain the XYZ coordinates.
    :param additional_dimensions:
        A dictionary, mapping column indices to names of additional data dimensions.
        They will be read from the file and are accessible under their names from the
        created Epoch objects.
        Additional column indexes start with 3.
    :type parse_opts: dict
    """

    # Resolve the given path
    filename = find_file(filenames[0])

    # Ensure that usecols is not passed by the user, we need to use this
    if "usecols" in parse_opts:
        raise Py4DGeoError(
            "read_from_xyz cannot be customized by using usecols, please use xyz_columns, normal_columns or additional_dimensions instead!"
        )

    # Read the point cloud
    logger.info(f"Reading point cloud from file '{filename}'")

    try:
        cloud = np.genfromtxt(
            filename, dtype=np.float64, usecols=xyz_columns, **parse_opts
        )
    except ValueError:
        raise Py4DGeoError("Malformed XYZ file")

    # Potentially read normals
    normals = None
    if normal_columns:
        if len(normal_columns) != 3:
            raise Py4DGeoError("normal_columns need to be a list of three integers!")

        try:
            normals = np.genfromtxt(
                filename, dtype=np.float64, usecols=normal_columns, **parse_opts
            )
        except ValueError:
            raise Py4DGeoError("Malformed XYZ file")

    # Potentially read additional_dimensions passed by the user
    additional_columns = np.empty(
        shape=(cloud.shape[0], 1),
        dtype=np.dtype([(name, "<f8") for name in additional_dimensions.values()]),
    )

    add_cols = list(sorted(additional_dimensions.keys()))
    try:
        parsed_additionals = np.genfromtxt(
            filename, dtype=np.float64, usecols=add_cols, **parse_opts
        )
        # Ensure that the parsed array is two-dimensional, even if only
        # one additional dimension was given (avoids an edge case)
        parsed_additionals = parsed_additionals.reshape(-1, 1)
    except ValueError:
        raise Py4DGeoError("Malformed XYZ file")

    for i, col in enumerate(add_cols):
        additional_columns[additional_dimensions[col]] = parsed_additionals[
            :, i
        ].reshape(-1, 1)

    # Finalize the construction of the new epoch
    new_epoch = Epoch(cloud, normals=normals, additional_dimensions=additional_columns)

    if len(filenames) == 1:
        # End recursion and return non-tuple to make the case that the user
        # called this with only one filename more intuitive
        return new_epoch
    else:
        # Go into recursion
        return (new_epoch,) + _as_tuple(
            read_from_xyz(
                *filenames[1:],
                xyz_columns=xyz_columns,
                normal_columns=normal_columns,
                additional_dimensions=additional_dimensions,
                **parse_opts,
            )
        )


def read_from_las(*filenames, normal_columns=[], additional_dimensions={}):
    """Create an epoch from a LAS/LAZ file

    :param filename:
        The filename to read from. It is expected to be in LAS/LAZ format
        and will be processed using laspy.
    :type filename: str
    :param normal_columns:
        The column names of the normal vector components, e.g. "NormalX", "nx", "normal_x" etc., keep in mind that there
        must be exactly 3 columns. Leave empty, if your data file does not contain normals.
    :type normal_columns: list
    :param additional_dimensions:
        A dictionary, mapping names of additional data dimensions in the input
        dataset to additional data dimensions in our epoch data structure.
    :type additional_dimensions: dict
    """

    # Resolve the given path
    filename = find_file(filenames[0])

    # Read the lasfile using laspy
    logger.info(f"Reading point cloud from file '{filename}'")
    lasfile = laspy.read(filename)

    cloud = np.vstack(
        (
            lasfile.x,
            lasfile.y,
            lasfile.z,
        )
    ).transpose()

    normals = None
    if normal_columns:
        if len(normal_columns) != 3:
            raise Py4DGeoError("normal_columns need to be a list of three strings!")

        normals = np.vstack(
            [
                lasfile.points[normal_columns[0]],
                lasfile.points[normal_columns[1]],
                lasfile.points[normal_columns[2]],
            ]
        ).transpose()

    # set scan positions
    # build additional_dimensions dtype structure
    additional_columns = np.empty(
        shape=(cloud.shape[0], 1),
        dtype=np.dtype([(name, "<f8") for name in additional_dimensions.values()]),
    )
    for column_id, column_name in additional_dimensions.items():
        additional_columns[column_name] = np.array(
            lasfile.points[column_id], dtype=np.int32
        ).reshape(-1, 1)

    # Construct Epoch and go into recursion
    new_epoch = Epoch(
        cloud,
        normals=normals,
        timestamp=lasfile.header.creation_date,
        additional_dimensions=additional_columns,
    )

    if len(filenames) == 1:
        # End recursion and return non-tuple to make the case that the user
        # called this with only one filename more intuitive
        return new_epoch
    else:
        # Go into recursion
        return (new_epoch,) + _as_tuple(
            read_from_las(
                *filenames[1:],
                normal_columns=normal_columns,
                additional_dimensions=additional_dimensions,
            )
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


def scan_positions_info_from_dict(info_dict: dict):
    if info_dict is None:
        return None
    if not isinstance(info_dict, dict):
        raise Py4DGeoError(f"The input scan position information should be dictionary.")
        return None
    # Compatible with both integer key and string key as index of the scan positions in json file
    # load scan positions from dictionary, standardize loading via json format dumps to string key
    scanpos_dict_load = json.loads(json.dumps(info_dict))
    sps_list = []
    for i in range(1, 1 + len(scanpos_dict_load)):
        sps_list.append(scanpos_dict_load[str(i)])

    for sp in sps_list:
        sp_check = True
        sp_check = False if len(sp["origin"]) != 3 else sp_check
        sp_check = False if not isinstance(sp["sigma_range"], float) else sp_check
        sp_check = False if not isinstance(sp["sigma_scan"], float) else sp_check
        sp_check = False if not isinstance(sp["sigma_yaw"], float) else sp_check
        if not sp_check:
            raise Py4DGeoError("Scan positions load failed, please check format. ")
    return sps_list
