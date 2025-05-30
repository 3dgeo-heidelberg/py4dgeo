import colorsys
import random
import typing
import pprint
import logging

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector

from sklearn.ensemble import RandomForestClassifier
from sklearn import set_config

set_config(display="diagram")

from abc import ABC, abstractmethod

from py4dgeo import Epoch
from py4dgeo.util import Py4DGeoError, find_file

try:
    from vedo import *

    interactive_available = True
except ImportError:
    interactive_available = False

__all__ = [
    "Viewer",
    "BaseTransformer",
    "PerPointComputation",
    "Segmentation",
    "ExtractSegments",
    "BuilderExtended_y",
    "BuilderExtended_y_Visually",
    "ClassifierWrapper",
    "PBM3C2",
    "build_input_scenario2_without_normals",
    "build_input_scenario2_with_normals",
    "PBM3C2WithSegments",
    "set_interactive_backend",
    "generate_random_extended_y",
    "LLSV_PCA_COLUMNS",
    "SEGMENTED_POINT_CLOUD_COLUMNS",
    "SEGMENT_COLUMNS",
    "generate_extended_y_from_prior_knowledge",
    "generate_possible_region_pairs",
    "DEFAULT_NO_SEGMENT",
    "DEFAULT_STD_DEVIATION_OF_NO_CORE_POINT",
    "add_no_corresponding_seg",
    "config_epoch0_as_segments",
]

logger = logging.getLogger("py4dgeo")

pp = pprint.PrettyPrinter(depth=4)

config_epoch0_as_segments = {
    "get_pipeline_options": True,
    "epoch0_Transform_PerPointComputation__skip": True,
    "epoch0_Transform_Segmentation__skip": True,
    "epoch0_Transform_Second_Segmentation__skip": True,
    "epoch0_Transform_ExtractSegments__skip": True,
}


class LLSV_PCA_COLUMNS:
    X_COLUMN = 0
    Y_COLUMN = 1
    Z_COLUMN = 2
    EPOCH_ID_COLUMN = 3
    EIGENVALUE0_COLUMN = 4
    EIGENVALUE1_COLUMN = 5
    EIGENVALUE2_COLUMN = 6
    EIGENVECTOR_0_X_COLUMN = 7
    EIGENVECTOR_0_Y_COLUMN = 8
    EIGENVECTOR_0_Z_COLUMN = 9
    EIGENVECTOR_1_X_COLUMN = 10
    EIGENVECTOR_1_Y_COLUMN = 11
    EIGENVECTOR_1_Z_COLUMN = 12
    EIGENVECTOR_2_X_COLUMN = 13
    EIGENVECTOR_2_Y_COLUMN = 14
    EIGENVECTOR_2_Z_COLUMN = 15
    LLSV_COLUMN = 16
    NUMBER_OF_COLUMNS = 17


class SEGMENTED_POINT_CLOUD_COLUMNS:
    X_COLUMN = 0
    Y_COLUMN = 1
    Z_COLUMN = 2
    EPOCH_ID_COLUMN = 3
    EIGENVALUE0_COLUMN = 4
    EIGENVALUE1_COLUMN = 5
    EIGENVALUE2_COLUMN = 6
    EIGENVECTOR_0_X_COLUMN = 7
    EIGENVECTOR_0_Y_COLUMN = 8
    EIGENVECTOR_0_Z_COLUMN = 9
    EIGENVECTOR_1_X_COLUMN = 10
    EIGENVECTOR_1_Y_COLUMN = 11
    EIGENVECTOR_1_Z_COLUMN = 12
    EIGENVECTOR_2_X_COLUMN = 13
    EIGENVECTOR_2_Y_COLUMN = 14
    EIGENVECTOR_2_Z_COLUMN = 15
    LLSV_COLUMN = 16
    SEGMENT_ID_COLUMN = 17
    STANDARD_DEVIATION_COLUMN = 18
    NUMBER_OF_COLUMNS = 19


class SEGMENT_COLUMNS:
    X_COLUMN = 0
    Y_COLUMN = 1
    Z_COLUMN = 2
    EPOCH_ID_COLUMN = 3
    EIGENVALUE0_COLUMN = 4
    EIGENVALUE1_COLUMN = 5
    EIGENVALUE2_COLUMN = 6
    EIGENVECTOR_0_X_COLUMN = 7
    EIGENVECTOR_0_Y_COLUMN = 8
    EIGENVECTOR_0_Z_COLUMN = 9
    EIGENVECTOR_1_X_COLUMN = 10
    EIGENVECTOR_1_Y_COLUMN = 11
    EIGENVECTOR_1_Z_COLUMN = 12
    EIGENVECTOR_2_X_COLUMN = 13
    EIGENVECTOR_2_Y_COLUMN = 14
    EIGENVECTOR_2_Z_COLUMN = 15
    LLSV_COLUMN = 16
    SEGMENT_ID_COLUMN = 17
    STANDARD_DEVIATION_COLUMN = 18
    NR_POINTS_PER_SEG_COLUMN = 19
    NUMBER_OF_COLUMNS = 20


# default value, for the points that are part of NO segment ( e.g. -1 )
DEFAULT_NO_SEGMENT = -1

# default standard deviation value for points that are not "core points"
DEFAULT_STD_DEVIATION_OF_NO_CORE_POINT = -1


def set_interactive_backend(backend="vtk"):
    """Set the interactive backend for selection of correspondent segments.

    All backends that can be used with the vedo library can be given here.
    E.g. the following backends are available: vtk, ipyvtk, k3d, 2d, ipygany, panel, itk
    """
    if interactive_available:
        from vedo import settings

        settings.default_backend = backend


def _extract_from_additional_dimensions(
    epoch: Epoch,
    column_names: typing.List[str],
    required_number_of_columns: typing.List[int] = [],
):
    """
    Build a numpy array using 'column_names' which are part of the 'additional_dimensions' field.
    The result will maintain the same order or the columns found in 'column_names'.

    :param epoch:
        Epoch class.
    :param column_names:
        list[ str ]
    :param required_number_of_columns:
        list[ int ]
        default [] , any number of parameter found gets accepted.
        The number of columns required to consider the output valid.
    :return
        numpy array
    """

    result = np.empty(shape=(epoch.cloud.shape[0], 0), dtype=float)

    for column in column_names:
        if column in epoch.additional_dimensions.dtype.names:
            result = np.concatenate(
                (result, epoch.additional_dimensions[column]), axis=1
            )
        else:
            logger.debug(
                f"Column '{column}' not found during _extract_from_additional_dimensions()"
            )

    assert (
        required_number_of_columns == []
        or result.shape[1] in required_number_of_columns
    ), "The number of column found is not a valid one."

    return result


def angle_difference_compute(normal1, normal2):
    """

    :param normal1:
        unit vector
    :param normal2:
        unit vector
    :return:
        numpy array of angles in degrees.
    """

    # normal1, normal2 have to be unit vectors ( and that is the case as a result of the SVD process )
    return np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0)) * 180.0 / np.pi


def geodesic_distance(v1, v2):
    """
    Compute the shortest angular distance between 2 unit vectors.

    :param v1:
        unit vector
    :param v2:
        unit vector
    :return:
        numpy array of angles in degrees.
    """

    return min(
        np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180.0 / np.pi,
        np.arccos(np.clip(np.dot(v1, -v2), -1.0, 1.0)) * 180.0 / np.pi,
    )


class Viewer:
    def __init__(self):
        self.sets = []
        self.plt = Plotter(axes=3)

    @staticmethod
    def HSVToRGB(h, s, v):
        """
        Convert from HSV ( Hue Saturation Value ) color to RGB ( Red Blue Green )

        :param h:
            float [0, 1]
        :param s:
            float [0, 1]
        :param v:
            float [0, 1]
        :return:
            tuple [ float, float, float ]
        """
        return colorsys.hsv_to_rgb(h, s, v)

    @staticmethod
    def get_distinct_colors(n):
        """
        Return a python list of 'n' distinct colors.

        :param n:
            number of colors.
        :return:
            python list of tuple [ float, float, float ]
        """

        huePartition = 1.0 / (n + 1)
        return [
            Viewer.HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)
        ]

    @staticmethod
    def read_np_ndarray_from_xyz(input_file_name: str) -> np.ndarray:
        """
        The reconstructed np.ndarray.
        :param input_file_name:
            The output file name.
        :return:
            np.ndarray
        """

        # Resolve the given path
        filename = find_file(input_file_name)

        # Read it
        try:
            logger.info(f"Reading np.ndarray from file '{filename}'")
            np_ndarray = np.genfromtxt(filename, delimiter=",")
        except ValueError:
            raise Py4DGeoError("Malformed file: " + str(filename))

        return np_ndarray

    @staticmethod
    def segmented_point_cloud_visualizer(
        X: np.ndarray, columns=SEGMENTED_POINT_CLOUD_COLUMNS
    ):
        """
        Visualize a segmented point cloud. ( the resulting point cloud after the segmentation process )

        :param X:
            numpy array (n_points, 19) with the following column structure:
            [
                x, y, z ( 3 columns ),
                EpochID ( 1 column ),
                Eigenvalues( 3 columns ), -> that correspond to the next 3 Eigenvectors
                Eigenvectors( 3 columns ) X 3 -> in descending order using vector norm 2,
                Lowest local surface variation ( 1 column ),
                Segment_ID ( 1 column ),
                Standard deviation ( 1 column )
            ]
        :return:
        """

        X_Y_Z_Columns = [columns.X_COLUMN, columns.Y_COLUMN, columns.Z_COLUMN]

        viewer = Viewer()

        nr_segments = int(X[:, columns.SEGMENT_ID_COLUMN].max())
        colors = Viewer.get_distinct_colors(nr_segments + 1)

        for i in range(0, nr_segments + 1):
            mask = X[:, columns.SEGMENT_ID_COLUMN] == float(i)
            # x,y,z
            set_cloud = X[mask, :][:, X_Y_Z_Columns]

            viewer.sets = viewer.sets + [Points(set_cloud, colors[i], alpha=1, r=10)]

        viewer.plt.show(viewer.sets).close()

    @staticmethod
    def segments_visualizer(X: np.ndarray, columns=SEGMENT_COLUMNS):
        """
        Segments visualizer.

        :param X:
            Each row is a segment, numpy array (1, 20) with the following column structure:
                [
                    x, y, z ( 3 columns ), -> segment, core point
                    EpochID ( 1 column ),
                    Eigenvalues( 3 columns ), -> that correspond to the next 3 Eigenvectors
                    Eigenvectors( 3 columns ) X 3 -> in descending order using vector norm 2,
                    Lowest local surface variation ( 1 column ),
                    Segment_ID ( 1 column ),
                    Standard deviation ( 1 column ),
                    Number of points found in Segment_ID segment ( 1 column )
                ]
        :param columns:
        """

        viewer = Viewer()

        nr_segments = X.shape[0]
        colors = [(1, 0, 0), (0, 1, 0)]

        viewer.plt.add_callback("KeyPress", viewer.toggle_transparency)

        for i in range(0, nr_segments):
            if X[i, columns.EPOCH_ID_COLUMN] == 0:
                color = colors[0]
            else:
                color = colors[1]

            ellipsoid = Ellipsoid(
                pos=(X[i, 0], X[i, 1], X[i, 2]),
                axis1=[
                    X[i, columns.EIGENVECTOR_0_X_COLUMN]
                    * X[i, columns.EIGENVALUE0_COLUMN]
                    * 0.5,
                    X[i, columns.EIGENVECTOR_0_Y_COLUMN]
                    * X[i, columns.EIGENVALUE0_COLUMN]
                    * 0.5,
                    X[i, columns.EIGENVECTOR_0_Z_COLUMN]
                    * X[i, columns.EIGENVALUE0_COLUMN]
                    * 0.3,
                ],
                axis2=[
                    X[i, columns.EIGENVECTOR_1_X_COLUMN]
                    * X[i, columns.EIGENVALUE1_COLUMN]
                    * 0.5,
                    X[i, columns.EIGENVECTOR_1_Y_COLUMN]
                    * X[i, columns.EIGENVALUE1_COLUMN]
                    * 0.5,
                    X[i, columns.EIGENVECTOR_1_Z_COLUMN]
                    * X[i, columns.EIGENVALUE1_COLUMN]
                    * 0.5,
                ],
                axis3=[
                    X[i, columns.EIGENVECTOR_2_X_COLUMN] * 0.1,
                    X[i, columns.EIGENVECTOR_2_Y_COLUMN] * 0.1,
                    X[i, columns.EIGENVECTOR_2_Z_COLUMN] * 0.1,
                ],
                res=24,
                c=color,
                alpha=1,
            )
            # ellipsoid.caption(txt=str(i), size=(0.1, 0.05))
            ellipsoid.id = X[i, columns.SEGMENT_ID_COLUMN]
            ellipsoid.epoch = X[i, columns.EPOCH_ID_COLUMN]
            ellipsoid.isOn = True

            viewer.sets = viewer.sets + [ellipsoid]

        viewer.plt.show(
            viewer.sets,
            Text2D(
                "'z' - toggle transparency on/off "
                "'g' - toggle on/off red ellipsoids "
                "'d' - toggle on/off green ellipsoids",
                pos="top-left",
                bg="k",
                s=0.7,
            ),
        ).close()

    def toggle_transparency(self, evt):
        if evt.keyPressed == "z":
            logger.info("transparency toggle")
            for segment in self.sets:
                if segment.alpha() < 1.0:
                    segment.alpha(1)
                else:
                    segment.alpha(0.5)
            self.plt.render()

        if evt.keyPressed == "g":
            logger.info("toggle red")
            for segment in self.sets:
                if segment.epoch == 0:
                    if segment.isOn:
                        segment.off()
                    else:
                        segment.on()
                    segment.isOn = not segment.isOn
            self.plt.render()

        if evt.keyPressed == "d":
            logger.info("toggle green")
            for segment in self.sets:
                if segment.epoch == 1:
                    if segment.isOn:
                        segment.off()
                    else:
                        segment.on()
                    segment.isOn = not segment.isOn
            self.plt.render()


def generate_random_extended_y(
    X,
    extended_y_file_name="locally_generated_extended_y.csv",
    ratio=1 / 3,
    low=0,
    high=1,
    columns=SEGMENT_COLUMNS,
):
    """
        Generate a subset (1/3 from the total possible pairs) of random tuples of segments ID
        (where each ID gets to be extracted from a different epoch) which are randomly labeled
        between low and high)

    :param X:
        (n_segments, 20)
        Each row contains a segment.
    :param extended_y_file_name:
        Name of the file where the serialized result is saved.
    :param ratio:
        The size of the pairs' subset size.
    :param low:
        Default minimum random value used as label
    :param high:
        Default maximum random value used as label
    :param columns:
        Column mapping used by each segment.
    :return:
        numpy (m_pairs, 3)
            Where each row contains tuples of set0 segment id, set1 segment id, rand 0/1.
    """

    mask_epoch0 = X[:, columns.EPOCH_ID_COLUMN] == 0
    mask_epoch1 = X[:, columns.EPOCH_ID_COLUMN] == 1

    epoch0_set = X[mask_epoch0, :]  # all
    epoch1_set = X[mask_epoch1, :]  # all

    ratio = np.clip(ratio, 0, 1)
    nr_pairs = round(min(epoch0_set.shape[0], epoch1_set.shape[0]) * ratio)

    indx0_seg_id = random.sample(range(epoch0_set.shape[0]), nr_pairs)
    indx1_seg_id = random.sample(range(epoch1_set.shape[0]), nr_pairs)

    set0_seg_id = epoch0_set[indx0_seg_id, columns.SEGMENT_ID_COLUMN]
    set1_seg_id = epoch1_set[indx1_seg_id, columns.SEGMENT_ID_COLUMN]

    rand_y = list(np.random.randint(low, high + 1, nr_pairs))

    np.savetxt(
        extended_y_file_name,
        np.array([set0_seg_id, set1_seg_id, rand_y]).T,
        delimiter=",",
    )

    return np.array([set0_seg_id, set1_seg_id, rand_y]).T


def generate_possible_region_pairs(
    segments: np.ndarray, seg_id0_seg_id1_label: np.ndarray
):
    """
    :param segments:
        numpy array of shape (n_segments, segment_size)
    :param seg_id0_seg_id1_label:
        extended_y, numpy array (n_pairs, 3)
            where each row contains: (id_segment_epoch0, id_segment_epoch1, label=0/1)
    :return:
        numpy array of shape (m_pairs, 7) where each row contain:
            pairs_of_points[i, :3] -> a proposed position of a segment from epoch 0
            pairs_pf_points[i, 3:] -> a proposed position of a segment from epoch 1
            label: 0/1
    """

    segment_pairs = seg_id0_seg_id1_label

    out = np.empty(shape=(0, 7))
    for row in range(segment_pairs.shape[0]):
        id_epoch0 = int(segment_pairs[row, 0])
        id_epoch1 = int(segment_pairs[row, 1])
        label = int(segment_pairs[row, 2])
        points = np.hstack(
            (
                segments[id_epoch0, 0:3] + np.random.normal(0, 1),
                segments[id_epoch1, 3:6] + np.random.normal(0, 1),
                label,
            )
        )
        out = np.vstack((out, points))

    return out


def generate_extended_y_from_prior_knowledge(
    segments: np.ndarray,
    pairs_of_points: np.ndarray,
    threshold_max_distance: float,
    columns=SEGMENT_COLUMNS,
) -> np.ndarray:
    """
    :param segments:
        numpy array of shape (n_segments, segment_size)
    :param pairs_of_points:
        numpy array of shape (m_pairs, 7) where each row contain:
            pair_of_points[i, :3] -> a proposed position of a segment from epoch 0
            pair_pf_points[i, 3:] -> a proposed position of a segment from epoch 1
            label: 0/1
    :param threshold_max_distance:
        the radios accepted threshold for possible segments
    :param columns:
        Column mapping used by each segment.
    :return:
        extended_y, numpy array (n_pairs, 3)
            where each row contains: (id_segment_epoch0, id_segment_epoch1, label=0/1)
    """

    extended_y = np.empty(shape=(0, 3), dtype=float)

    # split points(segments) between epoch0 and epoch1
    epoch0_mask = segments[:, columns.EPOCH_ID_COLUMN] == 0
    epoch1_mask = segments[:, columns.EPOCH_ID_COLUMN] == 1

    X_Y_Z_Columns = [columns.X_COLUMN, columns.Y_COLUMN, columns.Z_COLUMN]

    epoch0_set = segments[epoch0_mask][X_Y_Z_Columns]
    epoch1_set = segments[epoch1_mask][X_Y_Z_Columns]

    # generate kd-tree for each of the 2 sets
    epoch0 = Epoch(epoch0_set.T)
    epoch0._validate_search_tree()
    epoch1 = Epoch(epoch1_set.T)
    epoch1._validate_search_tree()

    # search for the near segments and build the 'extended y'
    for row in pairs_of_points:
        seg_epoch0, seg_epoch1, label = np.split(ary=row, indices_or_sections=[3, 6])
        label = label[0]

        candidates_seg_epoch0 = epoch0._radius_search(
            seg_epoch0, threshold_max_distance
        )
        candidates_seg_epoch1 = epoch1._radius_search(
            seg_epoch1, threshold_max_distance
        )

        if len(candidates_seg_epoch0) > 0 and len(candidates_seg_epoch1) > 0:
            indx_min_epoch0 = candidates_seg_epoch0[
                np.linalg.norm(
                    epoch0.cloud[candidates_seg_epoch0] - seg_epoch0
                ).argmin()
            ]
            indx_min_epoch1 = candidates_seg_epoch1[
                np.linalg.norm(
                    epoch1.cloud[candidates_seg_epoch0] - seg_epoch1
                ).argmin()
            ]

            extended_y = np.vstack(
                (
                    extended_y,
                    # index segment epoch0 , index segment epoch1, label=0/1
                    np.array([indx_min_epoch0, indx_min_epoch1, label]),
                )
            )

    return extended_y


def add_no_corresponding_seg(
    segments: np.ndarray,
    extended_y: np.ndarray = None,
    threshold_max_distance: float = 3,
    algorithm="closest",
    extended_y_file_name: str = "extended_y.csv",
    columns=SEGMENT_COLUMNS,
):
    """
    :param segments:
        numpy array of shape (n_segments, segment_size)
    :param extended_y:
        numpy array (n_pairs, 3)
            where each row contains: (id_segment_epoch0, id_segment_epoch1, label=0/1)
    :param threshold_max_distance:
        the radios accepted threshold for possible segments
    :param algorithm:
        closest - select the closest segment, not used as the corresponding segment
        random - select a random segment, from proximity (threshold_max_distance),
        not used already as a corresponding segment
    :param columns:
        Column mapping used by each segment.
    :param extended_y_file_name:
        In case 'extended_y' is None, this file is used as input fallback.
    :return:
        extended_y, numpy array (n_pairs, 3)
            where each row contains: (id_segment_epoch0, id_segment_epoch1, label=0/1)
    """

    assert (
        algorithm == "closest" or algorithm == "random"
    ), "'selection' parameter can be 'closest/random'"

    if extended_y is None:
        # Resolve the given path
        filename = find_file(extended_y_file_name)
        # Read it
        try:
            logger.info(
                f"Reading tuples of (segment epoch0, segment epoch1, label) from file '{filename}'"
            )
            extended_y = np.genfromtxt(filename, delimiter=",")
        except ValueError:
            raise Py4DGeoError("Malformed file: " + str(filename))

    # construct the corresponding pairs from the set of all pairs
    extended_y_with_label_1 = extended_y[extended_y[:, 2] == 1]

    new_extended_y = np.empty(shape=(0, 3), dtype=float)

    # split points(segments) between epoch0 and epoch1
    epoch0_mask = segments[:, columns.EPOCH_ID_COLUMN] == 0
    epoch1_mask = segments[:, columns.EPOCH_ID_COLUMN] == 1

    # compute index of each segment as part of 'segments'
    epoch0_index = np.asarray(epoch0_mask).nonzero()[0]
    epoch1_index = np.asarray(epoch1_mask).nonzero()[0]

    X_Y_Z_Columns = [columns.X_COLUMN, columns.Y_COLUMN, columns.Z_COLUMN]

    epoch0_set = segments[epoch0_mask][X_Y_Z_Columns]
    epoch1_set = segments[epoch1_mask][X_Y_Z_Columns]

    epoch0_set = epoch0_set.T
    # generate search tree
    epoch1 = Epoch(epoch1_set.T)
    epoch1._validate_search_tree()

    # search for the near segments and build the 'extended y'
    # for row in pairs_of_points:
    for index, row in enumerate(epoch0_set):
        index_seg_epoch0 = epoch0_index[index]

        candidates_seg_epoch1 = epoch1._radius_search(row, threshold_max_distance)

        indexes_seg_epoch1 = epoch1_index[candidates_seg_epoch1]

        if len(indexes_seg_epoch1) == 0:
            continue

        if algorithm == "closest":
            index = (
                (extended_y[:, 0] == index_seg_epoch0)
                & (extended_y[:, 1] == indexes_seg_epoch1[0])
                & (extended_y[:, 2] == 1)
            )

            if not index.any():
                new_extended_y = np.vstack(
                    (new_extended_y, (index_seg_epoch0, indexes_seg_epoch1[0], 0))
                )
            else:
                if len(candidates_seg_epoch1) > 1:
                    new_extended_y = np.vstack(
                        (new_extended_y, (index_seg_epoch0, indexes_seg_epoch1[1], 0))
                    )

        if algorithm == "random":
            while True:
                rand_point = np.random.randint(low=0, high=len(indexes_seg_epoch1))
                index = (
                    (extended_y[:, 0] == index_seg_epoch0)
                    & (extended_y[:, 1] == indexes_seg_epoch1[rand_point])
                    & (extended_y[:, 2] == 1)
                )

                if not index.any():
                    new_extended_y = np.vstack(
                        (
                            new_extended_y,
                            (index_seg_epoch0, indexes_seg_epoch1[rand_point], 0),
                        )
                    )
                    break

    return np.vstack((extended_y, new_extended_y))


class BaseTransformer(TransformerMixin, BaseEstimator, ABC):
    def __init__(self, skip=False, output_file_name=None, columns=None):
        """
        :param skip:
            Whether the current transform is applied or not.
        :param output_file_name:
            File where the result of the 'Transform()' method, a numpy array, is dumped.
        """

        self.skip = skip
        self.output_file_name = output_file_name
        self.columns = columns
        super(BaseTransformer, self).__init__()

    @abstractmethod
    def _fit(self, X, y=None):
        """
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        pass

    @abstractmethod
    def _transform(self, X):
        """
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
        """
        pass

    def fit(self, X, y=None):
        if self.skip:
            return self

        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]

        # Return the transformer
        logger.info("Transformer Fit")

        return self._fit(X, y)

    def transform(self, X):
        """
        param: X
            numpy array
        """

        if self.skip:
            if self.output_file_name != None:
                logger.debug(
                    f"The output file, {self.output_file_name} "
                    f"was set but the transformation process is skipped! (no output)"
                )
            return X

        # Check if fit had been called
        check_is_fitted(self, "n_features_")

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise Py4DGeoError(
                "Shape of input is different from what was seen in `fit`"
            )

        logger.info("Transformer Transform")

        out = self._transform(X)

        if self.output_file_name != None:
            np.savetxt(self.output_file_name, out, delimiter=",")
            logger.info(f"Saving Transform output in file: {self.output_file_name}")

        return out


class PerPointComputation(BaseTransformer):
    def __init__(
        self, skip=False, radius=10, output_file_name=None, columns=LLSV_PCA_COLUMNS
    ):
        """

        :param skip:
            Whether the current transform is applied or not.
        :param radius:
            The radius used to extract the neighbour points using KD-tree.
        :param output_file_name:
            File where the result of the 'Transform()' method, a numpy array, is dumped.
        """

        super(PerPointComputation, self).__init__(
            skip=skip, output_file_name=output_file_name, columns=columns
        )
        self.radius = radius

    def _llsv_and_pca(self, x, X):
        """
        Compute PCA (implicitly, the normal vector as well) and lowest local surface variation
        for point "x" using the set "X" as input.

        :param x:
            a reference to a row, part of the returned structure, of the following form:
                [
                    x, y, z,
                    EpochID,
                    Eigenvalues( 3 columns ),
                    Eigenvectors( 3 columns ) X 3 -> in descending order using vector norm 2,
                    Lowest local surface variation ( 1 column )
                ]
        :param X:
            Subset of the point cloud of numpy array (m_samples, 3), that is found around 'x'(x,y,z) inside a 'radius'.
        :return:
            return a populated x with
            (Eigenvalues, Eigenvectors, Lowest local surface variation)
        """

        size = X.shape[0]

        # compute mean
        X_avg = np.mean(X, axis=0)
        B = X - np.tile(X_avg, (size, 1))

        # Find principal components (SVD)
        U, S, VT = np.linalg.svd(B.T / np.sqrt(size), full_matrices=0)

        x[-13:] = np.hstack(
            (
                S.reshape(1, -1),
                U[:, 0].reshape(1, -1),
                U[:, 1].reshape(1, -1),
                U[:, 2].reshape(1, -1),
                (S[-1] / np.sum(S)).reshape(1, -1),
            )
        )

        return x

    def _fit(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """

        return self

    def _transform(self, X):
        """
        Extending X matrix by adding eigenvalues, eigenvectors, and lowest local surface variation columns.

        :param X:
            A numpy array with x, y, z, EpochID columns.
        :return:
            numpy matrix extended containing the following columns:
                x, y, z ( 3 columns ),
                EpochID ( 1 column ),
                Eigenvalues( 3 columns ),
                Eigenvectors( 3 columns ) X 3 [ in descending normal order ],
                Lowest local surface variation( 1 column )
        """

        X_Y_Z_Columns = [
            self.columns.X_COLUMN,
            self.columns.Y_COLUMN,
            self.columns.Z_COLUMN,
        ]

        mask_epoch0 = X[:, self.columns.EPOCH_ID_COLUMN] == 0
        mask_epoch1 = X[:, self.columns.EPOCH_ID_COLUMN] == 1

        epoch0_set = X[mask_epoch0, :-1]
        epoch1_set = X[mask_epoch1, :-1]

        # currently, Epoch class doesn't accept the empty point set, as a constructor parameter.
        _epoch = [
            Epoch(epoch_set)
            for epoch_set in [epoch0_set, epoch1_set]
            if epoch_set.shape[0] > 0
        ]

        for current_epoch in _epoch:
            current_epoch._validate_search_tree()

        # add extra columns
        # Eigenvalues( 3 columns ) |
        # Eigenvectors( 3 columns ) X 3 [ in descending normal order ] |
        # Lowest local surface variation( 1 column )
        # Total 13 new columns
        new_columns = np.zeros((X.shape[0], 13))
        X = np.hstack((X, new_columns))

        # this process can be parallelized!
        return np.apply_along_axis(
            lambda x: self._llsv_and_pca(
                x,
                _epoch[int(x[self.columns.EPOCH_ID_COLUMN])].cloud[
                    _epoch[int(x[self.columns.EPOCH_ID_COLUMN])]._radius_search(
                        x[X_Y_Z_Columns], self.radius
                    )
                ],
            ),
            1,
            X,
        )


class Segmentation(BaseTransformer):
    def __init__(
        self,
        skip=False,
        radius=2,
        angle_diff_threshold=1,
        distance_3D_threshold=1.5,
        distance_orthogonal_threshold=1.5,
        llsv_threshold=1,
        roughness_threshold=5,
        max_nr_points_neighborhood=100,
        min_nr_points_per_segment=5,
        with_previously_computed_segments=False,
        output_file_name=None,
        columns=SEGMENTED_POINT_CLOUD_COLUMNS,
    ):
        """

        :param skip:
            Whether the current transform is applied or not.
        :param radius:
            The radius used to extract the neighbour points using KD-tree during segmentation process.
        :param angle_diff_threshold:
            Angular deviation threshold for a point candidate’s local normal vector to the normal vector
            of the initial seed point.
        :param distance_3D_threshold:
            Norm 2 distance threshold of the point candidate to the current set of points,
            during the segmentation process.
        :param distance_orthogonal_threshold:
            Orthogonal distance threshold of the point candidate to the current plane segment
            used during the segmentation process.
        :param llsv_threshold:
            The threshold on local surface variation.
        :param roughness_threshold:
            Threshold on local roughness.
        :param max_nr_points_neighborhood:
            The maximum number of points in the neighborhood of the point the candidate used
            for checking during the segmentation process.
        :param min_nr_points_per_segment:
            The minimum number of points required to consider a segment as valid.
        :param with_previously_computed_segments:
            Used for differentiating between the first and the second segmentation.
            ( must be refactored!!! )
        param output_file_name:
            File where the result of the 'Transform()' method, a numpy array, is dumped.
        """

        super(Segmentation, self).__init__(
            skip=skip, output_file_name=output_file_name, columns=columns
        )

        self.radius = radius
        self.angle_diff_threshold = angle_diff_threshold
        self.distance_3D_threshold = distance_3D_threshold
        self.distance_orthogonal_threshold = distance_orthogonal_threshold
        self.llsv_threshold = llsv_threshold
        self.roughness_threshold = roughness_threshold
        self.max_nr_points_neighborhood = max_nr_points_neighborhood
        self.min_nr_points_per_segment = min_nr_points_per_segment
        self.with_previously_computed_segments = with_previously_computed_segments

    def angle_difference_check(self, normal1, normal2):
        """
        Check whether the angle between 2 normalized vectors is less than
        the used segmentation threshold "angle_diff_threshold" (in degrees)

        :param normal1:
            normalized vector
        :param normal2:
            normalized vector
        :return:
            True/False
        """

        # normal1, normal2 have to be unit vectors (and that is the case as a result of the SVD process)
        return angle_difference_compute(normal1, normal2) <= self.angle_diff_threshold

    def distance_3D_set_check(
        self, point, segment_id, X, X_Y_Z_Columns, SEGMENT_ID_COLUMN
    ):
        """
        Check whether the distance between the candidate point and all the points, currently part of the segment
        is less than the 'distance_3D_threshold'.

        :param point:
            Numpy array (3, 1) candidate point during segmentation process.
        :param segment_id:
            The segment ID for which the candidate point is considered.
        :param X:
            numpy array (n_samples, 19)
        :param X_Y_Z_Columns:
            python list containing the indexes of the X,Y,Z columns.
        :param SEGMENT_ID_COLUMN:
            The column index used as the segment id.
        :return:
            True/False
        """

        point_mask = X[:, SEGMENT_ID_COLUMN] == segment_id

        # can be optimized by changing the norm
        return (
            np.min(
                np.linalg.norm(X[point_mask, :3] - point.reshape(1, 3), ord=2, axis=1)
            )
            <= self.distance_3D_threshold
        )

    def compute_distance_orthogonal(self, candidate_point, plane_point, plane_normal):
        """
        Compute the orthogonal distance between the candidate point and the segment represented by its plane.

        :param candidate_point:
            numpy array (3, 1) candidate point during segmentation process.
        :param plane_point:
            numpy array (3, 1) representing a point (most likely, a core point), part of the segment.
        :param plane_normal:
            numpy array (3, 1)
        :return:
            True/False
        """

        d = -plane_point.dot(plane_normal)
        distance = (plane_normal.dot(candidate_point) + d) / np.linalg.norm(
            plane_normal
        )

        return distance

    def distance_orthogonal_check(self, candidate_point, plane_point, plane_normal):
        """
        Check whether the orthogonal distance between the candidate point and the segment represented by its plane
        is less than the 'distance_orthogonal_threshold'.

        :param candidate_point:
            numpy array (3, 1) candidate point during segmentation process.
        :param plane_point:
            numpy array (3, 1) representing a point (most likely, a core point), part of the segment.
        :param plane_normal:
             numpy array (3, 1)
        :return:
            True/False
        """

        distance = self.compute_distance_orthogonal(
            candidate_point, plane_point, plane_normal
        )
        return distance - self.distance_orthogonal_threshold <= 0

    def lowest_local_surface_variance_check(self, llsv):
        """
        Check lowest local surface variance threshold.

        :param llsv:
            lowest local surface variance
        :return:
            True/False
        """

        return llsv <= self.llsv_threshold

    def _fit(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """

        return self
        pass

    def _transform(self, X):
        """
        It applies the segmentation process.

        :param X:
            Rows of points from Epoch0 and Epoch1
            X = (
                    X0, Rows of points from Epoch 0
                    X1  Rows of points from Epoch 1     <- OPTIONAL
                )
            It is assumed that rows for Epoch0 and Epoch1 are not interleaved!

            numpy array (n_points, 17) with the following column structure:
            [
                x, y, z ( 3 column ),
                EpochID ( 1 column ),
                Eigenvalues( 3 columns ), -> that correspond to the next 3 Eigenvectors
                Eigenvectors( 3 columns ) X 3 -> in descending order using vector norm 2,
                Lowest local surface variation ( 1 column )
            ]
        :return:
            numpy array (n_points, 19) with the following column structure:
            [
                x, y, z ( 3 columns ),
                EpochID ( 1 column ),
                Eigenvalues( 3 columns ), -> that correspond to the next 3 Eigenvectors
                Eigenvectors( 3 columns ) X 3 -> in descending order using vector norm 2,
                Lowest local surface variation ( 1 column ),
                Segment_ID ( 1 column ),
                Standard deviation ( 1 column )
            ]
        """

        X_Y_Z_Columns = [
            self.columns.X_COLUMN,
            self.columns.Y_COLUMN,
            self.columns.Z_COLUMN,
        ]

        Normal_Columns = [
            self.columns.EIGENVECTOR_2_X_COLUMN,
            self.columns.EIGENVECTOR_2_Y_COLUMN,
            self.columns.EIGENVECTOR_2_Z_COLUMN,
        ]

        # the new columns are added only if they weren't already been added previously
        if not self.with_previously_computed_segments:
            new_column_segment_id = np.full(
                (X.shape[0], 1), DEFAULT_NO_SEGMENT, dtype=float
            )
            X = np.hstack((X, new_column_segment_id))

            new_column_std_deviation = np.full(
                (X.shape[0], 1), DEFAULT_STD_DEVIATION_OF_NO_CORE_POINT, dtype=float
            )
            X = np.hstack((X, new_column_std_deviation))

        mask_epoch0 = X[:, self.columns.EPOCH_ID_COLUMN] == 0
        mask_epoch1 = X[:, self.columns.EPOCH_ID_COLUMN] == 1

        assert (
            mask_epoch0.shape[0] > 0
        ), "The input X must contain at least elements from Epoch 0, e.g. EPOCH_ID_COLUMN==0"

        logger.debug(
            f"'X' contains {np.count_nonzero(mask_epoch0)} elements from Epoch 0 "
            f"and {np.count_nonzero(mask_epoch1)} from Epoch 1"
        )

        epoch0_set = X[mask_epoch0][:, X_Y_Z_Columns]
        epoch1_set = X[mask_epoch1][:, X_Y_Z_Columns]

        _epoch = [
            Epoch(epoch_set)
            for epoch_set in [epoch0_set, epoch1_set]
            if epoch_set.shape[0] > 0
        ]

        for current_epoch in _epoch:
            current_epoch._validate_search_tree()

        # sort by the "Lowest local surface variation"
        sort_indx_epoch0 = X[mask_epoch0, self.columns.LLSV_COLUMN].argsort()
        sort_indx_epoch1 = X[mask_epoch1, self.columns.LLSV_COLUMN].argsort()
        sort_indx_epoch = [sort_indx_epoch0, sort_indx_epoch1]

        # it is assumed that the rows for Epoch0 and rows for Epoch1 are not interleaved!
        offset_in_X = [0, sort_indx_epoch0.shape[0]]

        # initialization required between multiple Segmentations
        seg_id = np.max(X[:, self.columns.SEGMENT_ID_COLUMN])

        for epoch_id in range(len(_epoch)):
            for indx_row in sort_indx_epoch[epoch_id] + offset_in_X[epoch_id]:
                # no part of a segment yet
                if X[indx_row, self.columns.SEGMENT_ID_COLUMN] < 0:
                    seg_id += 1
                    X[indx_row, self.columns.SEGMENT_ID_COLUMN] = seg_id

                    cumulative_distance_for_std_deviation = 0
                    nr_points_for_std_deviation = 0

                    indx_kd_tree_list = _epoch[epoch_id]._radius_search(
                        X[indx_row, X_Y_Z_Columns], self.radius
                    )[: self.max_nr_points_neighborhood]
                    for indx_kd_tree in indx_kd_tree_list:
                        if (
                            X[
                                indx_kd_tree + offset_in_X[epoch_id],
                                self.columns.SEGMENT_ID_COLUMN,
                            ]
                            < 0
                            and self.angle_difference_check(
                                X[indx_row, Normal_Columns],
                                X[indx_kd_tree + offset_in_X[epoch_id], Normal_Columns],
                            )
                            and self.distance_3D_set_check(
                                X[indx_kd_tree + offset_in_X[epoch_id], X_Y_Z_Columns],
                                seg_id,
                                X,
                                X_Y_Z_Columns,
                                self.columns.SEGMENT_ID_COLUMN,
                            )
                            and self.distance_orthogonal_check(
                                X[indx_kd_tree + offset_in_X[epoch_id], X_Y_Z_Columns],
                                X[indx_row, X_Y_Z_Columns],
                                X[indx_row, Normal_Columns],
                            )
                            and self.lowest_local_surface_variance_check(
                                X[
                                    indx_kd_tree + offset_in_X[epoch_id],
                                    self.columns.LLSV_COLUMN,
                                ]
                            )
                        ):
                            X[
                                indx_kd_tree + offset_in_X[epoch_id],
                                self.columns.SEGMENT_ID_COLUMN,
                            ] = seg_id
                            cumulative_distance_for_std_deviation += (
                                self.compute_distance_orthogonal(
                                    X[
                                        indx_kd_tree + offset_in_X[epoch_id],
                                        X_Y_Z_Columns,
                                    ],
                                    X[indx_row, X_Y_Z_Columns],
                                    X[indx_row, Normal_Columns],
                                )
                                ** 2
                            )
                            nr_points_for_std_deviation += 1

                    nr_points_segment = np.count_nonzero(
                        X[:, self.columns.SEGMENT_ID_COLUMN] == seg_id
                    )

                    # not enough points or 'roughness_threshold' exceeded
                    if (
                        nr_points_segment < self.min_nr_points_per_segment
                        or cumulative_distance_for_std_deviation
                        / nr_points_for_std_deviation
                        >= self.roughness_threshold
                    ):
                        mask_seg_id = X[:, self.columns.SEGMENT_ID_COLUMN] == seg_id
                        X[mask_seg_id, self.columns.SEGMENT_ID_COLUMN] = (
                            DEFAULT_NO_SEGMENT
                        )
                        # since we don't have a new segment
                        seg_id -= 1
                    else:
                        X[indx_row, self.columns.STANDARD_DEVIATION_COLUMN] = (
                            cumulative_distance_for_std_deviation
                            / nr_points_for_std_deviation
                        )
        return X


class PostPointCloudSegmentation(BaseTransformer):
    def __init__(
        self,
        skip=False,
        compute_normal=True,
        output_file_name=None,
        columns=SEGMENTED_POINT_CLOUD_COLUMNS,
    ):
        """
        :param skip:
            Whether the current transform is applied or not.
        :param compute_normal:

        :param output_file_name:
            File where the result of the 'Transform()' method, a numpy array, is dumped.
        :param columns:

        """

        super().__init__(skip=skip, output_file_name=output_file_name, columns=columns)
        self.compute_normal = compute_normal

    def compute_distance_orthogonal(self, candidate_point, plane_point, plane_normal):
        """
        Compute the orthogonal distance between the candidate point and the segment represented by its plane.

        :param candidate_point:
            numpy array (3, 1) candidate point during segmentation process.
        :param plane_point:
            numpy array (3, 1) representing a point (most likely, a core point), part of the segment.
        :param plane_normal:
            numpy array (3, 1)
        :return:
            True/False
        """

        d = -plane_point.dot(plane_normal.T)
        distance = (plane_normal.dot(candidate_point) + d) / np.linalg.norm(
            plane_normal
        )

        return distance

    def pca_compute_normal_and_mean(self, X):
        """
            Perform PCA.
            The order of the eigenvalues and eigenvectors is consistent.

        :param X:
            numpy array of shape (n_points, 3) with [x, y, z] columns.
        :return:
            a tuple of:
                Eig. values as numpy array of shape (1, 3)
                Eig. vector0 as numpy array of shape (1, 3)
                Eig. vector1 as numpy array of shape (1, 3)
                Eig. vector2 (normal vector) as numpy array of shape (1, 3)
                Position of the normal vector as numpy array of shape (1, 3),
                    approximated as the mean of the input points.
        """

        size = X.shape[0]

        # compute mean
        X_avg = np.mean(X, axis=0)
        B = X - np.tile(X_avg, (size, 1))

        # Find principal components (SVD)
        U, S, VT = np.linalg.svd(B.T / np.sqrt(size), full_matrices=0)

        assert S[0] != 0, "eig. value should not be 0!"
        assert S[1] != 0, "eig. value should not be 0!"
        assert S[2] != 0, "eig. value should not be 0!"

        # Eig. values,
        # Eig. Vector0, Eig. Vector1, Eig. Vector2( the norma vector),
        # 'position' of the normal vector
        return (
            S.reshape(1, -1),
            U[:, 0].reshape(1, -1),
            U[:, 1].reshape(1, -1),
            U[:, 2].reshape(1, -1),
            X_avg.reshape(1, -1),
        )

    def _fit(self, X, y=None):
        """
        :param X:
        :param y:
        :return:
        """

        return self
        pass

    def _transform(self, X):
        """

        :param X:
        :return:
        """

        X_Y_Z_Columns = [
            self.columns.X_COLUMN,
            self.columns.Y_COLUMN,
            self.columns.Z_COLUMN,
        ]

        Eigval = [
            self.columns.EIGENVALUE0_COLUMN,
            self.columns.EIGENVALUE1_COLUMN,
            self.columns.EIGENVALUE2_COLUMN,
        ]

        Eigvec0 = [
            self.columns.EIGENVECTOR_0_X_COLUMN,
            self.columns.EIGENVECTOR_0_Y_COLUMN,
            self.columns.EIGENVECTOR_0_Z_COLUMN,
        ]

        Eigvec1 = [
            self.columns.EIGENVECTOR_1_X_COLUMN,
            self.columns.EIGENVECTOR_1_Y_COLUMN,
            self.columns.EIGENVECTOR_1_Z_COLUMN,
        ]

        Normal_Columns = [
            self.columns.EIGENVECTOR_2_X_COLUMN,
            self.columns.EIGENVECTOR_2_Y_COLUMN,
            self.columns.EIGENVECTOR_2_Z_COLUMN,
        ]

        highest_segment_id_used = int(X[:, self.columns.SEGMENT_ID_COLUMN].max())

        for i in range(0, highest_segment_id_used + 1):
            mask = X[:, self.columns.SEGMENT_ID_COLUMN] == float(i)
            # extract all points, that are part of the same segment
            set_cloud = X[mask, :][:, X_Y_Z_Columns]

            eig_values, e0, e1, normal, position = self.pca_compute_normal_and_mean(
                set_cloud
            )

            indexes = np.where(mask == True)[0]

            # compute the closest point from the current segment to calculated "position"
            indx_min_in_indexes = np.linalg.norm(
                x=set_cloud - position, axis=1
            ).argmin()
            indx_min_in_X = indexes[indx_min_in_indexes]

            X[indx_min_in_X, Eigval] = eig_values
            X[indx_min_in_X, Eigvec0] = e0
            X[indx_min_in_X, Eigvec1] = e1

            if self.compute_normal:
                X[indx_min_in_X, Normal_Columns] = normal

            cumulative_distance_for_std_deviation = 0
            nr_points_for_std_deviation = indexes.shape[0] - 1
            for indx in indexes:
                cumulative_distance_for_std_deviation += (
                    self.compute_distance_orthogonal(
                        X[indx, X_Y_Z_Columns],
                        X[indx_min_in_X, X_Y_Z_Columns],
                        normal.reshape(1, -1),
                    )
                    ** 2
                )

            X[indx_min_in_X, self.columns.STANDARD_DEVIATION_COLUMN] = (
                cumulative_distance_for_std_deviation / nr_points_for_std_deviation
            )

        return X


class ExtractSegments(BaseTransformer):
    def __init__(self, skip=False, output_file_name=None, columns=SEGMENT_COLUMNS):
        """

        :param skip:
            Whether the current transform is applied or not.
        :param output_file_name:
            File where the result of the 'Transform()' method, a numpy array, is dumped.
        """

        super(ExtractSegments, self).__init__(
            skip=skip, output_file_name=output_file_name, columns=columns
        )

    def _fit(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """

        return self
        pass

    def _transform(self, X):
        """
        Transform the numpy array of 'point cloud' to numpy array of 'segments' and extend the structure by adding
        a new column containing the 'number of points found in Segment_ID'. During this process, only one point
        that is part of a segment ( the core point ) is maintained, while all the others are discarded.

        :param X:
            numpy array (n_points, 19) with the following column structure:
            [
                x, y, z ( 3 columns ), -> point from cloud
                EpochID ( 1 column ),
                Eigenvalues( 3 columns ), -> that correspond to the next 3 Eigenvectors
                Eigenvectors( 3 columns ) X 3 -> in descending order using vector norm 2,
                Lowest local surface variation ( 1 column ),
                Segment_ID ( 1 column ),
                Standard deviation ( 1 column )
            ]
        :return:
            numpy array (n_segments, 20) with the following column structure:
            [
                x, y, z ( 3 columns ), -> segment, core point
                EpochID ( 1 column ),
                Eigenvalues( 3 columns ), -> that correspond to the next 3 Eigenvectors
                Eigenvectors( 3 columns ) X 3 -> in descending order using vector norm 2,
                Lowest local surface variation ( 1 column ),
                Segment_ID ( 1 column ),
                Standard deviation ( 1 column ),
                Number of points found in Segment_ID segment ( 1 column )
            ]
        """

        max_segment_id = int(X[:, self.columns.SEGMENT_ID_COLUMN].max())
        X_Segments = np.empty(
            (int(max_segment_id) + 1, self.columns.NUMBER_OF_COLUMNS), dtype=float
        )

        for i in range(0, max_segment_id + 1):
            mask = X[:, self.columns.SEGMENT_ID_COLUMN] == float(i)
            set_cloud = X[mask, :]  # all
            nr_points = set_cloud.shape[0]

            # arg_min = set_cloud[:, LLSV_COLUMN].argmin()
            # X_Segments[i, :-1] = set_cloud[arg_min, :]

            # find the CoM point, e.g. the point which has STD != DEFAULT_STD_DEVIATION_OF_NO_CORE_POINT
            mask_std = (
                set_cloud[:, self.columns.STANDARD_DEVIATION_COLUMN]
                != DEFAULT_STD_DEVIATION_OF_NO_CORE_POINT
            )
            set_cloud_std = set_cloud[mask_std, :]
            assert (
                set_cloud_std.shape[0] == 1
            ), "Only one element within a segment should have the standard deviation computed!"
            X_Segments[i, :-1] = set_cloud_std[0, :]

            X_Segments[i, self.columns.NR_POINTS_PER_SEG_COLUMN] = nr_points

        return X_Segments


class BuilderExtended_y(ABC):
    def __init__(self, columns):
        super().__init__()
        self.columns = columns

    @abstractmethod
    def generate_extended_y(self, X):
        """
        Generates tuples of ( segment index epoch 0, segment index epoch 1, 0/1 label )

        :param X:
            numpy array of shape (n_segments, segment_features_size) containing all the segments for both,
            epoch 0 and epoch 1. Each row is a segment.
        :return:
            numpy array with shape (n_segments, 3)
        """
        pass


class BuilderExtended_y_Visually(BuilderExtended_y):
    def __init__(self, columns=SEGMENT_COLUMNS):
        super(BuilderExtended_y_Visually, self).__init__(columns=columns)

        self.current_pair = [None] * 2
        self.constructed_extended_y = np.empty(shape=(0, 3))

    def toggle_transparenct(self, evt):
        if evt.keyPressed == "z":
            # transparency toggle
            for segment in self.sets:
                if segment.alpha() < 1.0:
                    segment.alpha(1)
                else:
                    segment.alpha(0.5)
            self.plt.render()

        if evt.keyPressed == "g":
            # toggle red
            for segment in self.sets:
                if segment.epoch == 0:
                    if segment.isOn == True:
                        segment.off()
                    else:
                        segment.on()
                    segment.isOn = not segment.isOn
            self.plt.render()

        if evt.keyPressed == "d":
            # toggle green
            for segment in self.sets:
                if segment.epoch == 1:
                    if segment.isOn == True:
                        segment.off()
                    else:
                        segment.on()
                    segment.isOn = not segment.isOn
            self.plt.render()

    def controller(self, evt):
        """

        :param evt:
        :return:
        """
        if not evt.actor:
            # no hit, return
            return
        logger.debug("point coords =%s", str(evt.picked3d), exc_info=1)
        if evt.isPoints:
            logger.debug("evt.actor = s", str(evt.actor))
            self.current_pair[int(evt.actor.epoch)] = evt.actor.id

        if self.current_pair[0] != None and self.current_pair[1] != None:
            # we have a pair
            self.add_pair_button.status(self.add_pair_button.states[1])
        else:
            # we don't
            self.add_pair_button.status(self.add_pair_button.states[0])

    def event_add_pair_button(self):
        """

        :return:
        """

        if self.current_pair[0] != None and self.current_pair[1] != None:
            try:
                self.constructed_extended_y = np.vstack(
                    (
                        self.constructed_extended_y,
                        np.array(
                            [
                                self.current_pair[0],
                                self.current_pair[1],
                                int(self.label.status()),
                            ]
                        ),
                    )
                )

                self.current_pair[0] = None
                self.current_pair[1] = None
                self.label.status(self.label.states[0])  # firs state "None"

                self.add_pair_button.switch()
            except:
                logger.error("You must select 0 or 1 as label")

    def segments_visualizer(self, X):
        """

        :param X:
        :return:
        """

        self.sets = []

        nr_segments = X.shape[0]
        colors = [(1, 0, 0), (0, 1, 0)]
        self.plt = Plotter(axes=3)

        self.plt.add_callback("EndInteraction", self.controller)
        self.plt.add_callback("KeyPress", self.toggle_transparenct)

        for i in range(0, nr_segments):
            # mask = X[:, 17] == float(i)
            # set_cloud = X[mask, :3]  # x,y,z

            if X[i, self.columns.EPOCH_ID_COLUMN] == 0:
                color = colors[0]
            else:
                color = colors[1]

            # self.sets = self.sets + [Points(set_cloud, colors[i], alpha=1, r=10)]

            # self.sets = self.sets + [ Point( pos=(X[i, 0],X[i, 1],X[i, 2]), r=15, c=colors[i], alpha=1 ) ]
            ellipsoid = Ellipsoid(
                pos=(X[i, 0], X[i, 1], X[i, 2]),
                axis1=[
                    X[i, self.columns.EIGENVECTOR_0_X_COLUMN]
                    * X[i, self.columns.EIGENVALUE0_COLUMN]
                    * 0.5,
                    X[i, self.columns.EIGENVECTOR_0_Y_COLUMN]
                    * X[i, self.columns.EIGENVALUE0_COLUMN]
                    * 0.5,
                    X[i, self.columns.EIGENVECTOR_0_Z_COLUMN]
                    * X[i, self.columns.EIGENVALUE0_COLUMN]
                    * 0.3,
                ],
                axis2=[
                    X[i, self.columns.EIGENVECTOR_1_X_COLUMN]
                    * X[i, self.columns.EIGENVALUE1_COLUMN]
                    * 0.5,
                    X[i, self.columns.EIGENVECTOR_1_Y_COLUMN]
                    * X[i, self.columns.EIGENVALUE1_COLUMN]
                    * 0.5,
                    X[i, self.columns.EIGENVECTOR_1_Z_COLUMN]
                    * X[i, self.columns.EIGENVALUE1_COLUMN]
                    * 0.5,
                ],
                axis3=[
                    X[i, self.columns.EIGENVECTOR_2_X_COLUMN] * 0.1,
                    X[i, self.columns.EIGENVECTOR_2_Y_COLUMN] * 0.1,
                    X[i, self.columns.EIGENVECTOR_2_Z_COLUMN] * 0.1,
                ],
                res=24,
                c=color,
                alpha=1,
            )

            ellipsoid.id = X[i, self.columns.SEGMENT_ID_COLUMN]
            ellipsoid.epoch = X[i, self.columns.EPOCH_ID_COLUMN]
            ellipsoid.isOn = True
            self.sets = self.sets + [ellipsoid]

        self.label = self.plt.add_button(
            lambda: self.label.switch(),
            states=["Label (0/1)", "0", "1"],  # None
            c=["w", "w", "w"],
            bc=["bb", "lr", "lg"],
            pos=(0.90, 0.25),
            size=24,
        )

        self.add_pair_button = self.plt.add_button(
            self.event_add_pair_button,
            states=["Select pair", "Add pair"],
            c=["w", "w"],
            bc=["lg", "lr"],
            pos=(0.90, 0.15),
            size=24,
        )

        self.plt.show(
            self.sets,
            Text2D(
                "Select multiple pairs of red-green Ellipsoids with their corresponding labels (0/1) "
                "and then press 'Select pair'\n"
                "'z' - toggle transparency on/off 'g' - toggle on/off red ellipsoids 'd' toggle on/off red ellipsoids",
                pos="top-left",
                bg="k",
                s=0.7,
            ),
        ).close()
        return self.constructed_extended_y

    def generate_extended_y(self, X):
        """
        :param X:
        :return:
        """

        return self.segments_visualizer(X)


class ClassifierWrapper(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        neighborhood_search_radius=3,
        threshold_probability_most_similar=0.8,
        diff_between_most_similar_2=0.1,
        classifier=RandomForestClassifier(),
        columns=SEGMENT_COLUMNS,
    ):
        """
        :param neighborhood_search_radius:
            Maximum accepted Euclidean distance for any candidate segments.
        :param threshold_probability_most_similar:
            Lower bound probability threshold for the most similar ( candidate ) plane.
        :param diff_between_most_similar_2:
            Lower bound threshold of difference between first 2, most similar planes.
        :param classifier:
            The classifier used, default is RandomForestClassifier. ( sk-learn )
        :param columns:
            Column mapping used by seg_epoch0 and seg_epoch0
        """

        super().__init__()

        self.neighborhood_search_radius = neighborhood_search_radius
        self.threshold_probability_most_similar = threshold_probability_most_similar
        self.diff_between_most_similar_2 = diff_between_most_similar_2
        self.classifier = classifier
        self.columns = columns

    def compute_similarity_between(
        self, seg_epoch0: np.ndarray, seg_epoch1: np.ndarray
    ) -> np.ndarray:
        """
        Similarity function between 2 segments.

        :param seg_epoch0:
            segment from epoch0, numpy array (1, 20) with the following column structure:
                [
                    x, y, z ( 3 columns ), -> segment, core point
                    EpochID ( 1 column ),
                    Eigenvalues( 3 columns ), -> that correspond to the next 3 Eigenvectors
                    Eigenvectors( 3 columns ) X 3 -> in descending order using vector norm 2,
                    Lowest local surface variation ( 1 column ),
                    Segment_ID ( 1 column ),
                    Standard deviation ( 1 column ),
                    Number of points found in Segment_ID segment ( 1 column )
                ]
        :param seg_epoch1:
            segment from epoch1, same structure as 'seg_epoch0'
        :return:
            numpy array of shape (6,) containing:
                angle, -> angle between plane normal vectors
                points_density_diff, -> difference between points density between pairs of segments
                eigen_value_smallest_diff, -> difference in the quality of plane fit (smallest eigenvalue)
                eigen_value_largest_diff, -> difference in plane extension (largest eigenvalue)
                eigen_value_middle_diff, -> difference in orthogonal plane extension (middle eigenvalue)
                nr_points_diff, -> difference in number of points per plane
        """

        Normal_Columns = [
            self.columns.EIGENVECTOR_2_X_COLUMN,
            self.columns.EIGENVECTOR_2_Y_COLUMN,
            self.columns.EIGENVECTOR_2_Z_COLUMN,
        ]

        angle = angle_difference_compute(
            seg_epoch0[Normal_Columns], seg_epoch1[Normal_Columns]
        )

        points_density_seg_epoch0 = seg_epoch0[
            self.columns.NR_POINTS_PER_SEG_COLUMN
        ] / (
            seg_epoch0[self.columns.EIGENVALUE0_COLUMN]
            * seg_epoch0[self.columns.EIGENVALUE1_COLUMN]
        )

        points_density_seg_epoch1 = seg_epoch1[
            self.columns.NR_POINTS_PER_SEG_COLUMN
        ] / (
            seg_epoch1[self.columns.EIGENVALUE0_COLUMN]
            * seg_epoch1[self.columns.EIGENVALUE1_COLUMN]
        )

        points_density_diff = abs(points_density_seg_epoch0 - points_density_seg_epoch1)

        eigen_value_smallest_diff = abs(
            seg_epoch0[self.columns.EIGENVALUE2_COLUMN]
            - seg_epoch1[self.columns.EIGENVALUE2_COLUMN]
        )
        eigen_value_largest_diff = abs(
            seg_epoch0[self.columns.EIGENVALUE0_COLUMN]
            - seg_epoch1[self.columns.EIGENVALUE0_COLUMN]
        )
        eigen_value_middle_diff = abs(
            seg_epoch0[self.columns.EIGENVALUE1_COLUMN]
            - seg_epoch1[self.columns.EIGENVALUE1_COLUMN]
        )

        nr_points_diff = abs(
            seg_epoch0[self.columns.NR_POINTS_PER_SEG_COLUMN]
            - seg_epoch1[self.columns.NR_POINTS_PER_SEG_COLUMN]
        )

        return np.array(
            [
                angle,
                points_density_diff,
                eigen_value_smallest_diff,
                eigen_value_largest_diff,
                eigen_value_middle_diff,
                nr_points_diff,
            ]
        )

    def _build_X_similarity(self, y_row, X):
        """

        :param y_row:
            numpy array of ( segment epoch0 id, segment epoch1 id, label(0/1) )
        :param X:
            numpy array of shape (n_segments, segment_features_size) containing all the segments for both,
            epoch 0 and epoch 1. Each row is a segment.
        :return:
            numpy array containing the similarity value between 2 segments.
        """

        seg_epoch0 = X[int(y_row[0]), :]
        seg_epoch1 = X[int(y_row[1]), :]

        return self.compute_similarity_between(seg_epoch0, seg_epoch1)

    def fit(self, X, y):
        """
        This method takes care of the learning process by training the chosen 'classifier', using labeled data.

        :param X:
            numpy array (n_segments, segment_size)
        :param y:
            numpy array of shape (m_extended_y, 3) where 'extended y' has the following structure:
        ( tuples of index segment from epoch0, index segment from epoch1, label(0/1) )
        """

        X_similarity = np.apply_along_axis(
            lambda y_row: self._build_X_similarity(y_row, X), 1, y
        )

        # Check that X and y have correct shape
        # X_similarity, y = check_X_y(X_similarity, y, multi_output=True)

        # Store the classes seen during fit
        # self.classes_ = unique_labels(y[:, 2])

        # self.X_ = X_similarity
        # self.y_ = y[:, 2]

        logger.info(f"Fit ClassifierWrapper")

        # Return the classifier
        return self.classifier.fit(X_similarity, y[:, 2])

    def predict(self, X):
        """
        For a set of segments from epoch 0 and epoch 1 it computes which one corresponds.

        :param X:
            numpy array (n_segments, 20) with the following column structure:
            [
                x, y, z ( 3 columns ), -> segment, core point
                EpochID ( 1 column ),
                Eigenvalues( 3 columns ), -> that correspond to the next 3 Eigenvectors
                Eigenvectors( 3 columns ) X 3 -> in descending order using vector norm 2,
                Lowest local surface variation ( 1 column ),
                Segment_ID ( 1 column ),
                Standard deviation ( 1 column ),
                Number of points found in Segment_ID segment ( 1 column )
            ]

        :return:
            numpy array where each row contains a pair of segments:
                [segment epoch 0, segment epoch 1]
        """

        # Check if fit had been called
        # check_is_fitted(self, ["X_", "y_"])

        # Input validation
        # X = check_array(X)

        mask_epoch0 = X[:, self.columns.EPOCH_ID_COLUMN] == 0
        mask_epoch1 = X[:, self.columns.EPOCH_ID_COLUMN] == 1

        epoch0_set = X[mask_epoch0, :]  # all
        epoch1_set = X[mask_epoch1, :]  # all

        self.epoch1_segments = Epoch(
            epoch1_set[
                :, [self.columns.X_COLUMN, self.columns.Y_COLUMN, self.columns.Z_COLUMN]
            ]
        )
        self.epoch1_segments._validate_search_tree()

        list_segments_pair = np.empty((0, epoch0_set.shape[1] + epoch1_set.shape[1]))

        # this operation can be parallelized
        for epoch0_set_row in epoch0_set:
            list_candidates = self.epoch1_segments._radius_search(
                epoch0_set_row, self.neighborhood_search_radius
            )

            list_classified = np.array(
                [
                    self.classifier.predict_proba(
                        self.compute_similarity_between(
                            epoch0_set_row, epoch1_set[candidate, :]
                        ).reshape(1, -1)
                    )[0][1]
                    for candidate in list_candidates
                ]
            )

            if len(list_classified) < 2:
                continue

            most_similar = list_classified.argsort()[-2:]

            if (
                most_similar[1] >= self.threshold_probability_most_similar
                and abs(most_similar[1] - most_similar[0])
                >= self.diff_between_most_similar_2
            ):
                list_segments_pair = np.vstack(
                    (
                        list_segments_pair,
                        np.hstack(
                            (epoch0_set_row, epoch1_set[most_similar[-1], :])
                        ).reshape(1, -1),
                    )
                )

        return list_segments_pair


class PBM3C2:
    def __init__(
        self,
        per_point_computation=PerPointComputation(),
        segmentation=Segmentation(),
        second_segmentation=Segmentation(),
        extract_segments=ExtractSegments(),
        classifier=ClassifierWrapper(),
    ):

        logger.warning(
            f"This method is in experimental stage and undergoing active development."
        )

        """
        :param per_point_computation:
            lowest local surface variation and PCA computation. (computes the normal vector as well)
        :param segmentation:
            The object used for the first segmentation.
        :param second_segmentation:
            The object used for the second segmentation.
        :param extract_segments:
            The object used for building the segments.
        :param classifier:
            An instance of ClassifierWrapper class. The default wrapped classifier used is sk-learn RandomForest.
        """

        self._per_point_computation = per_point_computation
        self._segmentation = segmentation
        self._second_segmentation = second_segmentation
        self._extract_segments = extract_segments
        self._classifier = classifier

        self._second_segmentation.set_params(with_previously_computed_segments=True)

    def _reconstruct_input_with_normals(self, epoch, epoch_id, columns):
        """
        It is an adapter from [x, y, z, N_x, N_y, N_z, Segment_ID] column structure of input 'epoch'
        to an output equivalent with the following pipeline computation:
                ("Transform LLSV_and_PCA"), ("Transform Segmentation"), ("Transform Second Segmentation")

        Note: When comparing distance results between this notebook and the base algorithm notebook, you might notice,
        that results do not necessarily agree even if the given segmentation information is exactly
        the same as the one computed in the base algorithm.
        This is due to the reconstruction process in this algorithm being forced to select the segment position
        (exported as the core point) from the segment points instead of reconstructing the correct position
        from the base algorithm.

        :param epoch:
            Epoch object where each row has the following format: [x, y, z, N_x, N_y, N_z, Segment_ID]
        :param epoch_id:
            is 0 or 1 and represents one of the epochs used as part of distance computation.
        :param columns:

        :return:
            numpy array of shape (n_points, 19) with the following column structure:
                [
                    x,y,z, -> Center of the Mass
                    EPOCH_ID_COLUMN, ->0/1
                    Eigenvalue 1, Eigenvalue 2, Eigenvalue 3,
                    Eigenvector0 (x,y,z),
                    Eigenvector1 (x,y,z),
                    Eigenvector2 (x,y,z), -> Normal vector
                    LLSV_COLUMN, -> lowest local surface variation
                    SEGMENT_ID_COLUMN,
                    STANDARD_DEVIATION_COLUMN
                ]
        """

        # x, y, z, N_x, N_y, N_z, Segment_ID
        assert epoch.shape[1] == 3 + 3 + 1, "epoch size mismatch!"

        return np.hstack(
            (
                epoch[:, :3],  # x,y,z      X 3
                np.full(
                    (epoch.shape[0], 1), epoch_id, dtype=float
                ),  # EPOCH_ID_COLUMN    X 1
                np.full((epoch.shape[0], 3), 0, dtype=float),  # Eigenvalue X 3
                np.full(
                    (epoch.shape[0], 6), 0, dtype=float
                ),  # Eigenvector0, Eigenvector1        X 6
                epoch[:, 3:6],  # Eigenvector2      X 3
                np.full((epoch.shape[0], 1), 0, dtype=float).reshape(
                    -1, 1
                ),  # LLSV_COLUMN
                epoch[:, -1].reshape(-1, 1),  # SEGMENT_ID_COLUMN
                np.full(
                    (epoch.shape[0], 1),
                    DEFAULT_STD_DEVIATION_OF_NO_CORE_POINT,
                    dtype=float,
                ).reshape(
                    -1, 1
                ),  # STANDARD_DEVIATION_COLUMN
            )
        )

    def _reconstruct_input_without_normals(self, epoch, epoch_id, columns):
        """
        It is an adapter from [x, y, z, Segment_ID] column structure of input 'epoch'
        to an output equivalent with the following pipeline computation:
            ("Transform LLSV_and_PCA"), ("Transform Segmentation"), ("Transform Second Segmentation")

        Note: When comparing distance results between this notebook and the base algorithm notebook, you might notice,
        that results do not necessarily agree even if the given segmentation information is exactly
        the same as the one computed in the base algorithm.
        This is due to the reconstruction process in this algorithm being forced to select the segment position
        (exported as the core point) from the segment points instead of reconstructing the correct position
        from the base algorithm.

        :param epoch:
            Epoch object where each row contains by: [x, y, z, Segment_ID]
        :param epoch_id:
            is 0 or 1 and represents one of the epochs used as part of distance computation.
        :param columns

        :return:
            numpy array of shape (n_points, 19) with the following column structure:
                [
                    x,y,z, -> Center of the Mass
                    Epoch_ID, ->0/1
                    Eigenvalue 1, Eigenvalue 2, Eigenvalue 3,
                    Eigenvector0 (x,y,z),
                    Eigenvector1 (x,y,z),
                    Eigenvector2 (x,y,z), -> Normal vector
                    LLSV, -> lowest local surface variation
                    Segment_ID,
                    Standard deviation
                ]
        """

        # [x, y, z, Segment_ID] or [x, y, z, N_x, N_y, N_z, Segment_ID]
        assert (
            epoch.shape[1] == 3 + 1 or epoch.shape[1] == 3 + 3 + 1
        ), "epoch size mismatch!"

        return np.hstack(
            (
                epoch[:, :3],  # x,y,z     X 3
                np.full(
                    (epoch.shape[0], 1), epoch_id, dtype=float
                ),  # EPOCH_ID_COLUMN X 1
                np.full((epoch.shape[0], 3), 0, dtype=float),  # Eigenvalue X 3
                np.full(
                    (epoch.shape[0], 6), 0, dtype=float
                ),  # Eigenvector0, Eigenvector1 X 6
                np.full((epoch.shape[0], 3), 0, dtype=float),  # Eigenvector2 X 3
                np.full((epoch.shape[0], 1), 0, dtype=float).reshape(
                    -1, 1
                ),  # LLSV_COLUMN
                epoch[:, -1].reshape(-1, 1),  # SEGMENT_ID_COLUMN
                np.full(
                    (epoch.shape[0], 1),
                    DEFAULT_STD_DEVIATION_OF_NO_CORE_POINT,
                    dtype=float,
                ).reshape(
                    -1, 1
                ),  # STANDARD_DEVIATION_COLUMN
            )
        )

    @staticmethod
    def _print_default_parameters(kwargs, pipeline_param_dict):
        """
        :param kwargs:
        :param pipeline_param_dict
        """

        # print the default parameters
        if ("get_pipeline_options", True) in kwargs.items():
            logger.info(
                f"----\n "
                f"The default parameters are:\n "
                f"{pp.pformat(pipeline_param_dict)} \n"
                f"----\n"
            )
            del kwargs["get_pipeline_options"]

    @staticmethod
    def _overwrite_pipeline_parameters(
        kwargs, pipeline, message="The pipeline parameters after overwriting are:"
    ):
        """
        :param kwargs:
        :param pipeline:

        :return: unused_kwargs
            The unused parameters / not found as part of 'pipeline'
        """

        unused_kwargs = {}

        # if we have parameters
        if len(kwargs.items()) > 0:
            pipeline_params = pipeline.get_params()
            # overwrite the default parameters
            for key, value in kwargs.items():
                if key in pipeline_params.keys():
                    pipeline.set_params(**{key: value})
                    logger.debug(f"The pipeline parameter '{key}' is now '{value}'")
                else:
                    unused_kwargs[key] = value
            logger.info(
                f"----\n "
                f"{message} \n"
                f"{pp.pformat(pipeline.get_params())} \n"
                f"----\n"
            )
        else:
            logger.info("No pipeline parameter is overwritten")

        return unused_kwargs

    def generate_extended_labels_interactively(
        self,
        epoch0: typing.Union[Epoch, None] = None,
        epoch1: typing.Union[Epoch, None] = None,
        builder_extended_y: BuilderExtended_y_Visually = BuilderExtended_y_Visually(),
        **kwargs,
    ) -> typing.Union[typing.Tuple[np.ndarray, np.ndarray], None]:
        """
        Given 2 Epochs, it builds a pair of (segments and 'extended y').

        :param epoch0:
            Epoch object.
        :param epoch1:
            Epoch object.
        :param builder_extended_y:
            The object is used for generating 'extended y', visually.
        :param kwargs:

            Used for customize the default pipeline parameters.

            Getting the default parameters:
            e.g. "get_pipeline_options"
                In case this parameter is True, the method will print the pipeline options as kwargs.

            e.g. "output_file_name" (of a specific step in the pipeline) default value is "None".
                In case of setting it, the result of computation at that step is dump as xyz file.
            e.g. "distance_3D_threshold" (part of Segmentation Transform)

            this process is stateless

        :return:
            tuple [Segments, 'extended y'] | None

            where:

            'Segments' has the following column structure:
                    X_COLUMN, Y_COLUMN, Z_COLUMN, -> Center of Gravity
                    EPOCH_ID_COLUMN, -> 0/1
                    EIGENVALUE0_COLUMN, EIGENVALUE1_COLUMN, EIGENVALUE2_COLUMN,
                    EIGENVECTOR_0_X_COLUMN, EIGENVECTOR_0_Y_COLUMN, EIGENVECTOR_0_Z_COLUMN,
                    EIGENVECTOR_1_X_COLUMN, EIGENVECTOR_1_Y_COLUMN, EIGENVECTOR_1_Z_COLUMN,
                    EIGENVECTOR_2_X_COLUMN, EIGENVECTOR_2_Y_COLUMN, EIGENVECTOR_2_Z_COLUMN, -> Normal vector
                    LLSV_COLUMN, -> lowest local surface variation
                    SEGMENT_ID_COLUMN,
                    STANDARD_DEVIATION_COLUMN,
                    NR_POINTS_PER_SEG_COLUMN,

            'extended y' has the following structure: (tuples of index segment from epoch0, index segment from epoch1,
            label(0/1)) used for learning.
        """

        if not interactive_available:
            logger.error("Interactive session not available in this environment.")
            return

        labeling_pipeline = Pipeline(
            [
                ("Transform_PerPointComputation", self._per_point_computation),
                ("Transform_Segmentation", self._segmentation),
                ("Transform_Second_Segmentation", self._second_segmentation),
                ("Transform_ExtractSegments", self._extract_segments),
            ]
        )

        # print the default parameters
        PBM3C2._print_default_parameters(
            kwargs=kwargs, pipeline_param_dict=labeling_pipeline.get_params()
        )

        # no computation
        if epoch0 is None or epoch1 is None:
            # logger.info("epoch0 and epoch1 are required, no parameter changes applied")
            return

        # save the default pipeline options
        default_options = labeling_pipeline.get_params()

        # overwrite the default parameters
        unused_kwargs = PBM3C2._overwrite_pipeline_parameters(
            kwargs=kwargs, pipeline=labeling_pipeline
        )
        if len(unused_kwargs) > 0:
            logger.warning(
                f"The parameters '{unused_kwargs.keys()}' are not part of the pipeline parameters: \n "
                f"{pp.pformat(labeling_pipeline.get_params())}"
            )

        # apply the pipeline
        X0 = np.hstack((epoch0.cloud[:, :], np.zeros((epoch0.cloud.shape[0], 1))))
        X1 = np.hstack((epoch1.cloud[:, :], np.ones((epoch1.cloud.shape[0], 1))))
        X = np.vstack((X0, X1))
        labeling_pipeline.fit(X)
        segments = labeling_pipeline.transform(X)

        # restore the default pipeline options
        unused_kwargs = PBM3C2._overwrite_pipeline_parameters(
            kwargs=default_options,
            pipeline=labeling_pipeline,
            message="The pipeline parameters after restoration are: ",
        )
        assert (
            len(unused_kwargs) == 0
        ), "All default options should be found when default parameter restoration is done"

        return segments, builder_extended_y.generate_extended_y(segments)

    def export_segmented_point_cloud_and_segments(
        self,
        epoch0: Epoch = None,
        epoch1: Epoch = None,
        x_y_z_id_epoch0_file_name: typing.Union[str, None] = "x_y_z_id_epoch0.xyz",
        x_y_z_id_epoch1_file_name: typing.Union[str, None] = "x_y_z_id_epoch1.xyz",
        extracted_segments_file_name: typing.Union[
            str, None
        ] = "extracted_segments.seg",
        concatenate_name="",
        **kwargs,
    ) -> typing.Union[
        typing.Tuple[np.ndarray, typing.Union[np.ndarray, None], np.ndarray], None
    ]:
        """
        For each epoch, it returns the segmentation of the point cloud as a numpy array (n_points, 4)
        and it also serializes them using the provided file names.
        where each row has the following structure: x, y, z, segment_id

        It also generates a numpy array of segments of the form:
                    X_COLUMN, Y_COLUMN, Z_COLUMN, -> Center of Gravity
                    EPOCH_ID_COLUMN, -> 0/1
                    EIGENVALUE0_COLUMN, EIGENVALUE1_COLUMN, EIGENVALUE2_COLUMN,
                    EIGENVECTOR_0_X_COLUMN, EIGENVECTOR_0_Y_COLUMN, EIGENVECTOR_0_Z_COLUMN,
                    EIGENVECTOR_1_X_COLUMN, EIGENVECTOR_1_Y_COLUMN, EIGENVECTOR_1_Z_COLUMN,
                    EIGENVECTOR_2_X_COLUMN, EIGENVECTOR_2_Y_COLUMN, EIGENVECTOR_2_Z_COLUMN, -> Normal vector
                    LLSV_COLUMN, -> lowest local surface variation
                    SEGMENT_ID_COLUMN,
                    STANDARD_DEVIATION_COLUMN,
                    NR_POINTS_PER_SEG_COLUMN,

        :param epoch0:
            Epoch object | None
        :param epoch1:
            Epoch object | None
        :param x_y_z_id_epoch0_file_name:
            The output file name for epoch0, point cloud segmentation, saved as a numpy array (n_points, 4)
            (x,y,z, segment_id)
            | None
        :param x_y_z_id_epoch1_file_name:
            The output file name for epoch1, point cloud segmentation, saved as a numpy array (n_points, 4)
            (x,y,z, segment_id)
            | None
        :param extracted_segments_file_name:
            The output file name for the file containing the segments, saved as a numpy array containing
            the column structure introduced above.
            | None
        :param concatenate_name:
            String that is utilized to uniquely identify the same transformer between multiple pipelines.
        :param kwargs:

            Used for customize the default pipeline parameters.

            Getting the default parameters:
            e.g. "get_pipeline_options"
                In case this parameter is True, the method will print the pipeline options as kwargs.

            e.g. "output_file_name" (of a specific step in the pipeline) default value is "None".
                In case of setting it, the result of computation at that step is dump as xyz file.
            e.g. "distance_3D_threshold" (part of Segmentation Transform)

            this process is stateless

        :return:
            tuple [ x_y_z_id_epoch0, x_y_z_id_epoch1 | None, extracted_segments ] | None
        """

        pipe_segmentation = Pipeline(
            [
                (
                    concatenate_name + "_Transform_PerPointComputation",
                    self._per_point_computation,
                ),
                (concatenate_name + "_Transform_Segmentation", self._segmentation),
                (
                    concatenate_name + "_Transform_Second_Segmentation",
                    self._second_segmentation,
                ),
            ]
        )

        pipe_extract_segments = Pipeline(
            [
                (
                    concatenate_name + "_Transform_ExtractSegments",
                    self._extract_segments,
                ),
            ]
        )

        # print the default parameters
        PBM3C2._print_default_parameters(
            kwargs=kwargs,
            pipeline_param_dict={
                **pipe_segmentation.get_params(),
                **pipe_extract_segments.get_params(),
            },
        )

        # no computation
        if epoch0 is None:  # or epoch1 is None:
            # logger.info("epoch0 is required, no parameter changes applied")
            return

        # save the default pipeline options
        default_options = {
            **pipe_segmentation.get_params(),
            **pipe_extract_segments.get_params(),
        }
        del default_options["memory"]
        del default_options["steps"]
        del default_options["verbose"]

        # overwrite the default parameters
        unused_kwargs = PBM3C2._overwrite_pipeline_parameters(
            kwargs=kwargs, pipeline=pipe_segmentation
        )
        unused_kwargs = PBM3C2._overwrite_pipeline_parameters(
            kwargs=unused_kwargs, pipeline=pipe_extract_segments
        )
        if len(unused_kwargs) > 0:
            logger.warning(
                f"The parameters '{unused_kwargs.keys()}' are not part of the pipeline parameters: \n "
                f"{pp.pformat({**pipe_segmentation.get_params(), **pipe_extract_segments.get_params()})}"
            )

        # apply the pipeline

        if isinstance(epoch0, Epoch):
            X0 = np.hstack((epoch0.cloud[:, :], np.zeros((epoch0.cloud.shape[0], 1))))
        else:
            X0 = epoch0

        if isinstance(epoch1, Epoch):
            X1 = np.hstack((epoch1.cloud[:, :], np.ones((epoch1.cloud.shape[0], 1))))
            X = np.vstack((X0, X1))
        else:
            X = X0

        pipe_segmentation.fit(X)
        # 'out' contains the segmentation of the point cloud.
        out = pipe_segmentation.transform(X)

        pipe_extract_segments.fit(out)
        # 'extracted_segments' contains the new set of segments
        extracted_segments = pipe_extract_segments.transform(out)

        # restore the default pipeline options
        unused_default_options = PBM3C2._overwrite_pipeline_parameters(
            kwargs=default_options,
            pipeline=pipe_segmentation,
            message="The pipeline parameters after restoration are: ",
        )
        unused_default_options = PBM3C2._overwrite_pipeline_parameters(
            kwargs=unused_default_options,
            pipeline=pipe_extract_segments,
            message="The pipeline parameters after restoration are: ",
        )
        assert (
            len(unused_default_options) == 0
        ), "All default options should be found when default parameter restoration is done"

        columns = self._second_segmentation.columns

        Extract_Columns = [
            columns.X_COLUMN,
            columns.Y_COLUMN,
            columns.Z_COLUMN,
            columns.SEGMENT_ID_COLUMN,
        ]

        mask_epoch0 = out[:, columns.EPOCH_ID_COLUMN] == 0
        mask_epoch1 = out[:, columns.EPOCH_ID_COLUMN] == 1

        out_epoch0 = out[mask_epoch0, :]
        out_epoch1 = out[mask_epoch1, :]

        x_y_z_id_epoch0 = out_epoch0[:, Extract_Columns]  # x,y,z, Seg_ID
        x_y_z_id_epoch1 = out_epoch1[:, Extract_Columns]  # x,y,z, Seg_ID

        if x_y_z_id_epoch0_file_name != None:
            logger.debug(f"Save 'x_y_z_id_epoch0' in file: {x_y_z_id_epoch0_file_name}")
            np.savetxt(x_y_z_id_epoch0_file_name, x_y_z_id_epoch0, delimiter=",")
        else:
            logger.debug(f"'x_y_z_id_epoch0' is not saved")

        if x_y_z_id_epoch1_file_name != None:
            logger.debug(f"Save 'x_y_z_id_epoch1' in file: {x_y_z_id_epoch1_file_name}")
            np.savetxt(x_y_z_id_epoch1_file_name, x_y_z_id_epoch1, delimiter=",")
        else:
            logger.debug(f"'x_y_z_id_epoch1' is not saved")

        if extracted_segments_file_name != None:
            logger.debug(
                f"Save 'extracted_segments' in file: {extracted_segments_file_name}"
            )
            np.savetxt(extracted_segments_file_name, extracted_segments, delimiter=",")
        else:
            logger.debug(f"'extracted_segments' is not saved")

        return x_y_z_id_epoch0, x_y_z_id_epoch1, extracted_segments

    def training(
        self,
        segments: np.ndarray = None,
        extended_y: np.ndarray = None,
        extracted_segments_file_name: str = "extracted_segments.seg",
        extended_y_file_name: str = "extended_y.csv",
    ) -> None:
        """
        It applies the training algorithm for the input pairs of Segments 'segments'
        and extended labels 'extended_y'.

        :param segments:
            'Segments' numpy array of shape (n_segments, segment_size)

            It has the following column structure:
                    X_COLUMN, Y_COLUMN, Z_COLUMN, -> Center of Gravity
                    EPOCH_ID_COLUMN, -> 0/1
                    EIGENVALUE0_COLUMN, EIGENVALUE1_COLUMN, EIGENVALUE2_COLUMN,
                    EIGENVECTOR_0_X_COLUMN, EIGENVECTOR_0_Y_COLUMN, EIGENVECTOR_0_Z_COLUMN,
                    EIGENVECTOR_1_X_COLUMN, EIGENVECTOR_1_Y_COLUMN, EIGENVECTOR_1_Z_COLUMN,
                    EIGENVECTOR_2_X_COLUMN, EIGENVECTOR_2_Y_COLUMN, EIGENVECTOR_2_Z_COLUMN, -> Normal vector
                    LLSV_COLUMN, -> lowest local surface variation
                    SEGMENT_ID_COLUMN,
                    STANDARD_DEVIATION_COLUMN,
                    NR_POINTS_PER_SEG_COLUMN,

        :param extended_y:
            numpy array of shape (m_labels, 3)
            has the following structure: (tuples of index segment from epoch0, index segment from epoch1,
            label(0/1))
        :param extracted_segments_file_name:
            In case 'X' is None segments are loaded using 'extracted_segments_file_name'.
        :param extended_y_file_name:
            In case 'extended_y' is None, this file is used as input fallback.
        """

        if segments is None:
            # Resolve the given path
            filename = find_file(extracted_segments_file_name)
            # Read it
            try:
                logger.info(f"Reading segments from file '{filename}'")
                segments = np.genfromtxt(filename, delimiter=",")
            except ValueError:
                raise Py4DGeoError("Malformed file: " + str(filename))

        if extended_y is None:
            # Resolve the given path
            filename = find_file(extended_y_file_name)
            # Read it
            try:
                logger.info(
                    f"Reading tuples of (segment epoch0, segment epoch1, label) from file '{filename}'"
                )
                extended_y = np.genfromtxt(filename, delimiter=",")
            except ValueError:
                raise Py4DGeoError("Malformed file: " + str(filename))

        training_pipeline = Pipeline(
            [
                ("Classifier", self._classifier),
            ]
        )

        # apply the pipeline
        training_pipeline.fit(segments, extended_y)

    def predict(
        self,
        epoch0: Epoch = None,
        epoch1: Epoch = None,
        epoch_additional_dimensions_lookup: typing.Dict[str, str] = None,
        **kwargs,
    ) -> typing.Union[np.ndarray, None]:
        """
        After extracting the segments from epoch0 and epoch1, it returns a numpy array of corresponding
        pairs of segments between epoch 0 and epoch 1.

        :param epoch0:
            Epoch object.
        :param epoch1:
            Epoch object.
        :param epoch_additional_dimensions_lookup:
            A dictionary that maps between the names of the columns used internally
            and the names of the columns used by both epoch0 and epoch1.

            No additional column is used.
        :param kwargs:

            Used for customize the default pipeline parameters.

            Getting the default parameters:
            e.g. "get_pipeline_options"
                In case this parameter is True, the method will print the pipeline options as kwargs.

            e.g. "output_file_name" (of a specific step in the pipeline) default value is "None".
                In case of setting it, the result of computation at that step is dump as xyz file.
            e.g. "distance_3D_threshold" (part of Segmentation Transform)

            this process is stateless

        :return:
            A numpy array ( n_pairs, segment_features_size*2 ) where each row contains a pair of segments.
            | None
        """

        pipe_classifier = Pipeline(
            [
                ("Classifier", self._classifier),
            ]
        )

        # arguments used, as part of the epoch0's pipeline.
        kwargs_epoch0 = {
            key: val
            for key, val in kwargs.items()
            if key.startswith("epoch0") or key == "get_pipeline_options"
        }

        # epoch0, segments computation
        out_for_epoch0 = self.export_segmented_point_cloud_and_segments(
            epoch0=epoch0,
            epoch1=None,
            x_y_z_id_epoch0_file_name=None,
            x_y_z_id_epoch1_file_name=None,
            extracted_segments_file_name=None,
            concatenate_name="epoch0",
            **kwargs_epoch0,
        )

        # arguments used, as part of the epoch1's pipeline.
        kwargs_epoch1 = {
            key: val
            for key, val in kwargs.items()
            if key.startswith("epoch1") or key == "get_pipeline_options"
        }

        # epoch1, segments computation
        out_for_epoch1 = self.export_segmented_point_cloud_and_segments(
            epoch0=epoch1,
            epoch1=None,
            x_y_z_id_epoch0_file_name=None,
            x_y_z_id_epoch1_file_name=None,
            extracted_segments_file_name=None,
            concatenate_name="epoch1",
            **kwargs_epoch1,
        )

        # arguments used, as part of the classifier 'pipeline'
        kwargs_classifier = {
            key: val
            for key, val in kwargs.items()
            if not key.startswith("epoch0") and not key.startswith("epoch1")
        }

        # print the default parameters for 'pipe_classifier'
        PBM3C2._print_default_parameters(
            kwargs=kwargs_classifier, pipeline_param_dict=pipe_classifier.get_params()
        )

        # no computation
        if epoch0 is None or epoch1 is None:
            # logger.info("epoch0 and epoch1 are required, no parameter changes applied")
            return

        # save the default pipe_classifier options
        default_options = pipe_classifier.get_params()

        # overwrite the default parameters
        unused_kwargs = PBM3C2._overwrite_pipeline_parameters(
            kwargs=kwargs_classifier, pipeline=pipe_classifier
        )
        if len(unused_kwargs) > 0:
            logger.error(
                f"The parameters: '{unused_kwargs.keys()}' are not part of the pipeline parameters: \n "
                f"{pp.pformat(pipe_classifier.get_params())}"
            )

        _0, _1, epoch0_segments = out_for_epoch0
        _0, _1, epoch1_segments = out_for_epoch1

        columns = self._extract_segments.columns

        # compute segment id offset
        max_seg_id_epoch0 = np.max(epoch0_segments[:, columns.SEGMENT_ID_COLUMN])
        # apply offset to segment from epoch1
        epoch1_segments[:, columns.SEGMENT_ID_COLUMN] += max_seg_id_epoch0 + 1

        # enforce epoch ID as '0' for 'epoch0_segments'
        epoch0_segments[:, columns.EPOCH_ID_COLUMN] = 0

        # change the epoch ID from 0 (the default one used during the previous computation for both epochs)
        # to 1 for 'epoch1_segments'
        epoch1_segments[:, columns.EPOCH_ID_COLUMN] = 1

        extracted_segments = np.vstack((epoch0_segments, epoch1_segments))

        out = pipe_classifier.predict(extracted_segments)

        # restore the default pipeline options
        unused_default_options = PBM3C2._overwrite_pipeline_parameters(
            kwargs=default_options,
            pipeline=pipe_classifier,
            message="The pipeline parameters after restoration are: ",
        )
        assert (
            len(unused_default_options) == 0
        ), "All default options should be found when default parameter restoration is done"

        return out

    def _compute_distances(
        self,
        epoch0_info: typing.Union[Epoch, np.ndarray] = None,
        epoch1: Epoch = None,
        alignment_error: float = 1.1,
        epoch_additional_dimensions_lookup: typing.Dict[str, str] = None,
        **kwargs,
    ) -> typing.Union[typing.Tuple[np.ndarray, np.ndarray], None]:
        """
        Compute the distance between 2 epochs. It also adds the following properties at the end of the computation:
            distances, corepoints (corepoints of epoch0), epochs (epoch0, epoch1), uncertainties

        :param epoch0_info:
            Epoch object.
            | np.ndarray
        :param epoch1:
            Epoch object.
        :param alignment_error:
            alignment error reg between point clouds.
        :param epoch_additional_dimensions_lookup:
            A dictionary that maps between the names of the columns used internally
            and the names of the columns used by both epoch0 and epoch1.
        :param kwargs:
            Used for customize the default pipeline parameters.

            Getting the default parameters:
            e.g. "get_pipeline_options"
                In case this parameter is True, the method will print the pipeline options as kwargs.

            e.g. "output_file_name" (of a specific step in the pipeline) default value is "None".
                In case of setting it, the result of computation at that step is dump as xyz file.
            e.g. "distance_3D_threshold" (part of Segmentation Transform)

            this process is stateless

        :return:
            tuple [distances, uncertainties]
                'distances' is np.array (nr_similar_pairs, 1)
                'uncertainties' is np.array (nr_similar_pairs,1) and it has the following structure:
                    dtype=np.dtype(
                        [
                            ("lodetection", "<f8"),
                            ("spread1", "<f8"),
                            ("num_samples1", "<i8"),
                            ("spread2", "<f8"),
                            ("num_samples2", "<i8"),
                        ]
                    )
            | None
        """

        logger.info(f"PBM3C2._compute_distances(...)")

        # A numpy array where each row contains a pair of segments.
        segments_pair = self.predict(
            epoch0_info,
            epoch1,
            epoch_additional_dimensions_lookup=epoch_additional_dimensions_lookup,
            **kwargs,
        )

        columns = self._extract_segments.columns

        # no computation
        if epoch0_info is None or epoch1 is None:
            # logger.info("epoch0 and epoch1 are required, no parameter changes applied")
            return

        size_segment = int(segments_pair.shape[1] / 2)
        nr_pairs = segments_pair.shape[0]

        epoch0_segments = segments_pair[:, :size_segment]
        epoch1_segments = segments_pair[:, size_segment:]

        # seg_id_epoch0, X_Column0, Y_Column0, Z_Column0, seg_id_epoch1, X_Column1, Y_Column1, Z_Column1, distance, uncertaintie
        output = np.empty((0, 10), dtype=float)

        for indx in range(nr_pairs):
            segment_epoch0 = epoch0_segments[indx]
            segment_epoch1 = epoch1_segments[indx]

            t0_CoG = segment_epoch0[
                [columns.X_COLUMN, columns.Y_COLUMN, columns.Z_COLUMN]
            ]
            t1_CoG = segment_epoch1[
                [columns.X_COLUMN, columns.Y_COLUMN, columns.Z_COLUMN]
            ]

            Normal_Columns = [
                columns.EIGENVECTOR_2_X_COLUMN,
                columns.EIGENVECTOR_2_Y_COLUMN,
                columns.EIGENVECTOR_2_Z_COLUMN,
            ]
            normal_vector_t0 = segment_epoch0[Normal_Columns]

            M3C2_dist = normal_vector_t0.dot(t0_CoG - t1_CoG)

            std_dev_normalized_squared_t0 = segment_epoch0[
                columns.STANDARD_DEVIATION_COLUMN
            ]
            std_dev_normalized_squared_t1 = segment_epoch1[
                columns.STANDARD_DEVIATION_COLUMN
            ]

            LoDetection = 1.96 * (
                np.sqrt(std_dev_normalized_squared_t0 + std_dev_normalized_squared_t1)
                + alignment_error
            )

            # seg_id_epoch0, X_Column0, Y_Column0, Z_Column0, seg_id_epoch1, X_Column1, Y_Column1, Z_Column1, distance, uncertaintie
            args = (
                np.array([segment_epoch0[columns.SEGMENT_ID_COLUMN]]),
                t0_CoG,
                np.array([segment_epoch1[columns.SEGMENT_ID_COLUMN]]),
                t1_CoG,
                np.array([M3C2_dist]),
                np.array([LoDetection]),
            )
            row = np.concatenate(args)
            output = np.vstack((output, row))

        # We don't return this anymore. It corresponds to the output structure of the original implementation.
        # return output

        # distance vector
        self.distances = output[:, -2]

        # corepoints to epoch0 ('initial' one)
        self.corepoints = Epoch(output[:, [1, 2, 3]])

        # epochs
        self.epochs = (epoch0_info, epoch1)

        self.uncertainties = np.empty(
            (output.shape[0], 1),
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

        self.uncertainties["lodetection"] = output[:, -1].reshape(-1, 1)

        self.uncertainties["spread1"] = np.sqrt(
            np.multiply(
                epoch0_segments[:, columns.STANDARD_DEVIATION_COLUMN],
                epoch0_segments[:, columns.NR_POINTS_PER_SEG_COLUMN],
            )
        ).reshape(-1, 1)
        self.uncertainties["spread2"] = np.sqrt(
            np.multiply(
                epoch1_segments[:, columns.STANDARD_DEVIATION_COLUMN],
                epoch1_segments[:, columns.NR_POINTS_PER_SEG_COLUMN],
            )
        ).reshape(-1, 1)

        self.uncertainties["num_samples1"] = (
            epoch0_segments[:, columns.NR_POINTS_PER_SEG_COLUMN]
            .astype(int)
            .reshape(-1, 1)
        )
        self.uncertainties["num_samples2"] = (
            epoch1_segments[:, columns.NR_POINTS_PER_SEG_COLUMN]
            .astype(int)
            .reshape(-1, 1)
        )

        return (self.distances, self.uncertainties)

    def compute_distances(
        self,
        epoch0: typing.Union[Epoch, np.ndarray] = None,
        epoch1: Epoch = None,
        alignment_error: float = 1.1,
        **kwargs,
    ) -> typing.Union[typing.Tuple[np.ndarray, np.ndarray], None]:
        """
        Compute the distance between 2 epochs. It also adds the following properties at the end of the computation:
            distances, corepoints (corepoints of epoch0), epochs (epoch0, epoch1), uncertainties

        :param epoch0:
            Epoch object.
            | np.ndarray
        :param epoch1:
            Epoch object.
        :param alignment_error:
            alignment error reg between point clouds.
        :param kwargs:
            Used for customize the default pipeline parameters.

            Getting the default parameters:
            e.g. "get_pipeline_options"
                In case this parameter is True, the method will print the pipeline options as kwargs.

            e.g. "output_file_name" (of a specific step in the pipeline) default value is "None".
                In case of setting it, the result of computation at that step is dump as xyz file.
            e.g. "distance_3D_threshold" (part of Segmentation Transform)

            this process is stateless

        :return:
            tuple [distances, uncertainties]
                'distances' is np.array (nr_similar_pairs, 1)
                'uncertainties' is np.array (nr_similar_pairs,1) and it has the following structure:
                    dtype=np.dtype(
                        [
                            ("lodetection", "<f8"),
                            ("spread1", "<f8"),
                            ("num_samples1", "<i8"),
                            ("spread2", "<f8"),
                            ("num_samples2", "<i8"),
                        ]
                    )
            | None
        """

        logger.info(f"PBM3C2.compute_distances(...)")

        return self._compute_distances(
            epoch0_info=epoch0,
            epoch1=epoch1,
            alignment_error=alignment_error,
            epoch_additional_dimensions_lookup=None,
            **kwargs,
        )


def build_input_scenario2_with_normals(
    epoch0, epoch1, columns=SEGMENTED_POINT_CLOUD_COLUMNS
):
    """
    Build a segmented point cloud with computed normals for each point.

    :param epoch0:
        x,y,z point cloud
    :param epoch1:
        x,y,z point cloud
    :param columns:

    :return:
        tuple [
                numpy array (n_point_samples, 7) new_epoch0
                numpy array (m_point_samples, 7) new_epoch1
        ]
        both containing: x,y,z, N_x,N_y,N_z, Segment_ID as columns.
    """

    Normal_Columns = [
        columns.EIGENVECTOR_2_X_COLUMN,
        columns.EIGENVECTOR_2_Y_COLUMN,
        columns.EIGENVECTOR_2_Z_COLUMN,
    ]

    x_y_z_Columns = [columns.X_COLUMN, columns.Y_COLUMN, columns.Z_COLUMN]

    X0 = np.hstack((epoch0.cloud[:, :], np.zeros((epoch0.cloud.shape[0], 1))))
    X1 = np.hstack((epoch1.cloud[:, :], np.ones((epoch1.cloud.shape[0], 1))))

    X = np.vstack((X0, X1))

    transform_pipeline = Pipeline(
        [
            ("Transform_PerPointComputation", PerPointComputation()),
            ("Transform_Segmentation", Segmentation()),
        ]
    )

    transform_pipeline.fit(X)
    out = transform_pipeline.transform(X)

    mask_epoch0 = out[:, columns.EPOCH_ID_COLUMN] == 0  # epoch0
    mask_epoch1 = out[:, columns.EPOCH_ID_COLUMN] == 1  # epoch1

    new_epoch0 = out[mask_epoch0, :]  # extract epoch0
    new_epoch1 = out[mask_epoch1, :]  # extract epoch1

    new_epoch0 = new_epoch0[
        :, x_y_z_Columns + Normal_Columns + [columns.SEGMENT_ID_COLUMN]
    ]
    new_epoch1 = new_epoch1[
        :, x_y_z_Columns + Normal_Columns + [columns.SEGMENT_ID_COLUMN]
    ]

    # x,y,z, N_x,N_y,N_z, Segment_ID
    return new_epoch0, new_epoch1
    pass


def build_input_scenario2_without_normals(
    epoch0, epoch1, columns=SEGMENTED_POINT_CLOUD_COLUMNS
):
    """
        Build a segmented point cloud.

    :param epoch0:
        x,y,z point cloud
    :param epoch1:
        x,y,z point cloud
    :param columns:

    :return:
        tuple [
            numpy array (n_point_samples, 4) new_epoch0
            numpy array (m_point_samples, 4) new_epoch1
        ]
        both containing: x,y,z,Segment_ID as columns.
    """

    x_y_z_Columns = [columns.X_COLUMN, columns.Y_COLUMN, columns.Z_COLUMN]

    Normal_Columns = [
        columns.EIGENVECTOR_2_X_COLUMN,
        columns.EIGENVECTOR_2_Y_COLUMN,
        columns.EIGENVECTOR_2_Z_COLUMN,
    ]

    X0 = np.hstack((epoch0.cloud[:, :], np.zeros((epoch0.cloud.shape[0], 1))))
    X1 = np.hstack((epoch1.cloud[:, :], np.ones((epoch1.cloud.shape[0], 1))))

    X = np.vstack((X0, X1))

    transform_pipeline = Pipeline(
        [
            ("Transform_PerPointComputation", PerPointComputation()),
            ("Transform_Segmentation", Segmentation()),
        ]
    )

    transform_pipeline.fit(X)
    out = transform_pipeline.transform(X)

    mask_epoch0 = out[:, columns.EPOCH_ID_COLUMN] == 0  # epoch0
    mask_epoch1 = out[:, columns.EPOCH_ID_COLUMN] == 1  # epoch1

    new_epoch0 = out[mask_epoch0, :]  # extract epoch0
    new_epoch1 = out[mask_epoch1, :]  # extract epoch1

    new_epoch0 = new_epoch0[:, x_y_z_Columns + [columns.SEGMENT_ID_COLUMN]]
    new_epoch1 = new_epoch1[:, x_y_z_Columns + [columns.SEGMENT_ID_COLUMN]]

    # x,y,z, Segment_ID
    return new_epoch0, new_epoch1
    pass


class PBM3C2WithSegments(PBM3C2):
    def __init__(
        self,
        per_point_computation=PerPointComputation(),
        segmentation=Segmentation(),
        second_segmentation=Segmentation(),
        extract_segments=ExtractSegments(),
        post_segmentation=PostPointCloudSegmentation(compute_normal=True),
        classifier=ClassifierWrapper(),
    ):
        """
        :param per_point_computation:
            lowest local surface variation and PCA computation. (computes the normal vector as well)
        :param segmentation:
            The object used for the first segmentation.
        :param second_segmentation:
            The object used for the second segmentation.
        :param extract_segments:
            The object used for building the segments.
        :param post_segmentation:
            A transform object used to 'reconstruct' the result that is achieved using the "PB_P3C2 class"
            pipeline at the end of the point cloud segmentation.

            The 'input' of this adaptor is formed by 2 epochs that contain as 'additional_dimensions'
            a segment_id column and optionally, precomputed normals as another 3 columns.

            The 'output' of this adaptor is:
            numpy array (n_point_samples, 19) with the following column structure:
            [
                x, y, z ( 3 columns ),
                EpochID ( 1 column ),
                Eigenvalues( 3 columns ), -> that correspond to the next 3 Eigenvectors
                Eigenvectors( 3 columns ) X 3 -> in descending order using vector norm 2,
                Lowest local surface variation ( 1 column ),
                Segment_ID ( 1 column ),
                Standard deviation ( 1 column )
            ]
        :param classifier:
            An instance of ClassifierWrapper class. The default wrapped classifier used is sk-learn RandomForest.
        """

        super().__init__(
            per_point_computation=per_point_computation,
            segmentation=segmentation,
            second_segmentation=second_segmentation,
            extract_segments=extract_segments,
            classifier=classifier,
        )

        self._post_segmentation = post_segmentation

    def generate_extended_labels_interactively(
        self,
        epoch0: Epoch = None,
        epoch1: Epoch = None,
        builder_extended_y: BuilderExtended_y_Visually = BuilderExtended_y_Visually(),
        epoch_additional_dimensions_lookup: typing.Dict[str, str] = dict(
            segment_id="segment_id", N_x="N_x", N_y="N_y", N_z="N_z"
        ),
        **kwargs,
    ) -> typing.Union[typing.Tuple[np.ndarray, np.ndarray], None]:
        """
        Given 2 Epochs, it builds a pair of (segments and 'extended y').

        :param epoch0:
            Epoch object,
            contains as 'additional_dimensions' a segment_id column (mandatory)
            and optionally, precomputed normals as another 3 columns.
        :param epoch1:
            Epoch object,
            contains as 'additional_dimensions' a segment_id column (mandatory)
            and optionally, precomputed normals as another 3 columns.
        :param builder_extended_y:
            The object is used for generating 'extended y', visually.
        :param epoch_additional_dimensions_lookup:
            A dictionary that maps between the names of the columns used internally to identify:
                segment id of the points: "segment_id"  -> Mandatory part of the epochs
                Normal x-axes vector: "N_x"             -> Optionally part of the epochs
                Normal y-axes vector: "N_y"             -> Optionally part of the epochs
                Normal z-axes vector: "N_z"             -> Optionally part of the epochs
            and the names of the columns used by both epoch0 and epoch1.
        :param kwargs:

            Used for customize the default pipeline parameters.

            Getting the default parameters:
            e.g. "get_pipeline_options"
                In case this parameter is True, the method will print the pipeline options as kwargs.

            e.g. "output_file_name" (of a specific step in the pipeline) default value is "None".
                In case of setting it, the result of computation at that step is dump as xyz file.
            e.g. "distance_3D_threshold" (part of Segmentation Transform)

            this process is stateless

        :return:
            tuple [Segments, 'extended y'] | None

            where:

            'Segments' has the following column structure:
                    X_COLUMN, Y_COLUMN, Z_COLUMN, -> Center of Gravity
                    EPOCH_ID_COLUMN, -> 0/1
                    EIGENVALUE0_COLUMN, EIGENVALUE1_COLUMN, EIGENVALUE2_COLUMN,
                    EIGENVECTOR_0_X_COLUMN, EIGENVECTOR_0_Y_COLUMN, EIGENVECTOR_0_Z_COLUMN,
                    EIGENVECTOR_1_X_COLUMN, EIGENVECTOR_1_Y_COLUMN, EIGENVECTOR_1_Z_COLUMN,
                    EIGENVECTOR_2_X_COLUMN, EIGENVECTOR_2_Y_COLUMN, EIGENVECTOR_2_Z_COLUMN, -> Normal vector
                    LLSV_COLUMN, -> lowest local surface variation
                    SEGMENT_ID_COLUMN,
                    STANDARD_DEVIATION_COLUMN,
                    NR_POINTS_PER_SEG_COLUMN,

            'extended y' has the following structure: (tuples of index segment from epoch0, index segment from epoch1,
            label(0/1)) used for learning.
        """

        if not interactive_available:
            logger.error("Interactive session not available in this environment.")
            return

        transform_pipeline = Pipeline(
            [
                ("Transform_Post_Segmentation", self._post_segmentation),
                ("Transform_ExtractSegments", self._extract_segments),
            ]
        )

        # print the default parameters
        PBM3C2._print_default_parameters(
            kwargs=kwargs, pipeline_param_dict=transform_pipeline.get_params()
        )

        # no computation
        if epoch0 is None or epoch1 is None:
            # logger.info("epoch0 and epoch1 are required, no parameter changes applied")
            return

        # save the default pipeline options
        default_options = transform_pipeline.get_params()

        # overwrite the default parameters
        PBM3C2._overwrite_pipeline_parameters(
            kwargs=kwargs, pipeline=transform_pipeline
        )

        # extract columns

        epoch0_normals = _extract_from_additional_dimensions(
            epoch=epoch0,
            column_names=[
                epoch_additional_dimensions_lookup["N_x"],
                epoch_additional_dimensions_lookup["N_y"],
                epoch_additional_dimensions_lookup["N_z"],
            ],
            required_number_of_columns=[0, 3],
        )

        epoch0_segment_id = _extract_from_additional_dimensions(
            epoch=epoch0,
            column_names=[
                epoch_additional_dimensions_lookup["segment_id"],
            ],
            required_number_of_columns=[1],
        )

        epoch0 = np.concatenate(
            (epoch0.cloud, epoch0_normals, epoch0_segment_id),
            axis=1,
        )

        epoch1_normals = _extract_from_additional_dimensions(
            epoch=epoch1,
            column_names=[
                epoch_additional_dimensions_lookup["N_x"],
                epoch_additional_dimensions_lookup["N_y"],
                epoch_additional_dimensions_lookup["N_z"],
            ],
            required_number_of_columns=[0, 3],
        )

        epoch1_segment_id = _extract_from_additional_dimensions(
            epoch=epoch1,
            column_names=[
                epoch_additional_dimensions_lookup["segment_id"],
            ],
            required_number_of_columns=[1],
        )

        epoch1 = np.concatenate(
            (epoch1.cloud, epoch1_normals, epoch1_segment_id),
            axis=1,
        )

        X = None
        for epoch_id, current_epoch in enumerate([epoch0, epoch1]):
            if current_epoch.shape[1] == 4:
                # [x, y, z, segment_id] columns
                assert self._post_segmentation.compute_normal, (
                    "The reconstruction process doesn't have, as input, the Normal vector columns, hence, "
                    "the normal vector computation is mandatory."
                )

                logger.info(
                    f"Reconstruct post segmentation output using [x, y, z, segment_id] "
                    f"columns from epoch{epoch_id} "
                )

                if X is not None:
                    X = np.vstack(
                        (
                            X,
                            self._reconstruct_input_without_normals(
                                epoch=current_epoch,
                                epoch_id=epoch_id,
                                columns=self._post_segmentation.columns,
                            ),
                        )
                    )
                else:
                    X = self._reconstruct_input_without_normals(
                        epoch=current_epoch,
                        epoch_id=epoch_id,
                        columns=self._post_segmentation.columns,
                    )
            else:
                # [x, y, z, N_x, N_y, N_z, segment_id] columns
                logger.info(
                    f"Reconstruct post segmentation output using [x, y, z, N_x, N_y, N_z, segment_id] "
                    f"columns from epoch{epoch_id}"
                )

                if X is not None:
                    X = np.vstack(
                        (
                            X,
                            self._reconstruct_input_with_normals(
                                epoch=current_epoch,
                                epoch_id=epoch_id,
                                columns=self._post_segmentation.columns,
                            ),
                        )
                    )
                else:
                    X = self._reconstruct_input_with_normals(
                        epoch=current_epoch,
                        epoch_id=epoch_id,
                        columns=self._post_segmentation.columns,
                    )

        # apply the pipeline

        transform_pipeline.fit(X)
        segments = transform_pipeline.transform(X)

        # restore the default pipeline options
        unused_kwargs = PBM3C2._overwrite_pipeline_parameters(
            kwargs=default_options,
            pipeline=transform_pipeline,
            message="The pipeline parameters after restoration are: ",
        )
        assert (
            len(unused_kwargs) == 0
        ), "All default options should be found when default parameter restoration is done"

        return segments, builder_extended_y.generate_extended_y(segments)

    def reconstruct_post_segmentation_output(
        self,
        epoch0: Epoch = None,
        epoch1: Epoch = None,
        extracted_segments_file_name: typing.Union[
            str, None
        ] = "extracted_segments.seg",
        epoch_additional_dimensions_lookup: typing.Dict[str, str] = dict(
            segment_id="segment_id", N_x="N_x", N_y="N_y", N_z="N_z"
        ),
        concatenate_name: str = "",
        **kwargs,
    ) -> typing.Union[
        typing.Tuple[np.ndarray, typing.Union[np.ndarray, None], np.ndarray], None
    ]:
        """
        'reconstruct' the result that is achieved using the "PB_P3C2 class" pipeline, by applying
            ("Transform LLSV_and_PCA"), ("Transform Segmentation"),
            ("Transform Second Segmentation") ("Transform ExtractSegments")
        using, as input, segmented point clouds.

        :param epoch0:
            Epoch object,
            contains as 'additional_dimensions' a segment_id column (mandatory)
            and optionally, precomputed normals as another 3 columns.
        :param epoch1:
            Epoch object,
            contains as 'additional_dimensions' a segment_id column (mandatory)
            and optionally, precomputed normals as another 3 columns.
        :param extracted_segments_file_name:
            out file
            The file has the following structure:
            numpy array with shape (n_segments_samples, 20) where the column structure is as following:
                [
                    X_COLUMN, Y_COLUMN, Z_COLUMN, -> Center of Gravity
                    EPOCH_ID_COLUMN, -> 0/1
                    EIGENVALUE0_COLUMN, EIGENVALUE1_COLUMN, EIGENVALUE2_COLUMN,
                    EIGENVECTOR_0_X_COLUMN, EIGENVECTOR_0_Y_COLUMN, EIGENVECTOR_0_Z_COLUMN,
                    EIGENVECTOR_1_X_COLUMN, EIGENVECTOR_1_Y_COLUMN, EIGENVECTOR_1_Z_COLUMN,
                    EIGENVECTOR_2_X_COLUMN, EIGENVECTOR_2_Y_COLUMN, EIGENVECTOR_2_Z_COLUMN, -> Normal vector
                    LLSV_COLUMN, -> lowest local surface variation
                    SEGMENT_ID_COLUMN,
                    STANDARD_DEVIATION_COLUMN,
                    NR_POINTS_PER_SEG_COLUMN,
                ]
        :param epoch_additional_dimensions_lookup:
            A dictionary that maps between the names of the columns used internally to identify:
                segment id of the points: "segment_id"  -> Mandatory part of the epochs
                Normal x-axes vector: "N_x"             -> Optionally part of the epochs
                Normal y-axes vector: "N_y"             -> Optionally part of the epochs
                Normal z-axes vector: "N_z"             -> Optionally part of the epochs
            and the names of the columns used by both epoch0 and epoch1.
        :param concatenate_name:
            String that is utilized to uniquely identify the same transformer between multiple pipelines.
        :param kwargs:
            Used for customize the default pipeline parameters.
            Getting the default parameters:
            e.g. "get_pipeline_options"
                In case this parameter is True, the method will print the pipeline options as kwargs.
            e.g. "output_file_name" (of a specific step in the pipeline) default value is "None".
                In case of setting it, the result of computation at that step is dump as xyz file.
            e.g. "distance_3D_threshold" (part of Segmentation Transform)

            this process is stateless
        :return:
            tuple
            [
                numpy array with shape (n, 4|7) corresponding to epoch0 and
                    containing [x,y,z,segment_id] | [x,y,z,N_x,N_y,N_z,segment_id],
                numpy array with shape (m, 4|7) corresponding to epoch1 and
                    containing [x,y,z,segment_id] | [x,y,z,N_x,N_y,N_z,segment_id] | None,
                numpy array with shape (p, 20) corresponding to extracted_segments
            ]
            | None
        """

        transform_pipeline = Pipeline(
            [
                (
                    concatenate_name + "_Transform_Post Segmentation",
                    self._post_segmentation,
                ),
                (
                    concatenate_name + "_Transform_ExtractSegments",
                    self._extract_segments,
                ),
            ]
        )

        # print the default parameters
        PBM3C2._print_default_parameters(
            kwargs=kwargs, pipeline_param_dict=transform_pipeline.get_params()
        )

        # no computation
        if epoch0 is None:  # or epoch1 is None:
            # logger.info("epoch0 is required, no parameter changes applied")
            return

        # save the default pipeline options
        default_options = transform_pipeline.get_params()

        # overwrite the default parameters
        PBM3C2._overwrite_pipeline_parameters(
            kwargs=kwargs, pipeline=transform_pipeline
        )

        # extract columns

        epoch0_normals = _extract_from_additional_dimensions(
            epoch=epoch0,
            column_names=[
                epoch_additional_dimensions_lookup["N_x"],
                epoch_additional_dimensions_lookup["N_y"],
                epoch_additional_dimensions_lookup["N_z"],
            ],
            required_number_of_columns=[0, 3],
        )

        epoch0_segment_id = _extract_from_additional_dimensions(
            epoch=epoch0,
            column_names=[
                epoch_additional_dimensions_lookup["segment_id"],
            ],
            required_number_of_columns=[1],
        )

        epoch0 = np.concatenate(
            (epoch0.cloud, epoch0_normals, epoch0_segment_id),
            axis=1,
        )

        if epoch1 != None:
            epoch1_normals = _extract_from_additional_dimensions(
                epoch=epoch1,
                column_names=[
                    epoch_additional_dimensions_lookup["N_x"],
                    epoch_additional_dimensions_lookup["N_y"],
                    epoch_additional_dimensions_lookup["N_z"],
                ],
                required_number_of_columns=[0, 3],
            )

            epoch1_segment_id = _extract_from_additional_dimensions(
                epoch=epoch1,
                column_names=[
                    epoch_additional_dimensions_lookup["segment_id"],
                ],
                required_number_of_columns=[1],
            )

            epoch1 = np.concatenate(
                (epoch1.cloud, epoch1_normals, epoch1_segment_id),
                axis=1,
            )

        epochs_list = [epoch0]
        if isinstance(epoch1, np.ndarray):
            epochs_list.append(epoch1)

        X = None
        for epoch_id, current_epoch in enumerate(epochs_list):
            if current_epoch.shape[1] == 4:
                # [x, y, z, segment_id] columns
                assert self._post_segmentation.compute_normal, (
                    "The reconstruction process doesn't have, as input, the Normal vector columns, hence, "
                    "the normal vector computation is mandatory."
                )

                logger.info(
                    f"Reconstruct post segmentation output using [x, y, z, segment_id] "
                    f"columns from epoch{epoch_id} "
                )

                if X is not None:
                    X = np.vstack(
                        (
                            X,
                            self._reconstruct_input_without_normals(
                                epoch=current_epoch,
                                epoch_id=epoch_id,
                                columns=self._post_segmentation.columns,
                            ),
                        )
                    )
                else:
                    X = self._reconstruct_input_without_normals(
                        epoch=current_epoch,
                        epoch_id=epoch_id,
                        columns=self._post_segmentation.columns,
                    )
            else:
                # [x, y, z, N_x, N_y, N_z, segment_id] columns
                logger.info(
                    f"Reconstruct post segmentation output using [x, y, z, N_x, N_y, N_z, segment_id] "
                    f"columns from epoch{epoch_id}"
                )

                if X is not None:
                    X = np.vstack(
                        (
                            X,
                            self._reconstruct_input_with_normals(
                                epoch=current_epoch,
                                epoch_id=epoch_id,
                                columns=self._post_segmentation.columns,
                            ),
                        )
                    )
                else:
                    X = self._reconstruct_input_with_normals(
                        epoch=current_epoch,
                        epoch_id=epoch_id,
                        columns=self._post_segmentation.columns,
                    )

        # apply the pipeline

        transform_pipeline.fit(X)
        extracted_segments = transform_pipeline.transform(X)

        if extracted_segments_file_name is not None:
            logger.info(f"'Segments' saved in file: {extracted_segments_file_name}")
            np.savetxt(extracted_segments_file_name, extracted_segments, delimiter=",")
        else:
            logger.debug(f"No file name set as output for 'segments'")

        # restore the default pipeline options
        unused_kwargs = PBM3C2._overwrite_pipeline_parameters(
            kwargs=default_options,
            pipeline=transform_pipeline,
            message="The pipeline parameters after restoration are: ",
        )
        assert (
            len(unused_kwargs) == 0
        ), "All default options should be found when default parameter restoration is done"

        return epoch0, epoch1, extracted_segments

    def predict(
        self,
        epoch0: Epoch = None,
        epoch1: Epoch = None,
        epoch_additional_dimensions_lookup: typing.Dict[str, str] = dict(
            segment_id="segment_id", N_x="N_x", N_y="N_y", N_z="N_z"
        ),
        **kwargs,
    ) -> typing.Union[np.ndarray, None]:
        """
        After the reconstruction of the result that is achieved using the "PB_P3C2 class" pipeline, by applying
        ("Transform LLSV_and_PCA"), ("Transform Segmentation"), ("Transform Second Segmentation"), ("Transform ExtractSegments")
        applied to the segmented point cloud of epoch0 and epoch1, it returns a numpy array of corresponding
        pairs of segments between epoch 0 and epoch 1.

        :param epoch0:
            Epoch object,
            contains as 'additional_dimensions' a segment_id column ( mandatory )
            and optionally, precomputed normals as another 3 columns.
            ( the structure must be consistent with the structure of epoch1 parameter )
        :param epoch1:
            Epoch object.
            contains as 'additional_dimensions' a segment_id column ( mandatory )
            and optionally, precomputed normals as another 3 columns.
            ( the structure must be consistent with the structure of epoch0 parameter )
        :param epoch_additional_dimensions_lookup:
            A dictionary that maps between the names of the columns used internally to identify:
                segment id of the points: "segment_id"  -> Mandatory part of the epochs
                Normal x-axes vector: "N_x"             -> Optionally part of the epochs
                Normal y-axes vector: "N_y"             -> Optionally part of the epochs
                Normal z-axes vector: "N_z"             -> Optionally part of the epochs
            and the names of the columns used by both epoch0 and epoch1.
        :param kwargs:
            Used for customize the default pipeline parameters.

            Getting the default parameters:
            e.g. "get_pipeline_options"
                In case this parameter is True, the method will print the pipeline options as kwargs.

            e.g. "output_file_name" (of a specific step in the pipeline) default value is "None".
                In case of setting it, the result of computation at that step is dump as xyz file.
            e.g. "distance_3D_threshold" (part of Segmentation Transform)

            this process is stateless
        :return:
            A numpy array of shape ( n_pairs, segment_size*2 ) where each row contains a pair of segments.
        """

        predicting_pipeline = Pipeline(
            [
                ("Transform_Post_Segmentation", self._post_segmentation),
                ("Transform_ExtractSegments", self._extract_segments),
                ("Classifier", self._classifier),
            ]
        )

        # print the default parameters
        PBM3C2._print_default_parameters(
            kwargs=kwargs, pipeline_param_dict=predicting_pipeline.get_params()
        )

        # no computation
        if epoch0 is None or epoch1 is None:
            # logger.info("epoch0 and epoch1 are required, no parameter changes applied")
            return

        # save the default pipeline options
        default_options = predicting_pipeline.get_params()

        # overwrite the default parameters
        PBM3C2._overwrite_pipeline_parameters(
            kwargs=kwargs, pipeline=predicting_pipeline
        )

        # extract columns

        epoch0_normals = _extract_from_additional_dimensions(
            epoch=epoch0,
            column_names=[
                epoch_additional_dimensions_lookup["N_x"],
                epoch_additional_dimensions_lookup["N_y"],
                epoch_additional_dimensions_lookup["N_z"],
            ],
            required_number_of_columns=[0, 3],
        )

        epoch0_segment_id = _extract_from_additional_dimensions(
            epoch=epoch0,
            column_names=[
                epoch_additional_dimensions_lookup["segment_id"],
            ],
            required_number_of_columns=[1],
        )

        epoch0 = np.concatenate(
            (epoch0.cloud, epoch0_normals, epoch0_segment_id),
            axis=1,
        )

        epoch1_normals = _extract_from_additional_dimensions(
            epoch=epoch1,
            column_names=[
                epoch_additional_dimensions_lookup["N_x"],
                epoch_additional_dimensions_lookup["N_y"],
                epoch_additional_dimensions_lookup["N_z"],
            ],
            required_number_of_columns=[0, 3],
        )

        epoch1_segment_id = _extract_from_additional_dimensions(
            epoch=epoch1,
            column_names=[
                epoch_additional_dimensions_lookup["segment_id"],
            ],
            required_number_of_columns=[1],
        )

        epoch1 = np.concatenate(
            (epoch1.cloud, epoch1_normals, epoch1_segment_id),
            axis=1,
        )

        X = None
        for epoch_id, current_epoch in enumerate([epoch0, epoch1]):
            if current_epoch.shape[1] == 4:
                # [x, y, z, segment_id] columns
                assert self._post_segmentation.compute_normal, (
                    "The reconstruction process doesn't have, as input, the Normal vector columns, hence, "
                    "the normal vector computation is mandatory."
                )

                logger.info(
                    f"Reconstruct post segmentation output using [x, y, z, segment_id] "
                    f"columns from epoch{epoch_id} "
                )

                if X is not None:
                    X = np.vstack(
                        (
                            X,
                            self._reconstruct_input_without_normals(
                                epoch=current_epoch,
                                epoch_id=epoch_id,
                                columns=self._post_segmentation.columns,
                            ),
                        )
                    )
                else:
                    X = self._reconstruct_input_without_normals(
                        epoch=current_epoch,
                        epoch_id=epoch_id,
                        columns=self._post_segmentation.columns,
                    )
            else:
                # [x, y, z, N_x, N_y, N_z, segment_id] columns
                logger.info(
                    f"Reconstruct post segmentation output using [x, y, z, N_x, N_y, N_z, segment_id] "
                    f"columns from epoch{epoch_id}"
                )

                if X is not None:
                    X = np.vstack(
                        (
                            X,
                            self._reconstruct_input_with_normals(
                                epoch=current_epoch,
                                epoch_id=epoch_id,
                                columns=self._post_segmentation.columns,
                            ),
                        )
                    )
                else:
                    X = self._reconstruct_input_with_normals(
                        epoch=current_epoch,
                        epoch_id=epoch_id,
                        columns=self._post_segmentation.columns,
                    )

        # apply the pipeline

        out = predicting_pipeline.predict(X)

        # restore the default pipeline options
        unused_kwargs = PBM3C2._overwrite_pipeline_parameters(
            kwargs=default_options,
            pipeline=predicting_pipeline,
            message="The pipeline parameters after restoration are: ",
        )
        assert (
            len(unused_kwargs) == 0
        ), "All default options should be found when default parameter restoration is done"

        return out

    def compute_distances(
        self,
        epoch0: Epoch = None,
        epoch1: Epoch = None,
        alignment_error: float = 1.1,
        epoch_additional_dimensions_lookup: typing.Dict[str, str] = dict(
            segment_id="segment_id", N_x="N_x", N_y="N_y", N_z="N_z"
        ),
        **kwargs,
    ) -> typing.Union[typing.Tuple[np.ndarray, np.ndarray], None]:
        """
        Compute the distance between 2 epochs. It also adds the following properties at the end of the computation:
            distances, corepoints (corepoints of epoch0), epochs (epoch0, epoch1), uncertainties

        :param epoch0:
            Epoch object,
            contains as 'additional_dimensions' a segment_id column (mandatory)
            and optionally, precomputed normals as another 3 columns.
        :param epoch1:
            Epoch object,
            contains as 'additional_dimensions' a segment_id column (mandatory)
            and optionally, precomputed normals as another 3 columns.
        :param alignment_error:
            alignment error reg between point clouds.
        :param epoch_additional_dimensions_lookup:
            A dictionary that maps between the names of the columns used internally to identify:
                segment id of the points: "segment_id"  -> Mandatory part of the epochs
                Normal x-axes vector: "N_x"             -> Optionally part of the epochs
                Normal y-axes vector: "N_y"             -> Optionally part of the epochs
                Normal z-axes vector: "N_z"             -> Optionally part of the epochs
            and the names of the columns used by both epoch0 and epoch1.
        :param kwargs:
            Used for customize the default pipeline parameters.

            Getting the default parameters:
            e.g. "get_pipeline_options"
                In case this parameter is True, the method will print the pipeline options as kwargs.

            e.g. "output_file_name" (of a specific step in the pipeline) default value is "None".
                In case of setting it, the result of computation at that step is dump as xyz file.
            e.g. "distance_3D_threshold" (part of Segmentation Transform)

            this process is stateless

        :return:
            tuple [distances, uncertainties]
                'distances' is np.array (nr_similar_pairs, 1)
                'uncertainties' is np array (nr_similar_pairs, 1) and it has the following structure:
                    dtype=np.dtype(
                        [
                            ("lodetection", "<f8"),
                            ("spread1", "<f8"),
                            ("num_samples1", "<i8"),
                            ("spread2", "<f8"),
                            ("num_samples2", "<i8"),
                        ]
                    )
            | None
        """

        logger.info(f"PBM3C2WithSegments.compute_distances(...)")

        return super()._compute_distances(
            epoch0_info=epoch0,
            epoch1=epoch1,
            alignment_error=alignment_error,
            epoch_additional_dimensions_lookup=epoch_additional_dimensions_lookup,
            **kwargs,
        )
