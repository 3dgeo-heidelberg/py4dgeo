import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# from sklearn.metrics import euclidean_distances

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.estimator_checks import check_estimator

from abc import ABC, abstractmethod

from py4dgeo.epoch import *
from py4dgeo import *

from sklearn import set_config

set_config(display="diagram")

from IPython import display

from sympy import Plane, Point3D

from vedo import *

import colorsys

import random


class BaseTransformer(TransformerMixin, BaseEstimator, ABC):
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
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        pass

    def fit(self, X, y=None):

        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]

        # Return the transformer
        print("Transformer Fit")
        return self._fit(X, y)

    def transform(self, X):

        # Check is fit had been called
        check_is_fitted(self, "n_features_")

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )

        print("Transformer Transform")
        return self._transform(X)


class Add_LLSV_and_PCA(BaseTransformer):
    def __init__(self, radius=10):

        self.radius = radius

    def llsv_and_pca(self, x, X):

        size = X.shape[0]

        X_avg = np.mean(X, axis=0)  # compute mean
        B = X - np.tile(X_avg, (size, 1))

        # Find principal components (SVD)
        U, S, VT = np.linalg.svd(B.T / np.sqrt(size), full_matrices=0)

        # lowest local surface variation, normal vector
        # return np.hstack(( S[-1] / np.sum(S), U[:,2] ))

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
        return self
        pass

    def _transform(self, X):

        X_Column = 0
        Y_Column = 1
        Z_Column = 2
        EpochID_Column = 3
        X_Y_Z_Columns = [X_Column, Y_Column, Z_Column]

        mask_epoch0 = X[:, EpochID_Column] == 0
        mask_epoch1 = X[:, EpochID_Column] == 1

        epoch0_set = X[mask_epoch0, :-1]
        epoch1_set = X[mask_epoch1, :-1]

        _epoch = [Epoch(epoch0_set), Epoch(epoch1_set)]

        _epoch[0].build_kdtree()
        _epoch[1].build_kdtree()

        # add extra columns
        # Eigenvalues( 3 columns ) |
        # Eigenvectors( 3 columns ) X 3 [ in descending normal order ] |
        # Lowest local surface variation( 1 column )
        # Total 13 new columns
        new_columns = np.zeros((X.shape[0], 13))
        X = np.hstack((X, new_columns))

        Eigenvalue0_Column = 4
        Eigenvalue1_Column = 5
        Eigenvalue2_Column = 6
        Eigenvector0x_Column = 7
        Eigenvector0y_Column = 8
        Eigenvector0z_Column = 9
        Eigenvector1x_Column = 10
        Eigenvector1y_Column = 11
        Eigenvector1z_Column = 12
        Eigenvector2x_Column = 13
        Eigenvector2y_Column = 14
        Eigenvector2z_Column = 15
        llsv_Column = 16
        Normal_Columns = [
            Eigenvector2x_Column,
            Eigenvector2y_Column,
            Eigenvector2z_Column,
        ]

        return np.apply_along_axis(
            lambda x: self.llsv_and_pca(
                x,
                _epoch[int(x[EpochID_Column])].cloud[
                    _epoch[int(x[EpochID_Column])].kdtree.radius_search(
                        x[X_Y_Z_Columns], self.radius
                    )
                ],
            ),
            1,
            X,
        )
        # return X
        pass


def angle_difference_check(normal1, normal2, angle_diff_threshold):

    # normal1, normal2 have to be unit vectors ( and that is the case as a result of the SVD process )
    return (
        np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0)) * 180.0 / np.pi
        <= angle_diff_threshold
    )


class Segmentation(BaseTransformer):
    def __init__(
        self,
        radius=2,
        angle_diff_threshold=1,
        disntance_3D_threshold=1.5,
        distance_orthogonal_threshold=1.5,
        llsv_threshold=1,
        roughness_threshold=5,
        max_nr_points_neighbourhood=100,
        min_nr_points_per_segment=5,
    ):

        self.radius = radius
        self.angle_diff_threshold = angle_diff_threshold
        self.disntance_3D_threshold = disntance_3D_threshold
        self.distance_orthogonal_threshold = distance_orthogonal_threshold
        self.llsv_threshold = llsv_threshold
        self.roughness_threshold = roughness_threshold
        self.max_nr_points_neighbourhood = max_nr_points_neighbourhood
        self.min_nr_points_per_segment = min_nr_points_per_segment

    # def angle_difference_check(self, normal1, normal2):
    #
    #     # normal1, normal2 have to be unit vectors ( and that is the case as a result of the SVD process )
    #     return np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0)) * 180./np.pi <= self.angle_diff_threshold

    def disntance_3D_check(
        self, point, segment_id, X, X_Y_Z_Columns, Segment_ID_Column
    ):
        point_mask = X[:, Segment_ID_Column] == segment_id

        # x,y,z
        # np.linalg.norm(X[point_mask, :3] - point.reshape(1, 3), ord=2, axis=1)
        # np.linalg.norm(X[point_mask, :3] - point.reshape(1, 3), ord=2, axis=1)
        # can be optimized by changing the norm
        return (
            np.min(
                np.linalg.norm(X[point_mask, :3] - point.reshape(1, 3), ord=2, axis=1)
            )
            <= self.disntance_3D_threshold
        )

    def distance_orthogonal_check(self, candidate_point, plane_point, plane_normal):
        # candidate_point = Point3D(candidate_point)
        # plane = Plane( Point3D(plane_point), normal_vector=plane_normal)
        # distance = plane.distance( candidate_point )
        #
        # return (distance.__float__()-self.distance_orthogonal_threshold<=0)

        d = -plane_point.dot(plane_normal)
        distance = (plane_normal.dot(candidate_point) + d) / np.linalg.norm(
            plane_normal
        )
        return distance - self.distance_orthogonal_threshold <= 0

    def lowest_local_suface_variance_check(self, llsv):
        return llsv <= self.llsv_threshold

    def roughness_check(self):
        return True

    def _fit(self, X, y=None):
        return self
        pass

    def _transform(self, X):

        X_Column = 0
        Y_Column = 1
        Z_Column = 2
        X_Y_Z_Columns = [X_Column, Y_Column, Z_Column]

        EpochID_Column = 3
        # Lowest local surface variation
        llsv_Column = -2

        Eigenvector2x_Column = 13
        Eigenvector2y_Column = 14
        Eigenvector2z_Column = 15
        llsv_Column = 16
        Normal_Columns = [
            Eigenvector2x_Column,
            Eigenvector2y_Column,
            Eigenvector2z_Column,
        ]

        # default, the points are part of no segment ( e.g. -1 )
        # WARNING!!! you can't apply this transformer 2 time as this transformer changes the structure of input 'X' !!!
        Default_No_Segment = -1
        new_columns = np.full((X.shape[0], 1), Default_No_Segment, dtype=float)
        X = np.hstack((X, new_columns))

        Segment_ID_Column = 17

        mask_epoch0 = X[:, EpochID_Column] == 0
        mask_epoch1 = X[:, EpochID_Column] == 1

        epoch0_set = X[mask_epoch0, :3]  # x,y,z
        epoch1_set = X[mask_epoch1, :3]  # x,y,z

        _epoch = [Epoch(epoch0_set), Epoch(epoch1_set)]

        _epoch[0].build_kdtree()
        _epoch[1].build_kdtree()

        # sort by "Lowest local surface variation"
        Sort_indx_epoch0 = X[mask_epoch0, llsv_Column].argsort()
        Sort_indx_epoch1 = X[mask_epoch1, llsv_Column].argsort()
        Sort_indx_epoch = [Sort_indx_epoch0, Sort_indx_epoch1]

        offset_in_X = [0, Sort_indx_epoch0.shape[0]]

        # initialization required between multiple iterations of these transform
        seg_id = np.max(X[:, Segment_ID_Column])

        for epoch_id in range(2):
            # seg_id = -1
            for indx_row in Sort_indx_epoch[epoch_id] + offset_in_X[epoch_id]:
                if X[indx_row, Segment_ID_Column] < 0:  # not part of a segment yet
                    seg_id += 1
                    X[indx_row, Segment_ID_Column] = seg_id
                    # this step can be preprocessed in a vectorized way!
                    indx_kd_tree_list = _epoch[epoch_id].kdtree.radius_search(
                        X[indx_row, X_Y_Z_Columns], self.radius
                    )[: self.max_nr_points_neighbourhood]
                    for indx_kd_tree in indx_kd_tree_list:
                        if (
                            X[indx_kd_tree + offset_in_X[epoch_id], Segment_ID_Column]
                            < 0
                            and angle_difference_check(
                                # self.angle_difference_check(
                                X[indx_row, Normal_Columns],
                                X[indx_kd_tree + offset_in_X[epoch_id], Normal_Columns],
                                self.angle_diff_threshold,
                            )
                            and self.disntance_3D_check(
                                X[indx_kd_tree + offset_in_X[epoch_id], X_Y_Z_Columns],
                                seg_id,
                                X,
                                X_Y_Z_Columns,
                                Segment_ID_Column,
                            )
                            and self.distance_orthogonal_check(
                                X[indx_kd_tree + offset_in_X[epoch_id], X_Y_Z_Columns],
                                X[indx_row, X_Y_Z_Columns],
                                X[indx_row, Normal_Columns],
                            )
                            and self.lowest_local_suface_variance_check(
                                X[indx_kd_tree + offset_in_X[epoch_id], llsv_Column]
                            )
                            and self.roughness_check()
                        ):
                            X[
                                indx_kd_tree + offset_in_X[epoch_id], Segment_ID_Column
                            ] = seg_id
                    # floating equality test must be changed with a more robust test !!!
                    nr_points_segment = np.count_nonzero(
                        X[:, Segment_ID_Column] == seg_id
                    )
                    if nr_points_segment < self.min_nr_points_per_segment:
                        mask_seg_id = X[:, Segment_ID_Column] == seg_id
                        X[mask_seg_id, Segment_ID_Column] = Default_No_Segment
                        seg_id -= 1  # since we don't have a new segment
                        pass
        return X


def HSVToRGB(h, s, v):
    return colorsys.hsv_to_rgb(h, s, v)


def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]


def SegmentationVisualizer(X):

    sets = []

    max = int(X[:, 17].max())
    colors = getDistinctColors(max + 1)

    plt = Plotter(axes=3)

    for i in range(0, max + 1):

        mask = X[:, 17] == float(i)
        set_cloud = X[mask, :3]  # x,y,z

        sets = sets + [Points(set_cloud, colors[i], alpha=1, r=10)]

    plt.show(sets).close()


def Geterate_Random_y(X):

    Segment_ID_Column = 17
    EpochID_Column = 3

    mask_epoch0 = X[:, EpochID_Column] == 0
    mask_epoch1 = X[:, EpochID_Column] == 1

    epoch0_set = X[mask_epoch0, :]  # all
    epoch1_set = X[mask_epoch1, :]  # all

    nr_pairs = min(epoch0_set.shape[0], epoch1_set.shape[0]) // 3

    indx0_seg_id = random.sample(range(epoch0_set.shape[0]), nr_pairs)
    indx1_seg_id = random.sample(range(epoch1_set.shape[0]), nr_pairs)

    set0_seg_id = epoch0_set[indx0_seg_id, Segment_ID_Column]
    set1_seg_id = epoch1_set[indx1_seg_id, Segment_ID_Column]

    rand_y_01 = list(np.random.randint(0, 2, nr_pairs))

    return np.array([set0_seg_id, set1_seg_id, rand_y_01]).T
    pass


class ExtractSegments(BaseTransformer):
    def __init__(self):
        pass

    def _fit(self, X, y=None):
        return self
        pass

    def _transform(self, X):

        X_Column = 0
        Y_Column = 1
        Z_Column = 2

        EpochID_Column = 3
        Eigenvalue0_Column = 4
        Eigenvalue1_Column = 5
        Eigenvalue2_Column = 6
        Eigenvector0x_Column = 7
        Eigenvector0y_Column = 8
        Eigenvector0z_Column = 9
        Eigenvector1x_Column = 10
        Eigenvector1y_Column = 11
        Eigenvector1z_Column = 12
        Eigenvector2x_Column = 13
        Eigenvector2y_Column = 14
        Eigenvector2z_Column = 15
        llsv_Column = 16
        Segment_ID_Column = 17

        # new column
        Nr_points_seg_Column = 18

        max = int(X[:, Segment_ID_Column].max())
        X_Segments = np.empty((int(max) + 1, 19), dtype=float)

        for i in range(0, max + 1):

            mask = X[:, Segment_ID_Column] == float(i)
            set_cloud = X[mask, :]  # all
            nr_points = set_cloud.shape[0]
            arg_min = set_cloud[:, llsv_Column].argmin()
            X_Segments[i, :-1] = set_cloud[arg_min, :]
            X_Segments[i, -1] = nr_points

        return X_Segments

    pass


class ExtendedClassifier(RandomForestClassifier):
    def __init__(self, angle_diff_threshold=1):

        self.angle_diff_threshold = angle_diff_threshold
        super().__init__()

    def build_X_similarity(self, y_column, X):

        X_Column = 0
        Y_Column = 1
        Z_Column = 2

        EpochID_Column = 3
        Eigenvalue0_Column = 4
        Eigenvalue1_Column = 5
        Eigenvalue2_Column = 6
        Eigenvector0x_Column = 7
        Eigenvector0y_Column = 8
        Eigenvector0z_Column = 9
        Eigenvector1x_Column = 10
        Eigenvector1y_Column = 11
        Eigenvector1z_Column = 12
        Eigenvector2x_Column = 13
        Eigenvector2y_Column = 14
        Eigenvector2z_Column = 15
        llsv_Column = 16
        Segment_ID_Column = 17

        Nr_points_seg_Column = 18

        Normal_Columns = [
            Eigenvector2x_Column,
            Eigenvector2y_Column,
            Eigenvector2z_Column,
        ]

        seg_epoch0 = X[int(y_column[0]), :]
        seg_epoch1 = X[int(y_column[1]), :]

        angle = (
            np.arccos(
                np.clip(
                    np.dot(seg_epoch0[Normal_Columns], seg_epoch1[Normal_Columns]),
                    -1.0,
                    1.0,
                )
            )
            * 180.0
            / np.pi
        )

        # implement this!!!
        point_density_diff = random.random() * 20

        eigen_value_smallest_diff = abs(
            seg_epoch0[Eigenvalue2_Column] - seg_epoch1[Eigenvalue2_Column]
        )
        eigen_value_largest_diff = abs(
            seg_epoch0[Eigenvalue0_Column] - seg_epoch1[Eigenvalue0_Column]
        )
        eigen_value_middle_diff = abs(
            seg_epoch0[Eigenvalue1_Column] - seg_epoch1[Eigenvalue1_Column]
        )

        nr_pointds_diff = abs(
            seg_epoch0[Nr_points_seg_Column] - seg_epoch1[Nr_points_seg_Column]
        )

        return np.array(
            [
                angle,
                point_density_diff,
                eigen_value_smallest_diff,
                eigen_value_largest_diff,
                eigen_value_middle_diff,
                nr_pointds_diff,
            ]
        )

    def fit(self, X, y):

        X_similarity = np.apply_along_axis(
            lambda y_column: self.build_X_similarity(y_column, X), 1, y
        )

        # extra functionality
        print("Classifier Fit")
        return super().fit(X_similarity, y[:, 2])

    def predict(self, X):
        # extra functionality

        X_similarity = np.apply_along_axis(
            lambda y_column: self.build_X_similarity(y_column, X), 1, y
        )

        print("Classifier Predict")
        return super().predict(X_similarity)

    # def decision_function(self, X):
    #     return self.predict(X)


util.ensure_test_data_availability()

Epoch0, Epoch1 = read_from_xyz("plane_horizontal_t1.xyz", "plane_horizontal_t2.xyz")

Sample = 441

X0 = np.hstack((Epoch0.cloud[:Sample, :], np.zeros((Sample, 1))))
X1 = np.hstack((Epoch1.cloud[:Sample, :], np.ones((Sample, 1))))

# | x | y | z | epoch I.D.
X = np.vstack((X0, X1))  # just some random input
y = np.vstack(
    (Epoch0.cloud[:Sample, 0], Epoch1.cloud[:Sample, 0])
)  # just some random input

Tr1 = Add_LLSV_and_PCA()
Tr1.fit(X, y)
X_tr1 = Tr1.transform(X)

Tr2 = Segmentation()
Tr2.fit(X_tr1, y)
X_tr2 = Tr2.transform(X_tr1)

print(X_tr2[:, 17])

# SegmentationVisualizer(X_tr2)

Tr3 = ExtractSegments()
Tr3.fit(X_tr2, y)
X_tr3 = Tr3.transform(X_tr2)

print(X_tr3)

y_random = Geterate_Random_y(X_tr3)

classifier = ExtendedClassifier()
classifier.fit(X_tr3, y_random)
# classifier.predict(X_tr3)

pipe = Pipeline(
    [
        ("Transform 1", Add_LLSV_and_PCA()),
        ("Transform 2", Segmentation()),
        ("Transform 3", ExtractSegments()),
        ("Classifier", ExtendedClassifier()),
    ]
)
