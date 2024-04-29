from py4dgeo.util import Py4DGeoError

from copy import deepcopy
import dataclasses
import numpy as np

import _py4dgeo


@dataclasses.dataclass(frozen=True)
class Transformation:
    """A transformation that can be applied to a point cloud"""

    affine_transformation: np.ndarray
    reduction_point: np.ndarray


def _fit_transform(A, B, reduction_point=None):
    """Find a transformation that fits two point clouds onto each other"""

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Apply the reduction_point if provided
    if reduction_point is not None:
        centroid_A -= reduction_point
        centroid_B -= reduction_point

    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # homogeneous transformation
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def iterative_closest_point(
    reference_epoch, epoch, max_iterations=50, tolerance=0.00001, reduction_point=None
):
    """Perform an Iterative Closest Point algorithm (ICP)

    :param reference_epoch:
        The reference epoch to match with.
    :type reference_epoch: py4dgeo.Epoch
    :param epoch:
        The epoch to be transformed to the reference epoch
    :type epoch: py4dgeo.Epoch
    :param max_iterations:
        The maximum number of iterations to be performed in the ICP algorithm
    :type max_iterations: int
    :param tolerance:
        The tolerance criterium used to terminate ICP iteration.
    :type tolerance: float
    :param reduction_point:
        A translation vector to apply before applying rotation and scaling.
        This is used to increase the numerical accuracy of transformation.
    :type reduction_point: np.ndarray
    """

    # Ensure that reference_epoch has its KDTree built
    if reference_epoch.kdtree.leaf_parameter() == 0:
        reference_epoch.build_kdtree()

    # Apply the default for the registration point
    if reduction_point is None:
        reduction_point = np.array([0, 0, 0])

    # Make a copy of the cloud to be transformed.
    cloud = epoch.cloud.copy()
    prev_error = 0

    for _ in range(max_iterations):
        neighbor_arrays = np.asarray(reference_epoch.kdtree.nearest_neighbors(cloud, 1))
        indices, distances = np.split(neighbor_arrays, 2, axis=0)

        indices = np.squeeze(indices.astype(int))
        distances = np.squeeze(distances)

        # Calculate a transform and apply it
        T = _fit_transform(
            cloud, reference_epoch.cloud[indices, :], reduction_point=reduction_point
        )
        _py4dgeo.transform_pointcloud_inplace(
            cloud, T, reduction_point, np.empty((1, 3))
        )

        # Determine convergence
        mean_error = np.mean(np.sqrt(distances))

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return Transformation(
        affine_transformation=_fit_transform(epoch.cloud, cloud),
        reduction_point=reduction_point,
    )


def point_to_plane_icp(
    reference_epoch, epoch, max_iterations=50, tolerance=0.00001, reduction_point=None
):
    """Perform a point to plane Iterative Closest Point algorithm (ICP), based on Gauss-Newton method for computing the least squares solution

    :param reference_epoch:
        The reference epoch to match with. This epoch has to have calculated normals.
    :type reference_epoch: py4dgeo.Epoch
    :param epoch:
        The epoch to be transformed to the reference epoch
    :type epoch: py4dgeo.Epoch
    :param max_iterations:
        The maximum number of iterations to be performed in the ICP algorithm
    :type max_iterations: int
    :param tolerance:
        The tolerance criterium used to terminate ICP iteration.
    :type tolerance: float
    :param reduction_point:
        A translation vector to apply before applying rotation and scaling.
        This is used to increase the numerical accuracy of transformation.
    :type reduction_point: np.ndarray
    """

    from py4dgeo.epoch import Epoch

    # Ensure that Epoch has calculated normals
    if reference_epoch.normals is None:
        raise Py4DGeoError(
            "Normals for this Reference Epoch have not been calculated! Please use Epoch.calculate_normals or load externally calculated normals."
        )

    # Ensure that reference_epoch has its KDTree built
    if reference_epoch.kdtree.leaf_parameter() == 0:
        reference_epoch.build_kdtree()

    # Apply the default for the registration point
    if reduction_point is None:
        reduction_point = np.array([0, 0, 0])

    # Make a copy of the cloud to be transformed.
    trans_epoch = epoch.copy()

    prev_error = 0
    for _ in range(max_iterations):
        neighbor_arrays = np.asarray(
            reference_epoch.kdtree.nearest_neighbors(trans_epoch.cloud, 1)
        )
        indices, distances = np.split(neighbor_arrays, 2, axis=0)

        indices = np.squeeze(indices.astype(int))
        distances = np.squeeze(distances)

        # Calculate a transform and apply it
        T = _py4dgeo.fit_transform_GN(
            trans_epoch.cloud,
            reference_epoch.cloud[indices, :],
            reference_epoch.normals[indices, :],
        )
        trans_epoch.transform(
            Transformation(affine_transformation=T, reduction_point=reduction_point)
        )

        # Determine convergence
        mean_error = np.mean(np.sqrt(distances))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return Transformation(
        affine_transformation=_py4dgeo.fit_transform_GN(
            epoch.cloud,
            trans_epoch.cloud,
            trans_epoch.normals,
        ),
        reduction_point=reduction_point,
    )


def calculate_bounding_box(point_cloud):
    """
    Calculate the bounding box of a point cloud.

    Parameters:
    - point_cloud: NumPy array with shape (N, 3), where N is the number of points.

    Returns:
    - min_bound: 1D array representing the minimum coordinates of the bounding box.
    - max_bound: 1D array representing the maximum coordinates of the bounding box.
    """
    min_bound = np.min(point_cloud, axis=0)
    max_bound = np.max(point_cloud, axis=0)

    return min_bound, max_bound


def calculate_bounding_box_change(
    bounding_box_min, bounding_box_max, transformation_matrix
):
    """Calculate the change in kdtree bounding box corners after applying a transformation matrix.
    Parameters:
    - bounding_box_min: 1D array representing the minimum coordinates of the bounding box.
    - bounding_box_max: 1D array representing the maximum coordinates of the bounding box.
    - transformation_matrix: 2D array representing the transformation matrix.
    Returns:
    - max_change: The maximum change in the bounding box corners.
    """

    # Convert bounding box to homogeneous coordinates
    bounding_box_min_homogeneous = np.concatenate((bounding_box_min, [1]))
    bounding_box_max_homogeneous = np.concatenate((bounding_box_max, [1]))
    bounding_box_min_homogeneous = np.reshape(bounding_box_min_homogeneous, (4, 1))
    bounding_box_max_homogeneous = np.reshape(bounding_box_max_homogeneous, (4, 1))

    # Calculate the change in bounding box corners
    bb_c2p1 = np.dot(transformation_matrix, bounding_box_min_homogeneous)
    bb_c2p2 = np.dot(transformation_matrix, bounding_box_max_homogeneous)

    dif_bb_pmin = np.sum(np.abs(bb_c2p1[:3] - bounding_box_min_homogeneous[:3]))
    dif_bb_pmax = np.sum(np.abs(bb_c2p2[:3] - bounding_box_max_homogeneous[:3]))

    return max(dif_bb_pmin, dif_bb_pmax)


def calculate_dis_threshold(epoch1, epoch2):
    """Calculate the distance threshold for the next iteration of the registration method
    Parameters:
    - epoch1: The reference epoch.
    - epoch2: Stable points of epoch.
    Returns:
    - dis_threshold: The distance threshold.
    """
    neighbor_arrays = np.asarray(epoch1.kdtree.nearest_neighbors(epoch2.cloud, 1))
    indices, distances = np.split(neighbor_arrays, 2, axis=0)
    distances = np.squeeze(distances)

    if indices.size > 0:
        # Calculate mean distance
        mean_dis = np.mean(np.sqrt(distances))

        # Calculate standard deviation
        std_dis = np.sqrt(np.mean((mean_dis - distances) ** 2))

        dis_threshold = mean_dis + 1.0 * std_dis

    return dis_threshold


def icp_with_stable_areas(
    reference_epoch,
    epoch,
    initial_distance_threshold,
    level_of_detection,
    reference_supervoxel_resolution,
    supervoxel_resolution,
    min_svp_num=10,
    reduction_point=None,
):
    """Perform a registration method

    :param reference_epoch:
        The reference epoch to match with. This epoch has to have calculated normals.
    :type reference_epoch: py4dgeo.Epoch
    :param epoch:
        The epoch to be transformed to the reference epoch
    :type epoch: py4dgeo.Epoch
    :param initial_distance_threshold:
        The upper boundary of the distance threshold in the iteration. It can be (1) an empirical value manually set by the user according to the approximate accuracy of coarse registration,
        or (2) calculated by the mean and standard of the nearest neighbor distances of all points.
    :type initial_distance_threshold: float
    :param level_of_detection:
        The lower boundary (minimum) of the distance threshold in the iteration.
        It can be  (1) an empirical value manually set by the user according to the approximate uncertainty of laser scanning measurements in different scanning configurations and scenarios
        (e.g., 1 cm for TLS point clouds in short distance and 4 cm in long distance, 8 cm for ALS point clouds, etc.),
        or (2) calculated by estimating the standard deviation from local modeling (e.g., using the level of detection in M3C2 or M3C2-EP calculations).
    :type level_of_detection: float
    :param reference_supervoxel_resolution:
       The approximate size of generated supervoxels for the reference epoch.
       It can be (1) an empirical value manually set by the user according to different surface geometries and scanning distance (e.g., 2-10 cm for indoor scenes, 1-3 m for landslide surface),
         or (2) calculated by 10-20 times the average point spacing (original resolution of point clouds). In both cases, the number of points in each supervoxel should be at least 10 (i.e., minSVPnum = 10).
    :type reference_supervoxel_resolution: float
    :param supervoxel_resolution:
         The same as `reference_supervoxel_resolution`, but for a different epoch.
    :type supervoxel_resolution: float
    :param min_svp_num:
         Minimum number of points for supervoxels to be taken into account in further calculations.
    :type min_svp_num: int
    :param reduction_point:
        A translation vector to apply before applying rotation and scaling.
        This is used to increase the numerical accuracy of transformation.
    :type reduction_point: np.ndarray

    """

    from py4dgeo.epoch import as_epoch

    # Ensure that reference_epoch has its KDTree build
    if reference_epoch.kdtree.leaf_parameter() == 0:
        reference_epoch.build_kdtree()

    # Ensure that epoch has its KDTree build
    if epoch.kdtree.leaf_parameter() == 0:
        epoch.build_kdtree()

    # Ensure that Epoch has calculated normals
    # Ensure that Epoch has calculated normals
    if reference_epoch.normals is None:
        raise Py4DGeoError(
            "Normals for this Reference Epoch have not been calculated! Please use Epoch.calculate_normals or load externally calculated normals."
        )

    # Ensure that Epoch has calculated normals

    # Ensure that Epoch has calculated normals
    if epoch.normals is None:
        raise Py4DGeoError(
            "Normals for this Epoch have not been calculated! Please use Epoch.calculate_normals or load externally calculated normals."
        )


    # Apply the default for the registration point
    if reduction_point is None:
        reduction_point = np.array([0, 0, 0])

    if initial_distance_threshold <= level_of_detection:
        initial_distance_threshold = level_of_detection

    transMatFinal = np.identity(4)  # Identity matrix for initial transMatFinal
    stage3 = stage4 = 0
    epoch_copy = epoch.copy()  # Create copy of epoch for applying transformation

    k = 50  # Number of nearest neighbors to consider in supervoxel segmentation

    clouds_pc1, _, centroids_pc1, _ = _py4dgeo.segment_pc_in_supervoxels(
        reference_epoch,
        reference_epoch.kdtree,
        reference_epoch.normals,
        reference_supervoxel_resolution,
        k,
        min_svp_num,
    )
    (
        clouds_pc2,
        normals2,
        centroids_pc2,
        boundary_points_pc2,
    ) = _py4dgeo.segment_pc_in_supervoxels(
        epoch, epoch.kdtree, epoch.normals, supervoxel_resolution, k, min_svp_num
    )

    centroids_pc1 = as_epoch(np.array(centroids_pc1))
    centroids_pc1.build_kdtree()
    centroids_pc2 = np.array(centroids_pc2)
    boundary_points_pc2 = np.concatenate(boundary_points_pc2, axis=0)

    _, reference_distances = np.split(
        np.asarray(reference_epoch.kdtree.nearest_neighbors(reference_epoch.cloud, 2)),
        2,
        axis=0,
    )
    basicRes = np.mean(np.squeeze(reference_distances))
    dis_threshold = initial_distance_threshold

    while stage4 == 0:
        cor_dist_ct = _py4dgeo.compute_correspondence_distances(
            centroids_pc1, centroids_pc2, clouds_pc1, len(reference_epoch.cloud)
        )
        # Calculation BP2-CT1
        cor_dist_bp = _py4dgeo.compute_correspondence_distances(
            centroids_pc1,
            boundary_points_pc2,
            clouds_pc1,
            len(reference_epoch.cloud),
        )
        # calculation BP2- CP1
        cor_dist_pc = _py4dgeo.compute_correspondence_distances(
            reference_epoch,
            boundary_points_pc2,
            clouds_pc1,
            len(reference_epoch.cloud),
        )

        stablePC2 = []  # Stable supervoxels
        normPC2 = []  # Stable supervoxel's normals

        dt_point = dis_threshold + 2 * basicRes

        for i in range(len(centroids_pc2)):
            if cor_dist_ct[i] < dis_threshold and all(
                cor_dist_bp[j + 6 * i] < dis_threshold
                and cor_dist_pc[j + 6 * i] < dt_point
                for j in range(6)
            ):
                stablePC2.append(clouds_pc2[i])
                normPC2.append(normals2[i])

        # Handle empty stablePC2
        if len(stablePC2) == 0:
            raise Py4DGeoError(
                "No stable supervoxels found! Please adjust the parameters."
            )

        stablePC2 = np.vstack(stablePC2)
        stablePC2 = as_epoch(stablePC2)
        normPC2 = np.vstack(normPC2)
        stablePC2.normals_attachment(normPC2)
        trans_mat_cur_obj = point_to_plane_icp(
            reference_epoch,
            stablePC2,
            max_iterations=50,
            tolerance=0.00001,
            reduction_point=reduction_point,
            reference_epoch,
            stablePC2,
            max_iterations=50,
            tolerance=0.00001,
            reduction_point=reduction_point,
        )

        trans_mat_cur = trans_mat_cur_obj.affine_transformation

        # BB
        initial_min_bound, initial_max_bound = calculate_bounding_box(epoch_copy.cloud)
        max_bb_change = calculate_bounding_box_change(
            initial_min_bound, initial_max_bound, trans_mat_cur
        )
        # update DT
        if stage3 == 0 and max_bb_change < 2 * level_of_detection:
            stage3 = 1
        elif dis_threshold == level_of_detection:
            stage4 = 1


        if stage3 == 0:
            dis_threshold = calculate_dis_threshold(reference_epoch, stablePC2)
            if dis_threshold <= level_of_detection:
                dis_threshold = level_of_detection

        if stage3 == 1 and stage4 == 0:
            dis_threshold = 0.8 * dis_threshold
            if dis_threshold <= level_of_detection:
                dis_threshold = level_of_detection

        # update values and apply changes
        # Apply the transformation to the epoch
        epoch_copy.transform(
            Transformation(
                affine_transformation=trans_mat_cur, reduction_point=reduction_point
            )
        )
        _py4dgeo.transform_pointcloud_inplace(
            centroids_pc2, trans_mat_cur, reduction_point, np.empty((1, 3))
        )
        _py4dgeo.transform_pointcloud_inplace(
            boundary_points_pc2, trans_mat_cur, reduction_point, np.empty((1, 3))
        )
        for i in range(len(clouds_pc2)):
            _py4dgeo.transform_pointcloud_inplace(
                clouds_pc2[i], trans_mat_cur, reduction_point, np.empty((1, 3))
            )

        transMatFinal = trans_mat_cur @ transMatFinal

    return Transformation(
        affine_transformation=transMatFinal,
        reduction_point=reduction_point,
    )
