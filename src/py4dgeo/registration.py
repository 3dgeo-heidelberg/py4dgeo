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


def _plane_Jacobian(Rot_a, n):
    """Calculate Jacobian for point to plane method"""

    J = np.zeros((1, 6))
    J[:, 3:] = n
    J[:, :3] = -np.cross(Rot_a, n)
    return J


def _point_Jacobian(Rot_p):
    """Calculate Jacobian for point to point method"""

    J = np.zeros((3, 6))
    J[:, 3:] = np.eye(3)
    J[:, 0] = np.cross(Rot_p, np.array([1, 0, 0]))
    J[:, 1] = np.cross(Rot_p, np.array([0, 1, 0]))
    J[:, 2] = np.cross(Rot_p, np.array([0, 0, 1]))
    return J


def _set_rot_trans(euler_array):
    """Calculate rotation and transformation matrix"""

    alpha, beta, gamma, x, y, z = euler_array

    Rot_z = np.array(
        [np.cos(gamma), np.sin(gamma), 0, -np.sin(gamma), np.cos(gamma), 0, 0, 0, 1]
    ).reshape(3, 3)
    Rot_y = np.array(
        [np.cos(beta), 0, -np.sin(beta), 0, 1, 0, np.sin(beta), 0, np.cos(beta)]
    ).reshape(3, 3)
    Rot_x = np.array(
        [1, 0, 0, 0, np.cos(alpha), np.sin(alpha), 0, -np.sin(alpha), np.cos(alpha)]
    ).reshape(3, 3)
    Rot = Rot_z @ Rot_y @ Rot_x
    return Rot, np.array([x, y, z])


def _fit_transform_GN(A, B, N):
    """Find a transformation that fits two point clouds onto each other using Gauss-Newton method
    for computing the least squares solution"""

    size_A = A.shape[1]
    size_B = B.shape[1]

    # Gauss-Newton Method a=p b=x
    H = np.zeros((6, 6))  # Hessian
    g = np.zeros((6, 1))  # gradient
    euler_array = np.zeros(6)
    Rot, trans = _set_rot_trans(euler_array)  # rotation & transformation
    chi = 0

    for i in range(min(size_A, size_B)):
        a, b, n = A[:, i], B[:, i], N[:, i]
        Rot_a = Rot @ a
        e = (Rot_a.reshape(3) + trans - b).dot(n)
        J = _plane_Jacobian(Rot_a, n)

        H += J.T @ J
        g += J.T * e
        chi += np.linalg.norm(e)

    update = -np.linalg.inv(H) @ g

    euler_array = euler_array + update.reshape(6)
    R, t = _set_rot_trans(euler_array)

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def _p_2_p_GN(A, B):
def _p_2_p_GN(A, B):
    """Find a transformation that fits two point clouds onto each other using point to point Gauss-Newton method
    for computing the least squares solution"""

    size_A = A.shape[1]
    size_B = B.shape[1]

    # Gauss-Newton Method
    H = np.zeros((6, 6))  # Hessian
    g = np.zeros((6, 1))  # gradient
    euler_array = np.zeros(6)
    Rot, trans = _set_rot_trans(euler_array)  # rotation & transformation
    chi = 0

    for i in range(min(size_A, size_B)):
        a, b = A[:, i], B[:, i]
        Rot_a = Rot @ a
        e = (Rot_a + trans).reshape(3) - b
        J = _point_Jacobian(Rot_a)
        H += J.T @ J
        g += J.T @ (e.reshape(3, 1))
        chi += np.linalg.norm(e)

    update = -np.linalg.inv(H) @ g
    euler_array = euler_array + update.reshape(6)

    R, t = _set_rot_trans(euler_array)
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def _fit_transform_LM(A, B, N):
    """Find a transformation that fits two point clouds onto each other using Levenberg-Marquardt method
    for computing the least squares solution"""

    size_A = A.shape[1]
    size_B = B.shape[1]

    lmda = 1e-2

    # Levenberg-Marquardt Method
    H = np.zeros((6, 6))  # Hessian
    g = np.zeros((6, 1))  # gradient
    euler_array = np.zeros(6)
    Rot, trans = _set_rot_trans(euler_array)  # rotation & transformation
    chi = 0

    for i in range(min(size_A, size_B)):
        a, b, n = A[:, i], B[:, i], N[:, i]
        Rot_a = Rot.dot(a)
        e = (Rot_a.reshape(3) + trans - b).dot(n)
        J = _plane_Jacobian(Rot_a, n)
        H += J.T.dot(J)
        g += J.T * e
        chi += np.linalg.norm(e)

    H += lmda * H * np.eye(6)
    update = -np.linalg.inv(H) @ g

    euler_array_new = euler_array + update.reshape(6)
    R, t = _set_rot_trans(euler_array)
    chi_new = 0
    for i in range(min(size_A, size_B)):
        e_new = ((R @ A[:, i]).reshape(3) + t - B[:, i]).dot(N[:, i])
        chi_new += np.linalg.norm(e_new)

    if chi_new > chi:
        lmda *= 10

    else:
        euler_array = euler_array_new
        lmda /= 10

    R, t = _set_rot_trans(euler_array)
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


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
        neighbor_arrays = np.asarray(reference_epoch.kdtree.nearest_neighbors(cloud))
        indices, distances = np.split(neighbor_arrays, 2, axis=0)

        indices = np.squeeze(indices.astype(int))
        distances = np.squeeze(distances)

        # Calculate a transform and apply it
        T = _fit_transform(
            cloud, reference_epoch.cloud[indices, :], reduction_point=reduction_point
        )
        _py4dgeo.transform_pointcloud_inplace(cloud, T, reduction_point)

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
            reference_epoch.kdtree.nearest_neighbors(trans_epoch.cloud)
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

        _py4dgeo.transform_pointcloud_inplace(trans_epoch.cloud, T, reduction_point)
        _py4dgeo.transform_pointcloud_inplace(trans_epoch.normals, T, reduction_point)

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


def point_to_plane_icp_LM(
    reference_epoch, epoch, max_iterations=50, tolerance=0.00001, reduction_point=None
):
    """Perform a point to plane Iterative Closest Point algorithm (ICP), based on Levenberg-Marquardt method for computing the least squares solution

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
    trans_epoch = deepcopy(epoch)

    prev_error = 0

    for _ in range(max_iterations):
        neighbor_arrays = np.asarray(
            reference_epoch.kdtree.nearest_neighbors(trans_epoch.cloud)
        )
        indices, distances = np.split(neighbor_arrays, 2, axis=0)

        indices = np.squeeze(indices.astype(int))
        distances = np.squeeze(distances)

        # Calculate a transform and apply it
        T = _fit_transform_LM(
            trans_epoch.cloud.transpose(1, 0),
            reference_epoch.cloud[indices, :].transpose(1, 0),
            reference_epoch.normals[indices, :].transpose(1, 0),
        )
        _py4dgeo.transform_pointcloud_inplace(trans_epoch.cloud, T, reduction_point)
        # Determine convergence
        mean_error = np.mean(np.sqrt(distances))

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    normals = Epoch.calculate_normals(trans_epoch)
    return Transformation(
        affine_transformation=_fit_transform_LM(
            epoch.cloud.transpose(1, 0),
            trans_epoch.cloud.transpose(1, 0),
            normals.transpose(1, 0),
        ),
        reduction_point=reduction_point,
    )


def p_to_p_icp(
    reference_epoch, epoch, max_iterations=50, tolerance=0.00001, reduction_point=None
):
    """Perform a point to point Iterative Closest Point algorithm (ICP), based on Gauss-Newton method

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
        neighbor_arrays = np.asarray(reference_epoch.kdtree.nearest_neighbors(cloud))
        indices, distances = np.split(neighbor_arrays, 2, axis=0)

        indices = np.squeeze(indices.astype(int))
        distances = np.squeeze(distances)

        # Calculate a transform and apply it
        T = _p_2_p_GN(
            cloud.transpose(1, 0), reference_epoch.cloud[indices, :].transpose(1, 0)
        )
        _py4dgeo.transform_pointcloud_inplace(cloud, T, reduction_point)

        # Determine convergence
        mean_error = np.mean(np.sqrt(distances))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return Transformation(
        affine_transformation=_p_2_p_GN(
            epoch.cloud.transpose(1, 0), cloud.transpose(1, 0)
        ),
        reduction_point=reduction_point,
    )


def compute_covariance_matrix(points):
    """Compute the covariance matrix of a set of points."""

    # Convert the list of points to a NumPy array for efficient operations
    data = np.array(points)

    # Check if the data is empty or has only one point
    if len(data) < 2:
        raise ValueError("Insufficient data points to compute covariance matrix")

    # Subtract the mean along each column (variable)
    centered_data = data - np.mean(data, axis=0)

    # Compute the covariance matrix efficiently
    covariance_matrix = np.dot(centered_data.T, centered_data) / (len(data) - 1)

    return covariance_matrix



def solve_plane_parameters(covariance_matrix):
    """Solve the parameters of a plane given its covariance matrix."""
    _, eigen_vectors = np.linalg.eigh(covariance_matrix)

    # Extract the eigenvector with the smallest eigenvalue
    normal_vector = eigen_vectors[:, 0]

    return normal_vector



def compute_cor_distance(epoch1, epoch2, clouds_pc1, check_size):
    """
    Compute the correspondence distance between two point clouds.
    Parameters:
    - epoch1: The epoch consists of core points of reference epoch.
    - epoch2: The epoch consists of core points of epoch to be transformed.
    - clouds_pc1: The supervoxels of the reference epoch.
    - check_size: The number of points in the reference epoch.
    Returns:
    - P2Pdist: The correspondence distance between the two point clouds.
    """

    neighbor_arrays = np.asarray(epoch1.kdtree.nearest_neighbors(epoch2.cloud))
    indices, distances = np.split(neighbor_arrays, 2, axis=0)

    indices = np.squeeze(indices.astype(int))

    distances = np.squeeze(distances)
    P2Pdist = []
    for i in range(len(epoch2.cloud)):
        if len(epoch1.cloud) != check_size:
            cov_matrix = compute_covariance_matrix(clouds_pc1[indices[i]])
            normal_vector = solve_plane_parameters(cov_matrix)
            nmlx, nmly, nmlz = normal_vector[:3]
            Dis_x = epoch1.cloud[indices[i]][0] - epoch2.cloud[i][0]
            Dis_y = epoch1.cloud[indices[i]][1] - epoch2.cloud[i][1]
            Dis_z = epoch1.cloud[indices[i]][2] - epoch2.cloud[i][2]
            res_dis = np.abs(Dis_x * nmlx + Dis_y * nmly + Dis_z * nmlz)
        else:
            res_dis = np.sqrt(distances[i])

        P2Pdist.append(res_dis)
    return P2Pdist


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
    neighbor_arrays = np.asarray(epoch1.kdtree.nearest_neighbors(epoch2.cloud))
    indices, distances = np.split(neighbor_arrays, 2, axis=0)
    distances = np.squeeze(distances)

    if indices.size > 0:
        # Calculate mean distance
        mean_dis = np.mean(np.sqrt(distances))

        # Calculate standard deviation
        std_dis = np.sqrt(np.mean((mean_dis - distances) ** 2))

        dis_threshold = mean_dis + 1.0 * std_dis

    return dis_threshold


def registration_method(
    reference_epoch,
    epoch,
    dis_threshold,
    lmdd,
    res1,
    res2,
    k=2,
    minSVPvalue=10,
    reduction_point=None,
):
    """Perform a registration method"""

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
            "Normals for this Reference Epoch have not been calculated! Please use Epoch.calculate_normals or load externally calculated normals."
        )


    # Apply the default for the registration point
    if reduction_point is None:
        reduction_point = np.array([0, 0, 0])


    if dis_threshold <= lmdd:
        dis_threshold = lmdd
    # For updating DT
    # For updating DT
    dtSeries = []
    dtSeries.append(dis_threshold)
    transMatFinal = np.identity(4)  # Identity matrix for initial transMatFinal
    stage3 = stage4 = 0

    clouds_pc1, _, centroids_pc1, _ = _py4dgeo.segment_pc_in_supervoxels(
        reference_epoch,
        reference_epoch.kdtree,
        reference_epoch.normals,
        res1,
        k,
        minSVPvalue,
    )
    (
        clouds_pc2,
        normals2,
        centroids_pc2,
        boundary_points_pc2,
    ) = _py4dgeo.segment_pc_in_supervoxels(
        epoch, epoch.kdtree, epoch.normals, res2, k, minSVPvalue
    )

    centroids_pc1 = as_epoch(np.array(centroids_pc1))
    centroids_pc1.build_kdtree()
    centroids_pc2 = as_epoch(np.array(centroids_pc2))
    centroids_pc2.build_kdtree()

    boundary_points_pc2 = np.concatenate(boundary_points_pc2, axis=0)
    boundary_points_pc2 = as_epoch(boundary_points_pc2)
    boundary_points_pc2.build_kdtree()

    steps = 0
    while stage4 == 0:
        # Calculation CT2-CT1
        cor_dist_ct = compute_cor_distance(
            centroids_pc1, centroids_pc2, clouds_pc1, len(reference_epoch.cloud)
        )
        # Calculation BP2-CT1
        cor_dist_bp = compute_cor_distance(
            centroids_pc1, boundary_points_pc2, clouds_pc1, len(reference_epoch.cloud)
        )
        # calculation BP2- CP1
        cor_dist_pc = compute_cor_distance(
            reference_epoch, boundary_points_pc2, clouds_pc1, len(reference_epoch.cloud)
        )

        stablePC2 = []  # Stable supervoxels
        normPC2 = []  # Stable supervoxel's normals
        unstablePC2 = []  # Unstable supervoxels
        stablePC2 = []  # Stable supervoxels
        normPC2 = []  # Stable supervoxel's normals
        unstablePC2 = []  # Unstable supervoxels

        dt_point = dis_threshold + 2 * res1
        stableSVnum = 0  # Number of stable SV in PC2

        for i, cloud in enumerate(centroids_pc2.cloud):
            if cor_dist_ct[i] < dis_threshold and all(
                cor_dist_bp[j + 6 * i] < dis_threshold
                and cor_dist_pc[j + 6 * i] < dt_point
                for j in range(6)
            ):
                stablePC2.append(clouds_pc2[i])
                normPC2.append(normals2[i])
                stableSVnum += 1
            else:
                unstablePC2.append(cloud)

        stablePC2 = np.vstack(stablePC2)
        stablePC2 = as_epoch(stablePC2)
        normPC2 = np.vstack(normPC2)
        stablePC2.normals_attachment(normPC2)

        # ICP
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
        initial_min_bound, initial_max_bound = calculate_bounding_box(epoch.cloud)
        max_bb_change = calculate_bounding_box_change(
            initial_min_bound, initial_max_bound, trans_mat_cur
        )
        max_bb_change = calculate_bounding_box_change(
            initial_min_bound, initial_max_bound, trans_mat_cur
        )

        # update DT
        if stage3 == 0 and max_bb_change < 2 * lmdd:
        # update DT
        if stage3 == 0 and max_bb_change < 2 * lmdd:
            stage3 = 1
        elif dis_threshold == lmdd:
            stage4 = 1


        if stage3 == 0:
            dis_threshold = calculate_dis_threshold(reference_epoch, stablePC2)
            if dis_threshold <= lmdd:
                dis_threshold = lmdd

        if stage3 == 1 and stage4 == 0:
            dis_threshold = 0.8 * dis_threshold
            dis_threshold = 0.8 * dis_threshold
            if dis_threshold <= lmdd:
                dis_threshold = lmdd


        # update values and apply changes
        dtSeries.append(dis_threshold)
        transMatFinal = trans_mat_cur @ transMatFinal

        _py4dgeo.transform_pointcloud_inplace(
            epoch.cloud, transMatFinal, reduction_point
        )

        _py4dgeo.transform_pointcloud_inplace(
            centroids_pc2.cloud, transMatFinal, reduction_point
        )

        _py4dgeo.transform_pointcloud_inplace(
            boundary_points_pc2.cloud, transMatFinal, reduction_point
        )

        for i in range(len(clouds_pc2)):
            _py4dgeo.transform_pointcloud_inplace(
                clouds_pc2[i], transMatFinal, reduction_point
            )

    return (
        Transformation(
            affine_transformation=transMatFinal,
            reduction_point=reduction_point,
        ),
        dtSeries,
    )

            _py4dgeo.transform_pointcloud_inplace(
                clouds_pc2[i], transMatFinal, reduction_point
            )

    return (
        Transformation(
            affine_transformation=transMatFinal,
            reduction_point=reduction_point,
        ),
        dtSeries,
    )
