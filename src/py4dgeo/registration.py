from py4dgeo.util import Py4DGeoError
from py4dgeo.epoch import Epoch
from copy import deepcopy

import numpy as np

import _py4dgeo


def _plane_Jacobian(Rot_a, n):
    """Calculate Jacobian for point to plane method"""

    J = np.zeros((1, 6))
    J[:, 3:] = n
    J[:, :3] = -np.cross(Rot_a, n)
    return J


def _point_Jacobian(Rot_p):
    """Calculate Jacobian for point to plane method"""

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

    assert A.shape == B.shape == N.shape
    size = A.shape[1]

    # Gauss-Newton Method a=p b=x
    H = np.zeros((6, 6))  # Hessian
    g = np.zeros((6, 1))  # gradient
    euler_array = np.zeros(6)
    Rot, trans = _set_rot_trans(euler_array)  # rotation & transformation
    chi = 0

    for i in range(size):
        a, b, n = A[:, i], B[:, i], N[:, i]
        Rot_a = Rot @ a
        e = (Rot_a.reshape(3) + trans - b).dot(n)
        J = _plane_Jacobian(Rot_a, n)

        H += J.T @ J

        g += J.T * e

        chi += np.linalg.norm(e)

    update = -np.linalg.inv(H) @ g  # UPDATE is VERY SMALL!!!!!!!!!!!!!

    euler_array = euler_array + update.reshape(6)
    R, t = _set_rot_trans(euler_array)

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def _p_2_p_GN(A, B):
    """Find a transformation that fits two point clouds onto each other using point to point Gauss-Newton method
    for computing the least squares solution"""

    assert A.shape == B.shape
    size = A.shape[1]

    # Gauss-Newton Method
    H = np.zeros((6, 6))  # Hessian
    g = np.zeros((6, 1))  # gradient
    euler_array = np.zeros(6)
    Rot, trans = _set_rot_trans(euler_array)  # rotation & transformation
    chi = 0

    for i in range(size):
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

    assert A.shape == B.shape == N.shape
    size = A.shape[1]

    lmda = 1e-2

    # Levenberg-Marquardt Method
    H = np.zeros((6, 6))  # Hessian
    g = np.zeros((6, 1))  # gradient
    euler_array = np.zeros(6)
    Rot, trans = _set_rot_trans(euler_array)  # rotation & transformation
    chi = 0

    for i in range(size):
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
    for i in range(size):
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
    reference_epoch, epoch, max_iterations=20, tolerance=0.001, reduction_point=None
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
        neighbor_lists = reference_epoch.kdtree.nearest_neighbors(cloud)
        indices, distances = zip(*neighbor_lists)
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

    return _fit_transform(epoch.cloud, cloud)
