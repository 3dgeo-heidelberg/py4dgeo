import numpy as np

import py4dgeo._py4dgeo as _py4dgeo

# import py4dgeo
#
print(_py4dgeo.__file__)


def _fit_transform(A, B):
    """Find a transformation that fits two point clouds onto each other"""

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

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
    print(_py4dgeo.__file__)
    # Make a copy of the cloud to be transformed.
    cloud = epoch.cloud.copy()

    prev_error = 0
    #
    for _ in range(max_iterations):
        neighbor_lists = reference_epoch.kdtree.nearest_neighbors(cloud)
        indices, distances = zip(*neighbor_lists)
        # Calculate a transform and apply it
        T = _fit_transform(cloud, reference_epoch.cloud[indices, :])
        _py4dgeo.transform_pointcloud_inplace(cloud, T, reduction_point)

        # Determine convergence
        mean_error = np.mean(np.sqrt(distances))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return _fit_transform(epoch.cloud, cloud)
