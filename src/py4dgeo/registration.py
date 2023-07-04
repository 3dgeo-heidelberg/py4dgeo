import numpy as np

import py4dgeo._py4dgeo as _py4dgeo


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


def iterative_closest_point(reference_epoch, epoch, max_iterations=20, tolerance=0.001):
    """Perform an Iterative Closest Point algorithm (ICP)

    :param reference_epoch:
    :param epoch:
    :param max_iterations:
    :param tolerance:
    """

    # Ensure that reference_epoch has its KDTree built
    if reference_epoch.kdtree.leaf_parameter() == 0:
        reference_epoch.build_kdtree()

    # Make a copy of the cloud to be transformed.
    cloud = epoch.cloud.copy()

    prev_error = 0

    for _ in range(max_iterations):
        indices, distances = reference_epoch.kdtree.nearest_neighbors(cloud)

        # Calculate a transform and apply it
        T = _fit_transform(cloud, reference_epoch.cloud[indices, :])
        _py4dgeo.transform_pointcloud_inplace(cloud, T)

        # Determine convergence
        mean_error = np.mean(np.sqrt(distances))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return _fit_transform(epoch.cloud, cloud)
