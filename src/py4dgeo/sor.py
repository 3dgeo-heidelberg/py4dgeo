import numpy as np
import py4dgeo


def statistical_outlier_removal(
    epoch: py4dgeo.Epoch,
    k: int = 8,
    std_dev_multiplier: float = 1.0,
    remove_points: bool = False,
) -> tuple[py4dgeo.Epoch, np.ndarray]:
    """
    Statistical Outlier Removal (SOR) filter.

    Parameters
    ----------
    epoch : py4dgeo.Epoch
        The point cloud to filter.
    k : int
        Number of nearest neighbors to consider.
    std_dev_multiplier : float
        Standard deviation multiplier for the outlier threshold.
    remove_points : bool
        If True, return an Epoch containing only inlier points.

    Returns
    -------
    epoch : py4dgeo.Epoch
        Original epoch, or filtered epoch if ``remove_points`` is True.
    inlier_outlier : np.ndarray
        Array with 0 for inliers and 1 for outliers. If ``remove_points`` is
        True, this array is aligned with the filtered epoch.
    """
    k = int(k)
    if k < 1:
        raise ValueError("SOR requires k >= 1")

    std_dev_multiplier = float(std_dev_multiplier)
    remove_points = bool(remove_points)

    # Convert to numpy array
    cloud = epoch.cloud
    N = cloud.shape[0]
    if N == 0:
        raise ValueError("Empty point cloud passed to SOR")

    n_neighbors = min(k, N - 1)

    if n_neighbors > 0:
        epoch.build_kdtree()

        distance_sum = np.zeros(N, dtype=float)
        # py4dgeo returns only the requested neighbor rank, so query ranks
        # 2..n+1 to skip each point itself and accumulate n distances.
        # This should be fixed at a later stage
        for neighbor_order in range(2, n_neighbors + 2):
            neighbor_arrays = np.asarray(
                epoch.kdtree.nearest_neighbors(cloud, neighbor_order)
            )
            _, distances = np.split(neighbor_arrays, 2, axis=0)
            distance_sum += np.sqrt(np.asarray(distances, dtype=float).reshape(-1))

        mean_dist_per_point = distance_sum / n_neighbors
    else:
        # Degenerate case: only one point, keep it
        mean_dist_per_point = np.zeros(N)

    # Global SOR threshold
    mu = mean_dist_per_point.mean()
    sigma = mean_dist_per_point.std()
    threshold = mu + std_dev_multiplier * sigma

    # Inliers
    mask = mean_dist_per_point <= threshold

    # Create inlier 0, outlier 1 array
    inlier_outlier = np.zeros(N, dtype=int)
    inlier_outlier[~mask] = 1

    print(f"SOR threshold: {round(threshold,3)}")

    if remove_points:
        normals = getattr(epoch, "_normals", None)
        additional_dimensions = epoch.additional_dimensions
        epoch = py4dgeo.Epoch(
            cloud[mask],
            normals=normals[mask] if normals is not None else None,
            additional_dimensions=(
                additional_dimensions[mask]
                if additional_dimensions is not None
                else None
            ),
            **epoch.metadata,
        )
        inlier_outlier = inlier_outlier[mask]

    return epoch, inlier_outlier
