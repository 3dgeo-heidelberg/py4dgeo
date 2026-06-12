import numpy as np
import py4dgeo


class SOR:
    def __init__(
        self,
        epoch: py4dgeo.Epoch,
        k: int = 8,
        std_dev_multiplier: float = 1.0,
        remove_points: bool = False,
    ):
        """
        Statistical Outlier Removal (SOR) filter.

        ----------
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
        """
        self.epoch = epoch
        self.k = int(k)
        if self.k < 1:
            raise ValueError("SOR requires k >= 1")

        self.std_dev_multiplier = float(std_dev_multiplier)
        self.remove_points = bool(remove_points)
        self.threshold = None

    def run(self) -> tuple[py4dgeo.Epoch, np.ndarray, np.ndarray]:
        """
        Run the SOR filter.

        Returns
        -------
        epoch : py4dgeo.Epoch
            Original epoch, or filtered epoch if ``remove_points`` is True.
        inlier_outlier : np.ndarray
            Array with 0 for inliers and 1 for outliers. If ``remove_points`` is
            True, this array is aligned with the filtered epoch.
        mean_distances : np.ndarray
            Mean distance to neighbors. If ``remove_points`` is True, this array
            is aligned with the filtered epoch.

        Notes
        -----
        The distance threshold used for classification is available as
        ``self.threshold`` after running the filter.
        """
        # Convert to numpy array
        cloud = self.epoch.cloud
        N = cloud.shape[0]
        if N == 0:
            raise ValueError("Empty point cloud passed to SOR")

        n_neighbors = min(self.k, N - 1)

        if n_neighbors > 0:
            self.epoch.build_kdtree()

            distance_sum = np.zeros(N, dtype=float)
            # py4dgeo returns only the requested neighbor rank, so query ranks
            # 2..n+1 to skip each point itself and accumulate n distances.
            # This should be fixed at a later stage
            for neighbor_order in range(2, n_neighbors + 2):
                neighbor_arrays = np.asarray(
                    self.epoch.kdtree.nearest_neighbors(cloud, neighbor_order)
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
        threshold = mu + self.std_dev_multiplier * sigma
        self.threshold = threshold

        # Inliers
        mask = mean_dist_per_point <= threshold

        # Create inlier 0, outlier 1 array
        inlier_outlier = np.zeros(N, dtype=int)
        inlier_outlier[~mask] = 1

        if self.remove_points:
            normals = getattr(self.epoch, "_normals", None)
            additional_dimensions = self.epoch.additional_dimensions
            self.epoch = py4dgeo.Epoch(
                cloud[mask],
                normals=normals[mask] if normals is not None else None,
                additional_dimensions=(
                    additional_dimensions[mask]
                    if additional_dimensions is not None
                    else None
                ),
                **self.epoch.metadata,
            )
            inlier_outlier = inlier_outlier[mask]
            mean_dist_per_point = mean_dist_per_point[mask]

        return self.epoch, inlier_outlier, mean_dist_per_point
