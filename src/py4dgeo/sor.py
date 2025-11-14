import numpy as np
from sklearn.neighbors import NearestNeighbors
import py4dgeo

class SOR:
    def __init__(self, epoch: py4dgeo.Epoch, k: int = 8, std_dev_multiplier: float = 1.0, remove_points: bool = False):
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
        Returns
        -------
        inlier_outlier : np.ndarray
            Array of shape (N,) with 0 for inliers and 1 for outliers.
        mean_distances : np.ndarray
            Array of shape (N,) with the mean distance to neighbors.
        threshold : float
            The distance threshold used to classify outliers.
        """
        self.epoch = epoch
        self.k = int(k)
        self.std_dev_multiplier = float(std_dev_multiplier)
        self.remove_points = remove_points

    def run(self) -> py4dgeo.Epoch:
        # Convert to numpy array
        N = self.epoch.cloud.shape[0]
        if N == 0:
            raise ValueError("Empty point cloud passed to SOR")

        # k+1 because the closest neighbor is the point itself
        n_neighbors = min(self.k + 1, N)

        # Build NN structure (kd_tree/ball_tree both fine)
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm="kd_tree",
            n_jobs=-1,
        ).fit(self.epoch.cloud)

        # distances: (N, n_neighbors)
        distances, indices = nn.kneighbors(self.epoch.cloud)

        # Drop self-distance in column 0
        if n_neighbors > 1:
            mean_dist_per_point = distances[:, 1:].mean(axis=1)
        else:
            # Degenerate case: only one point, keep it
            mean_dist_per_point = np.zeros(N)

        # Global SOR threshold
        mu = mean_dist_per_point.mean()
        sigma = mean_dist_per_point.std()
        threshold = mu + self.std_dev_multiplier * sigma

        # Inliers
        mask = mean_dist_per_point <= threshold

        # Create inlier 0, outlier 1 array
        inlier_outlier = np.zeros(N, dtype=int)
        inlier_outlier[~mask] = 1

        if self.remove_points:
            # Wrap back into Epoch
            ad = self.epoch.additional_dimensions
            self.epoch = py4dgeo.Epoch(self.epoch.cloud[mask],
                                        additional_dimensions=ad[mask])
            inlier_outlier = inlier_outlier[mask]
            mean_dist_per_point = mean_dist_per_point[mask]
        print("Threshold computed by SOR:", round(threshold,4))
        return self.epoch, inlier_outlier, mean_dist_per_point