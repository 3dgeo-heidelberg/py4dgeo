import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree
from tqdm import tqdm
from py4dgeo.util import Py4DGeoError
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class PBM3C2:
    """
    Correspondence-driven plane-based M3C2 for lower uncertainty in 3D topographic change quantification.

    This class implements the PBM3C2 algorithm as described in the paper by Zahs et al. (2022).
    """

    def __init__(self, registration_error=0.0):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.registration_error = registration_error
        self.epoch0_segment_metrics = None
        self.epoch1_segment_metrics = None
        self.epoch0_segments = None
        self.epoch1_segments = None
        self.correspondences = None

    @staticmethod
    def preprocess_epochs(epoch0, epoch1, correspondences_file):
        """
        Check and process the Segment IDs of Epoch objects to ensure they are globally unique.
        Simultaneously, adjust the correspondence files used for training based on the ID offsets.
        """
        print("Checking if the Segment ID is unique...")
        ids0 = np.unique(epoch0.additional_dimensions["segment_id"])
        ids1 = np.unique(epoch1.additional_dimensions["segment_id"])

        # Read CSV file
        correspondences_df = pd.read_csv(correspondences_file, header=None)

        # Check if CSV has header (column names)
        # If first row contains non-numeric values, it's likely a header
        first_row = correspondences_df.iloc[0]
        if not all(
            pd.api.types.is_numeric_dtype(type(val)) or isinstance(val, (int, float))
            for val in first_row
        ):
            raise Py4DGeoError(
                f"The correspondence file '{correspondences_file}' appears to contain column headers. "
                "Please provide a CSV file without headers (header=False when saving)."
            )

        # Additional check: if any value in first 2 columns is not numeric
        try:
            pd.to_numeric(correspondences_df.iloc[:, 0])
            pd.to_numeric(correspondences_df.iloc[:, 1])
        except (ValueError, TypeError):
            raise Py4DGeoError(
                f"The correspondence file '{correspondences_file}' contains non-numeric values in the first two columns. "
                "This file should not have column headers. Please save the file with header=False."
            )

        if not set(ids0).isdisjoint(set(ids1)):
            print("Detected overlapping Segment IDs, performing preprocessing...")

            max_id_epoch0 = ids0.max()
            offset = max_id_epoch0 + 1

            # update epoch1 segment_id
            new_ids_epoch1 = epoch1.additional_dimensions["segment_id"] + offset
            epoch1.additional_dimensions["segment_id"] = new_ids_epoch1

            # update correspondences_df (second column)
            correspondences_df.iloc[:, 1] = correspondences_df.iloc[:, 1] + offset

            print(
                f"Preprocessing complete. Epoch1 Segment IDs have been offset by {offset}."
            )
        else:
            print("No overlapping Segment IDs detected, no preprocessing needed.")

        return epoch0, epoch1, correspondences_df

    def _get_segments(self, epoch):
        """
        Extracts individual segments from an epoch, correctly handling the data
        structure where additional_dimensions is a dictionary of NumPy arrays.
        """
        add_dims = epoch.additional_dimensions
        segment_id_array = add_dims["segment_id"]
        unique_segment_ids = np.unique(segment_id_array)

        segments_dict = {}
        for seg_id in unique_segment_ids:
            indices = np.where(segment_id_array == seg_id)[0]

            if len(indices) > 0:
                segments_dict[seg_id] = {"points": epoch.cloud[indices]}

        return segments_dict

    def _create_segment_metrics(self, segments):
        """Creates a DataFrame with metrics for each segment."""
        metrics_list = []
        for segment_id, data in tqdm(segments.items(), desc="Extracting Features"):
            points = data["points"]
            if points.shape[0] < 3:
                continue

            cog = np.mean(points, axis=0)

            centered_points = points - cog
            cov_matrix = np.cov(centered_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            sort_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sort_indices]
            eigenvectors = eigenvectors[:, sort_indices]

            normal = eigenvectors[:, 2]

            distances_to_plane = np.dot(centered_points, normal)
            roughness = np.std(distances_to_plane)

            e1, e2, e3 = eigenvalues
            sum_eigenvalues = e1 + e2 + e3
            if sum_eigenvalues == 0:
                linearity = planarity = sphericity = 0
            else:
                linearity = (e1 - e2) / sum_eigenvalues
                planarity = (e2 - e3) / sum_eigenvalues
                sphericity = e3 / sum_eigenvalues

            metrics_list.append(
                {
                    "segment_id": segment_id,
                    "cog_x": cog[0],
                    "cog_y": cog[1],
                    "cog_z": cog[2],
                    "normal_x": normal[0],
                    "normal_y": normal[1],
                    "normal_z": normal[2],
                    "linearity": linearity,
                    "planarity": planarity,
                    "sphericity": sphericity,
                    "roughness": roughness,
                    "num_points": points.shape[0],
                }
            )

        return pd.DataFrame(metrics_list).set_index("segment_id")

    def _create_feature_array(self, df_t1, df_t2, correspondences):
        features = []
        for _, row in correspondences.iterrows():
            id1, id2 = int(row.iloc[0]), int(row.iloc[1])

            if id1 not in df_t1.index or id2 not in df_t2.index:
                continue

            metrics1 = df_t1.loc[id1]
            metrics2 = df_t2.loc[id2]

            cog1 = metrics1[["cog_x", "cog_y", "cog_z"]].values
            cog2 = metrics2[["cog_x", "cog_y", "cog_z"]].values
            cog_dist = np.linalg.norm(cog1 - cog2)

            normal1 = metrics1[["normal_x", "normal_y", "normal_z"]].values
            normal2 = metrics2[["normal_x", "normal_y", "normal_z"]].values
            dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
            normal_angle = np.arccos(dot_product)

            roughness_diff = abs(metrics1["roughness"] - metrics2["roughness"])

            features.append([cog_dist, normal_angle, roughness_diff])

        return np.array(features)

    def train(self, correspondences):
        positives = correspondences[correspondences[2] == 1]
        negatives = correspondences[correspondences[2] == 0]

        X_pos = self._create_feature_array(
            self.epoch0_segment_metrics, self.epoch1_segment_metrics, positives
        )
        X_neg = self._create_feature_array(
            self.epoch0_segment_metrics, self.epoch1_segment_metrics, negatives
        )

        if X_pos.shape[0] == 0 or X_neg.shape[0] == 0:
            raise ValueError("Training data is missing positive or negative examples.")

        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))

        self.clf.fit(X, y)

    def apply(self, apply_ids, search_radius=1.0):
        epoch1_cogs = self.epoch1_segment_metrics[["cog_x", "cog_y", "cog_z"]].values
        kdtree = KDTree(epoch1_cogs)

        found_correspondences = []

        for apply_id in tqdm(apply_ids, desc="Applying Classifier"):
            if apply_id not in self.epoch0_segment_metrics.index:
                continue

            cog0 = self.epoch0_segment_metrics.loc[apply_id][
                ["cog_x", "cog_y", "cog_z"]
            ].values.reshape(1, -1)
            indices = kdtree.query_radius(cog0, r=search_radius)[0]

            if len(indices) == 0:
                continue

            candidate_ids = self.epoch1_segment_metrics.index[indices]

            apply_df = pd.DataFrame(
                {"id1": [apply_id] * len(candidate_ids), "id2": candidate_ids}
            )
            X_apply = self._create_feature_array(
                self.epoch0_segment_metrics, self.epoch1_segment_metrics, apply_df
            )

            if len(X_apply) == 0:
                continue

            probabilities = self.clf.predict_proba(X_apply)[:, 1]
            best_match_idx = np.argmax(probabilities)

            found_correspondences.append([apply_id, candidate_ids[best_match_idx]])

        self.correspondences = pd.DataFrame(
            found_correspondences, columns=["epoch0_segment_id", "epoch1_segment_id"]
        )

    def _calculate_m3c2(self, segment1_id, segment2_id):
        metrics1 = self.epoch0_segment_metrics.loc[segment1_id]
        metrics2 = self.epoch1_segment_metrics.loc[segment2_id]

        cog1 = metrics1[["cog_x", "cog_y", "cog_z"]].values
        cog2 = metrics2[["cog_x", "cog_y", "cog_z"]].values
        normal1 = metrics1[["normal_x", "normal_y", "normal_z"]].values

        dist = np.dot(cog2 - cog1, normal1)

        sigma1_sq = metrics1["roughness"] ** 2
        sigma2_sq = metrics2["roughness"] ** 2
        n1 = metrics1["num_points"]
        n2 = metrics2["num_points"]

        if n1 == 0 or n2 == 0:
            lod = np.nan
        else:
            lod = (
                1.96 * np.sqrt(sigma1_sq / n1 + sigma2_sq / n2)
                + self.registration_error
            )

        return dist, lod

    def run(self, epoch0, epoch1, correspondences_file, apply_ids, search_radius=1.0):
        # Preprocess Epochs and Training Data (incase of overlapping Segment IDs)
        print("Preprocessing epochs and correspondences...")
        epoch0, epoch1, correspondences_for_training = self.preprocess_epochs(
            epoch0, epoch1, correspondences_file
        )

        print("Step 1: Loading and processing segments...")
        self.epoch0_segments = self._get_segments(epoch0)
        self.epoch1_segments = self._get_segments(epoch1)

        print("Step 2: Extracting features...")
        self.epoch0_segment_metrics = self._create_segment_metrics(self.epoch0_segments)
        self.epoch1_segment_metrics = self._create_segment_metrics(self.epoch1_segments)

        print("Step 3: Training classifier...")
        self.train(correspondences_for_training)

        print("Step 4: Finding correspondences...")
        self.apply(apply_ids=apply_ids, search_radius=search_radius)

        print("Step 5: Calculating distances...")
        distances, uncertainties = [], []
        if self.correspondences is None or self.correspondences.empty:
            print("Warning: No correspondences were found.")
        else:
            for _, row in self.correspondences.iterrows():
                id1, id2 = row["epoch0_segment_id"], row["epoch1_segment_id"]
                dist, lod = self._calculate_m3c2(id1, id2)
                distances.append(dist)
                uncertainties.append(lod)

            self.correspondences["distance"] = distances
            self.correspondences["uncertainty"] = uncertainties

        return self.correspondences

    def visualize_correspondences(
        self,
        epoch0_segment_id=None,
        show_all=False,
        num_samples=10,
        figsize=(12, 10),
        elev=30,
        azim=45,
    ):
        """
        Visualize matched plane segments and their correspondences.

        Priority logic:
        1. If 'epoch0_segment_id' is provided, plots only that specific correspondence (zoomed in).
        2. Else if 'show_all' is True, plots all correspondences (risk of poor performance).
        3. Else (default), plots 'num_samples' random correspondences.

        Parameters
        ----------
        epoch0_segment_id : int, optional
            The specific 'epoch0_segment_id' to visualize (default: None).
        show_all : bool, optional
            Flag to force plotting all correspondences (default: False).
        num_samples : int, optional
            Number of random correspondences to visualize if epoch0_segment_id is None and show_all is False (default: 10).
        figsize : tuple, optional
            Figure size (width, height) in inches (default: (12, 10)).
        elev : float, optional
            Elevation angle for 3D view in degrees (default: 30).
        azim : float, optional
            Azimuth angle for 3D view in degrees (default: 45).
        """

        if self.correspondences is None or self.correspondences.empty:
            raise ValueError("No correspondences found. Run run() first.")

        n_corr = len(self.correspondences)
        zoom_in = False  # Flag to control axis scaling

        # --- Sample Selection Logic ---
        if epoch0_segment_id is not None:
            # Mode 1: Plot specific ID
            corr_sample = self.correspondences[
                self.correspondences["epoch0_segment_id"] == epoch0_segment_id
            ]
            if corr_sample.empty:
                raise ValueError(
                    f"Epoch 0 Segment ID {epoch0_segment_id} not found in correspondences."
                )
            title_text = f"PBM3C2 Correspondence for Segment ID {epoch0_segment_id}"
            zoom_in = True  # Zoom in on the single segment

        elif show_all:
            # Mode 2: Plot all (user accepts risk)
            corr_sample = self.correspondences
            title_text = f"PBM3C2 Correspondences (showing all {n_corr})"

        else:
            # Mode 3: Plot random sample (default, safe)
            if n_corr > num_samples:
                sample_indices = np.random.choice(n_corr, num_samples, replace=False)
                corr_sample = self.correspondences.iloc[sample_indices]
                title_text = (
                    f"PBM3C2 Correspondences (showing {num_samples} of {n_corr})"
                )
            else:
                # Not enough data to sample, just show all
                corr_sample = self.correspondences
                title_text = f"PBM3C2 Correspondences (showing all {n_corr})"

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Helper class for 3D arrows
        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                super().__init__((0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def do_3d_projection(self, renderer=None):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                return np.min(zs)

        epoch0_plotted = False
        epoch1_plotted = False

        for idx, row in corr_sample.iterrows():
            id0 = row["epoch0_segment_id"]
            id1 = row["epoch1_segment_id"]

            if id0 not in self.epoch0_segments or id1 not in self.epoch1_segments:
                continue

            points0 = self.epoch0_segments[id0]["points"]
            points1 = self.epoch1_segments[id1]["points"]

            # Plot epoch0 segment (blue)
            label0 = "Epoch 0" if not epoch0_plotted else ""
            ax.scatter(
                points0[:, 0],
                points0[:, 1],
                points0[:, 2],
                c="blue",
                s=5,
                alpha=0.3,
                label=label0,
            )
            if label0:
                epoch0_plotted = True

            # Plot epoch1 segment (red)
            label1 = "Epoch 1" if not epoch1_plotted else ""
            ax.scatter(
                points1[:, 0],
                points1[:, 1],
                points1[:, 2],
                c="red",
                s=5,
                alpha=0.3,
                label=label1,
            )
            if label1:
                epoch1_plotted = True

            cog0 = self.epoch0_segment_metrics.loc[id0][
                ["cog_x", "cog_y", "cog_z"]
            ].values
            cog1 = self.epoch1_segment_metrics.loc[id1][
                ["cog_x", "cog_y", "cog_z"]
            ].values

            # Plot CoGs
            ax.scatter(
                *cog0, c="darkblue", s=100, marker="o", edgecolors="black", linewidths=2
            )
            ax.scatter(
                *cog1, c="darkred", s=100, marker="o", edgecolors="black", linewidths=2
            )

            # Draw arrow
            arrow = Arrow3D(
                [cog0[0], cog1[0]],
                [cog0[1], cog1[1]],
                [cog0[2], cog1[2]],
                mutation_scale=20,
                lw=2,
                arrowstyle="-|>",
                color="green",
                alpha=0.8,
            )
            ax.add_artist(arrow)

            # Add text label
            if "distance" in row:
                mid_point = (cog0 + cog1) / 2
                ax.text(
                    mid_point[0],
                    mid_point[1],
                    mid_point[2],
                    f'{row["distance"]:.3f}m',
                    fontsize=8,
                    color="green",
                    weight="bold",
                )

        # Setup plot labels and title
        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.set_zlabel("Z [m]", fontsize=12)
        ax.set_title(title_text, fontsize=14, weight="bold")

        if epoch0_plotted or epoch1_plotted:
            ax.legend(loc="upper right", fontsize=10)

        ax.view_init(elev=elev, azim=azim)

        if not zoom_in:
            # For multi-segment views, enforce equal aspect ratio
            # This makes the scene spatially correct, but shrinks arrows
            ax.set_box_aspect([1, 1, 1])
        # If zoom_in is True, we let matplotlib auto-scale the axes
        # to zoom in on the single correspondence

        plt.tight_layout()

        return fig, ax
