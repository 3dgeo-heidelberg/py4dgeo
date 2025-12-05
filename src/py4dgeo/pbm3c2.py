import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import cKDTree
from tqdm import tqdm
from py4dgeo.util import Py4DGeoError
from py4dgeo.epoch import Epoch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class PBM3C2:
    """
    Correspondence-driven plane-based M3C2 for lower uncertainty in 3D topographic change quantification.

    This class implements the PBM3C2 algorithm as described in Zahs et al. (2022).
    """

    def __init__(self, registration_error=0.0):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.registration_error = registration_error
        self.epoch0_segment_metrics = None
        self.epoch1_segment_metrics = None
        self.epoch0_segments = None
        self.epoch1_segments = None
        self.correspondences = None
        self.epoch0_id_mapping = None
        self.epoch1_id_mapping = None
        self.epoch0_reverse_mapping = None
        self.epoch1_reverse_mapping = None

    @staticmethod
    def preprocess_epochs(epoch0, epoch1, correspondences_file):
        """
        Assign globally unique segment IDs using independent sequential numbering.
        Map correspondence file IDs to the new ID scheme.

        Parameters
        ----------
        epoch0, epoch1 : Epoch
            Input epochs with segment_id in additional_dimensions
        correspondences_file : str
            Path to CSV file with correspondence data

        Returns
        -------
        tuple
            (processed_epoch0, processed_epoch1, remapped_correspondences,
             epoch0_id_mapping, epoch1_id_mapping,
             epoch0_reverse_mapping, epoch1_reverse_mapping)
        """
        print("Assigning globally unique segment IDs...")

        orig_ids0 = np.unique(epoch0.additional_dimensions["segment_id"])
        orig_ids1 = np.unique(epoch1.additional_dimensions["segment_id"])

        print(
            f"  Epoch 0: {len(orig_ids0)} unique segments (range: {orig_ids0.min()}-{orig_ids0.max()})"
        )
        print(
            f"  Epoch 1: {len(orig_ids1)} unique segments (range: {orig_ids1.min()}-{orig_ids1.max()})"
        )

        # Load correspondence file
        try:
            correspondences_arr = np.genfromtxt(
                correspondences_file, delimiter=",", dtype=np.float64
            )
            if correspondences_arr.ndim == 1:
                correspondences_arr = correspondences_arr.reshape(1, -1)
        except Exception as e:
            raise Py4DGeoError(
                f"Failed to read correspondence file '{correspondences_file}': {e}"
            )

        if correspondences_arr.shape[1] < 2:
            raise Py4DGeoError(
                f"The correspondence file must contain at least two columns (got {correspondences_arr.shape[1]})."
            )

        if not np.issubdtype(correspondences_arr.dtype, np.number):
            raise Py4DGeoError(
                "The correspondence file appears to contain non-numeric data."
            )

        corr_ids0 = correspondences_arr[:, 0]
        corr_ids1 = correspondences_arr[:, 1]

        # Validate correspondence IDs
        invalid_ids0 = ~np.isin(corr_ids0, orig_ids0)
        invalid_ids1 = ~np.isin(corr_ids1, orig_ids1)

        if invalid_ids0.any():
            invalid_list = corr_ids0[invalid_ids0]
            print(
                f"  Warning: {invalid_ids0.sum()} epoch0 IDs in correspondences don't exist in epoch0 segments"
            )
            print(f"    First 10 invalid IDs: {invalid_list[:10]}")

        if invalid_ids1.any():
            invalid_list = corr_ids1[invalid_ids1]
            print(
                f"  Warning: {invalid_ids1.sum()} epoch1 IDs in correspondences don't exist in epoch1 segments"
            )
            print(f"    First 10 invalid IDs: {invalid_list[:10]}")

        # Filter invalid correspondences
        valid_mask = ~invalid_ids0 & ~invalid_ids1
        if not valid_mask.all():
            print(f"  Filtering out {(~valid_mask).sum()} invalid correspondence pairs")
            correspondences_arr = correspondences_arr[valid_mask]

        if len(correspondences_arr) == 0:
            raise Py4DGeoError(
                "No valid correspondences remain after filtering. "
                "Please verify that segment IDs in the correspondence file match those in the input epochs."
            )

        # Create new independent ID mappings
        new_ids0 = np.arange(1, len(orig_ids0) + 1, dtype=np.int64)
        epoch0_id_mapping = dict(zip(orig_ids0, new_ids0))
        epoch0_reverse_mapping = dict(zip(new_ids0, orig_ids0))

        start_id_epoch1 = len(orig_ids0) + 1
        new_ids1 = np.arange(
            start_id_epoch1, start_id_epoch1 + len(orig_ids1), dtype=np.int64
        )
        epoch1_id_mapping = dict(zip(orig_ids1, new_ids1))
        epoch1_reverse_mapping = dict(zip(new_ids1, orig_ids1))

        print(f"  Assigned new IDs for Epoch 0: 1 to {len(orig_ids0)}")
        print(
            f"  Assigned new IDs for Epoch 1: {start_id_epoch1} to {start_id_epoch1 + len(orig_ids1) - 1}"
        )

        # Apply new IDs to epoch0
        old_seg_ids0 = epoch0.additional_dimensions["segment_id"]
        new_seg_ids0 = np.array([epoch0_id_mapping[sid] for sid in old_seg_ids0])
        new_add_dims0 = epoch0.additional_dimensions.copy()
        new_add_dims0["segment_id"] = new_seg_ids0
        new_epoch0 = Epoch(
            cloud=epoch0.cloud.copy(), additional_dimensions=new_add_dims0
        )

        # Apply new IDs to epoch1
        old_seg_ids1 = epoch1.additional_dimensions["segment_id"]
        new_seg_ids1 = np.array([epoch1_id_mapping[sid] for sid in old_seg_ids1])
        new_add_dims1 = epoch1.additional_dimensions.copy()
        new_add_dims1["segment_id"] = new_seg_ids1
        new_epoch1 = Epoch(
            cloud=epoch1.cloud.copy(), additional_dimensions=new_add_dims1
        )

        # Remap correspondence IDs
        remapped_corr = correspondences_arr.copy()
        remapped_corr[:, 0] = np.array(
            [epoch0_id_mapping[int(cid)] for cid in correspondences_arr[:, 0]]
        )
        remapped_corr[:, 1] = np.array(
            [epoch1_id_mapping[int(cid)] for cid in correspondences_arr[:, 1]]
        )

        print(f"  Remapped {len(remapped_corr)} correspondence pairs to new ID scheme")
        print("Preprocessing complete.\n")

        return (
            new_epoch0,
            new_epoch1,
            remapped_corr,
            epoch0_id_mapping,
            epoch1_id_mapping,
            epoch0_reverse_mapping,
            epoch1_reverse_mapping,
        )

    def _get_segments(self, epoch):
        """Extract individual segments from an epoch."""
        segment_id_array = epoch.additional_dimensions["segment_id"]
        unique_segment_ids = np.unique(segment_id_array)

        segments_dict = {}
        for seg_id in unique_segment_ids:
            indices = np.where(segment_id_array == seg_id)[0]
            if len(indices) > 0:
                segments_dict[seg_id] = {"points": epoch.cloud[indices].copy()}

        return segments_dict

    def _create_segment_metrics(self, segments):
        """Calculate geometric metrics for each segment."""
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
        """Create feature vectors for segment pairs."""
        features = []

        if isinstance(correspondences, np.ndarray):
            pairs = correspondences
        else:
            pairs = correspondences.values

        for row in pairs:
            id1, id2 = int(row[0]), int(row[1])

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
        """Train Random Forest classifier on labeled correspondences."""
        positives = correspondences[correspondences[:, 2] == 1]
        negatives = correspondences[correspondences[:, 2] == 0]

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

    def apply(self, apply_ids, search_radius=3.0):
        """Apply trained classifier to find correspondences."""
        epoch1_cogs = self.epoch1_segment_metrics[["cog_x", "cog_y", "cog_z"]].values
        kdtree = cKDTree(epoch1_cogs)
        found_correspondences = []

        for apply_id in tqdm(apply_ids, desc="Applying Classifier"):
            if apply_id not in self.epoch0_segment_metrics.index:
                continue

            cog0 = self.epoch0_segment_metrics.loc[apply_id][
                ["cog_x", "cog_y", "cog_z"]
            ].values
            indices = kdtree.query_ball_point(cog0, r=search_radius)

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

    def _calculate_cog_distance(self, segment1_id, segment2_id):
        """Calculate CoG distance and level of detection."""
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

    def run(self, epoch0, epoch1, correspondences_file, apply_ids, search_radius=3.0):
        """
        Execute complete PBM3C2 workflow.

        Parameters
        ----------
        epoch0, epoch1 : Epoch
            Input point cloud epochs with segment_id
        correspondences_file : str
            Path to CSV with training correspondences
        apply_ids : array-like
            Segment IDs to find correspondences for (using original IDs)
        search_radius : float
            Spatial search radius in meters

        Returns
        -------
        DataFrame
            Correspondences with distances and uncertainties (using original IDs)
        """
        print("=" * 60)
        print("PBM3C2 Processing Pipeline")
        print("=" * 60)

        print("\n[1/6] Preprocessing epochs and correspondences...")
        (
            epoch0_processed,
            epoch1_processed,
            correspondences_for_training,
            self.epoch0_id_mapping,
            self.epoch1_id_mapping,
            self.epoch0_reverse_mapping,
            self.epoch1_reverse_mapping,
        ) = self.preprocess_epochs(epoch0, epoch1, correspondences_file)

        print("[2/6] Loading and processing segments...")
        self.epoch0_segments = self._get_segments(epoch0_processed)
        self.epoch1_segments = self._get_segments(epoch1_processed)
        print(
            f"  Loaded {len(self.epoch0_segments)} segments from epoch 0, {len(self.epoch1_segments)} from epoch 1"
        )

        print("\n[3/6] Extracting geometric features...")
        self.epoch0_segment_metrics = self._create_segment_metrics(self.epoch0_segments)
        self.epoch1_segment_metrics = self._create_segment_metrics(self.epoch1_segments)
        print(
            f"  Computed metrics for {len(self.epoch0_segment_metrics)} + {len(self.epoch1_segment_metrics)} segments"
        )

        print("\n[4/6] Training Random Forest classifier...")
        self.train(correspondences_for_training)
        print(f"  Classifier trained on {len(correspondences_for_training)} pairs")

        print("\n[5/6] Finding correspondences...")
        remapped_apply_ids = [
            self.epoch0_id_mapping[orig_id]
            for orig_id in apply_ids
            if orig_id in self.epoch0_id_mapping
        ]

        if len(remapped_apply_ids) < len(apply_ids):
            print(
                f"  Warning: {len(apply_ids) - len(remapped_apply_ids)} apply_ids not found in epoch0"
            )

        self.apply(apply_ids=remapped_apply_ids, search_radius=search_radius)

        print("\n[6/6] Calculating M3C2 distances and uncertainties...")
        if self.correspondences is None or self.correspondences.empty:
            print("  Warning: No correspondences were found.")
            return self.correspondences

        distances, uncertainties = [], []
        for _, row in self.correspondences.iterrows():
            id1, id2 = row["epoch0_segment_id"], row["epoch1_segment_id"]
            dist, lod = self._calculate_cog_distance(id1, id2)
            distances.append(dist)
            uncertainties.append(lod)

        self.correspondences["distance"] = distances
        self.correspondences["uncertainty"] = uncertainties

        print("\n[Final] Mapping results back to original segment IDs...")
        self.correspondences["epoch0_original_id"] = self.correspondences[
            "epoch0_segment_id"
        ].map(self.epoch0_reverse_mapping)
        self.correspondences["epoch1_original_id"] = self.correspondences[
            "epoch1_segment_id"
        ].map(self.epoch1_reverse_mapping)

        cols = [
            "epoch0_original_id",
            "epoch1_original_id",
            "epoch0_segment_id",
            "epoch1_segment_id",
            "distance",
            "uncertainty",
        ]
        self.correspondences = self.correspondences[cols]

        print("=" * 60)
        print(f"Processing complete! Found {len(self.correspondences)} matches")
        print("=" * 60)

        return self.correspondences

    def visualize_correspondences(
        self,
        epoch0_segment_id=None,
        use_original_ids=True,
        show_all=False,
        num_samples=10,
        figsize=(12, 10),
        elev=30,
        azim=45,
    ):
        """
        Visualize matched plane segments and their correspondences.

        Parameters
        ----------
        epoch0_segment_id : int, optional
            Specific segment ID to visualize (original or new ID based on use_original_ids)
        use_original_ids : bool, optional
            If True, interpret epoch0_segment_id as original ID (default: True)
        show_all : bool, optional
            Plot all correspondences (default: False)
        num_samples : int, optional
            Number of random samples if show_all=False (default: 10)
        figsize : tuple, optional
            Figure size in inches
        elev : float, optional
            Elevation angle for 3D view in degrees
        azim : float, optional
            Azimuth angle for 3D view in degrees

        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axis objects
        """
        if self.correspondences is None or self.correspondences.empty:
            raise ValueError("No correspondences found. Run run() first.")

        n_corr = len(self.correspondences)
        zoom_in = False

        if epoch0_segment_id is not None:
            if (
                use_original_ids
                and "epoch0_original_id" in self.correspondences.columns
            ):
                corr_sample = self.correspondences[
                    self.correspondences["epoch0_original_id"] == epoch0_segment_id
                ]
                title_text = (
                    f"PBM3C2 Correspondence for Original Segment ID {epoch0_segment_id}"
                )
            else:
                corr_sample = self.correspondences[
                    self.correspondences["epoch0_segment_id"] == epoch0_segment_id
                ]
                title_text = f"PBM3C2 Correspondence for Segment ID {epoch0_segment_id}"

            if corr_sample.empty:
                raise ValueError(
                    f"Segment ID {epoch0_segment_id} not found in correspondences."
                )
            zoom_in = True

        elif show_all:
            corr_sample = self.correspondences
            title_text = f"PBM3C2 Correspondences (showing all {n_corr})"

        else:
            if n_corr > num_samples:
                sample_indices = np.random.choice(n_corr, num_samples, replace=False)
                corr_sample = self.correspondences.iloc[sample_indices]
                title_text = (
                    f"PBM3C2 Correspondences (showing {num_samples} of {n_corr})"
                )
            else:
                corr_sample = self.correspondences
                title_text = f"PBM3C2 Correspondences (showing all {n_corr})"

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

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

            ax.scatter(
                *cog0, c="darkblue", s=100, marker="o", edgecolors="black", linewidths=2
            )
            ax.scatter(
                *cog1, c="darkred", s=100, marker="o", edgecolors="black", linewidths=2
            )

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

        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.set_zlabel("Z [m]", fontsize=12)
        ax.set_title(title_text, fontsize=14, weight="bold")

        if epoch0_plotted or epoch1_plotted:
            ax.legend(loc="upper right", fontsize=10)

        ax.view_init(elev=elev, azim=azim)

        if not zoom_in:
            ax.set_box_aspect([1, 1, 1])

        plt.tight_layout()

        return fig, ax

    def get_original_ids(self, new_epoch0_id=None, new_epoch1_id=None):
        """Retrieve original segment IDs from new internal IDs."""
        result = []

        if new_epoch0_id is not None:
            orig_id = self.epoch0_reverse_mapping.get(new_epoch0_id, None)
            if orig_id is None:
                print(f"Warning: New epoch0 ID {new_epoch0_id} not found in mapping")
            result.append(orig_id)

        if new_epoch1_id is not None:
            orig_id = self.epoch1_reverse_mapping.get(new_epoch1_id, None)
            if orig_id is None:
                print(f"Warning: New epoch1 ID {new_epoch1_id} not found in mapping")
            result.append(orig_id)

        return tuple(result) if len(result) > 1 else (result[0] if result else None)

    def get_new_ids(self, orig_epoch0_id=None, orig_epoch1_id=None):
        """Retrieve new internal segment IDs from original IDs."""
        result = []

        if orig_epoch0_id is not None:
            new_id = self.epoch0_id_mapping.get(orig_epoch0_id, None)
            if new_id is None:
                print(
                    f"Warning: Original epoch0 ID {orig_epoch0_id} not found in mapping"
                )
            result.append(new_id)

        if orig_epoch1_id is not None:
            new_id = self.epoch1_id_mapping.get(orig_epoch1_id, None)
            if new_id is None:
                print(
                    f"Warning: Original epoch1 ID {orig_epoch1_id} not found in mapping"
                )
            result.append(new_id)

        return tuple(result) if len(result) > 1 else (result[0] if result else None)
