# py4dgeo/scor.py
from __future__ import annotations

import logging
import typing
from typing import Optional, Tuple

import numpy as np

from py4dgeo.epoch import Epoch
from py4dgeo.util import (
    Py4DGeoError
)

try:
    import numba
    NUMBA_AVAILABLE = True
    prange = numba.prange  # safe alias
except Exception:
    print("Numba is not installed.")
    print("Use `pip install numba` to install it to enable faster computations in ScOR.")
    print("You can run py4dgeo.ScOR without Numba, but it may be slower.")
    NUMBA_AVAILABLE = False
    prange = range


logger = logging.getLogger("py4dgeo")

# ---------------------------
# Helpers
# ---------------------------

if NUMBA_AVAILABLE:
    @numba.njit(fastmath=True)
    def per_point_mean_distance(base_pts, neigh_pts):
        K = base_pts.shape[0]
        M = neigh_pts.shape[0]
        out = np.empty(K, np.float64)

        for k in range(K):
            s = 0.0
            for m in range(M):
                dx0 = neigh_pts[m, 0] - base_pts[k, 0]
                dx1 = neigh_pts[m, 1] - base_pts[k, 1]
                dx2 = neigh_pts[m, 2] - base_pts[k, 2]
                d2 = dx0*dx0 + dx1*dx1 + dx2*dx2
                s += d2**0.5
            out[k] = s / M

        return out
else:
    def per_point_mean_distance(base_pts, neigh_pts):
        """
        NumPy fallback: compute per-point mean distance from base_pts to neigh_pts.
        """
        base_pts = np.asarray(base_pts, dtype=np.float64)
        neigh_pts = np.asarray(neigh_pts, dtype=np.float64)

        K = base_pts.shape[0]
        M = neigh_pts.shape[0]

        if K == 0 or M == 0:
            return np.zeros(K, dtype=np.float64)

        # Broadcasted differences: (M, K, 3)
        diff = neigh_pts[:, None, :] - base_pts[None, :, :]

        # Squared distances: (M, K)
        # einsum avoids creating an extra temporary from diff*diff + sum
        dists_sq = np.einsum("mki,mki->mk", diff, diff, dtype=np.float64)

        # Distances: (M, K)
        dists = np.sqrt(dists_sq, dtype=np.float64)

        # Mean over neighbors for each base point → (K,)
        return dists.mean(axis=0, dtype=np.float64)

def _xyz_to_spherical(xyz: np.ndarray, origin: np.ndarray):
    O = np.asarray(origin, dtype=np.float64)
    d = xyz - O
    dxy = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    r = np.sqrt(dxy ** 2 + d[:, 2] ** 2)
    theta = np.arctan2(d[:, 2], dxy)        # elevation from XY-plane
    phi = np.arctan2(d[:, 1], d[:, 0])      # azimuth
    return r, theta, phi


def _bin_angles(phi: np.ndarray, theta: np.ndarray, scan_res_deg: float):
    if scan_res_deg <= 0:
        raise Py4DGeoError("scan_resolution must be > 0 degrees.")
    step = float(scan_res_deg)
    phi_idx = np.round(np.rad2deg(phi) / step).astype(np.int64)
    theta_idx = np.round(np.rad2deg(theta) / step).astype(np.int64)
    return phi_idx, theta_idx

# ---------------------------
# ScOR implementation
# ---------------------------
class ScOR:
    def __init__(
        self,
        search_point_epoch: Epoch,
        neighborhood_candidate_epochs: Optional[Tuple[Epoch, ...]] = None,
        scan_position: typing.Sequence[float] = (0.0, 0.0, 0.0),
        scan_resolution: float = 0.015,  # degrees
        increment: float = 0.5,
    ):
        self.search_point_epoch = search_point_epoch

        if neighborhood_candidate_epochs is None:
            neighborhood_candidate_epochs = (search_point_epoch,)
        elif isinstance(neighborhood_candidate_epochs, Epoch):
            neighborhood_candidate_epochs = (neighborhood_candidate_epochs,)

        self.neighborhood_candidate_epochs = neighborhood_candidate_epochs

        self.scan_position = np.asarray(scan_position, dtype=np.float64)
        self.scan_resolution = float(scan_resolution)
        self.increment = float(increment)
        self.default_max_distance = float(99999.0)

        # Will be filled by steps:
        self.sp_r = None
        self.sp_phi_idx = None
        self.sp_theta_idx = None
        self.nc_xyz = None
        self.nc_phi_idx = None
        self.nc_theta_idx = None

        # bin → slice maps (using sorted indices)
        self._sp_order = None
        self._nc_order = None
        self._sp_bin_to_slice = {}
        self._nc_bin_to_slice = {}

        self.neighbor_offsets = None
        self.point_neighbor_dic = None
        self.expected_distance = None
        self.observed_distance = None
        self.scor_values = None

        # Binning metadata
        self._phi_min = None
        self._theta_min = None
        self._theta_span = None

        # Bin → slice (now keyed by integer bin IDs)
        self._sp_order = None
        self._nc_order = None
        self._sp_bin_to_slice = {}
        self._nc_bin_to_slice = {}

        # Precomputed neighbor indices
        self._bin_neighbors_idx = {}

    # ---------------------------
    # Precomputation
    # ---------------------------

    def get_phi_theta_range_phiIndex_thetaIndex(self):
        # --- search points ---
        sp_xyz = self.search_point_epoch.cloud
        self.sp_r, sp_theta, sp_phi = _xyz_to_spherical(sp_xyz, self.scan_position)
        sp_phi_idx, sp_theta_idx = _bin_angles(sp_phi, sp_theta, self.scan_resolution)
        self.sp_phi_idx = sp_phi_idx
        self.sp_theta_idx = sp_theta_idx

        # --- neighborhood candidates (all epochs merged) ---
        self.nc_xyz = np.concatenate(
            [epoch.cloud for epoch in self.neighborhood_candidate_epochs]
        )
        nc_r, nc_theta, nc_phi = _xyz_to_spherical(self.nc_xyz, self.scan_position)
        nc_phi_idx, nc_theta_idx = _bin_angles(nc_phi, nc_theta, self.scan_resolution)
        self.nc_phi_idx = nc_phi_idx
        self.nc_theta_idx = nc_theta_idx

        # --- bin id encoding ---
        phi_min = int(min(sp_phi_idx.min(), nc_phi_idx.min()))
        theta_min = int(min(sp_theta_idx.min(), nc_theta_idx.min()))
        phi_max = int(max(sp_phi_idx.max(), nc_phi_idx.max()))
        theta_max = int(max(sp_theta_idx.max(), nc_theta_idx.max()))

        theta_span = (theta_max - theta_min + 1)

        self._phi_min = phi_min
        self._theta_min = theta_min
        self._theta_span = theta_span

        # Helper lambdas (vectorized via broadcasting)
        def encode_bins(phi_idx, theta_idx):
            return (phi_idx - phi_min) * theta_span + (theta_idx - theta_min)

        sp_bin_ids = encode_bins(sp_phi_idx, sp_theta_idx)
        nc_bin_ids = encode_bins(nc_phi_idx, nc_theta_idx)

        # --- search points: sort + unique bins ---
        sp_order = np.argsort(sp_bin_ids)
        sp_bins_sorted = sp_bin_ids[sp_order]

        uniq_sp, idx_sp, cnt_sp = np.unique(
            sp_bins_sorted, return_index=True, return_counts=True
        )
        self._sp_order = sp_order
        self._sp_bin_to_slice = {
            int(bin_id): slice(int(start), int(start + count))
            for bin_id, start, count in zip(uniq_sp, idx_sp, cnt_sp)
        }

        # --- neighbor candidates: sort + unique bins ---
        nc_order = np.argsort(nc_bin_ids)
        nc_bins_sorted = nc_bin_ids[nc_order]

        uniq_nc, idx_nc, cnt_nc = np.unique(
            nc_bins_sorted, return_index=True, return_counts=True
        )
        self._nc_order = nc_order
        self._nc_bin_to_slice = {
            int(bin_id): slice(int(start), int(start + count))
            for bin_id, start, count in zip(uniq_nc, idx_nc, cnt_nc)
        }

    # ---------------------------
    # Neighborhood topology
    # ---------------------------
    def build_neighborhoods(self):
        inc = self.increment

        if inc == 0.5:
            neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif inc == 1:
            neighbor_offsets = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ]
        elif inc > 1:
            inc_i = int(inc)
            offs = []
            for i in range(-inc_i, inc_i + 1):
                for j in range(-inc_i, inc_i + 1):
                    if i == 0 and j == 0:
                        continue
                    offs.append((i, j))
            neighbor_offsets = offs
        else:
            raise Py4DGeoError(
                "Increment value must be 0.5 or any integer value greater than 0."
            )

        self.neighbor_offsets = neighbor_offsets

        theta_span = self._theta_span
        nc_bin_to_slice = self._nc_bin_to_slice
        nc_order = self._nc_order

        # Precompute neighbor offsets in bin-id space
        offset_ids = np.array(
            [dp * theta_span + dt for dp, dt in neighbor_offsets],
            dtype=np.int64,
        )

        bin_neighbors_idx = {}

        # Work on numpy array of existing SP bin_ids for speed
        sp_bin_ids = np.fromiter(
            self._sp_bin_to_slice.keys(), dtype=np.int64,
            count=len(self._sp_bin_to_slice),
        )

        for bin_id in sp_bin_ids:
            neigh_idx_chunks = []
            # Try each neighbor bin (as int id)
            for off in offset_ids:
                s = nc_bin_to_slice.get(int(bin_id + off))
                if s is not None and s.start != s.stop:
                    neigh_idx_chunks.append(nc_order[s])

            if neigh_idx_chunks:
                bin_neighbors_idx[int(bin_id)] = np.concatenate(neigh_idx_chunks)

        self._bin_neighbors_idx = bin_neighbors_idx



    # ---------------------------
    # Expected distances
    # ---------------------------
    def get_expected_distances(self):
        # Expected multiplier from neighbor offsets (constant)
        all_offset_distances = np.sqrt(
            np.array([dx * dx + dy * dy for dx, dy in self.neighbor_offsets], dtype=np.float64)
        )
        exp_distance_multiplier = float(all_offset_distances.mean())

        # Angular step (scan_res * ceil(increment)) in radians
        angle_step_rad = np.deg2rad(
            abs(self.scan_resolution) * abs(np.ceil(self.increment))
        )
        tan_step = np.tan(angle_step_rad)

        # Expected distance per search point
        self.expected_distance = self.sp_r * tan_step * exp_distance_multiplier

    # ---------------------------
    # Observed distances (hot path)
    # ---------------------------
    def get_observed_distances(self):
        N = self.sp_r.shape[0]
        base_xyz = self.search_point_epoch.cloud
        neigh_xyz = self.nc_xyz

        observed_distance = np.full(N, self.default_max_distance, dtype=np.float64)

        sp_order = self._sp_order
        sp_bin_to_slice = self._sp_bin_to_slice
        bin_neighbors_idx = self._bin_neighbors_idx

        for bin_id, neigh_idx in bin_neighbors_idx.items():
            sp_slice = sp_bin_to_slice[bin_id]
            base_idx = sp_order[sp_slice]
            base_pts = base_xyz[base_idx]

            if base_pts.size == 0 or neigh_idx.size == 0:
                continue

            neigh_pts = neigh_xyz[neigh_idx]

            mean_per_base = per_point_mean_distance(base_pts, neigh_pts)
            observed_distance[base_idx] = mean_per_base

        self.observed_distance = observed_distance



    # ---------------------------
    # ScOR values
    # ---------------------------

    def get_ScOR_values(self):
        obs = self.observed_distance
        # Avoid division by zero
        obs = np.where(obs == 0.0, 1e-6, obs)
        self.scor_values = np.clip(self.expected_distance / obs, 0.0, 1.0)

    # ---------------------------
    # Public API
    # ---------------------------

    def run(self):
        self.get_phi_theta_range_phiIndex_thetaIndex()
        self.build_neighborhoods()
        self.get_expected_distances()
        self.get_observed_distances()
        self.get_ScOR_values()
        return self.scor_values, self.expected_distance, self.observed_distance
