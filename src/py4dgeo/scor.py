from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from py4dgeo.epoch import Epoch
from py4dgeo.util import Py4DGeoError

try:
    import numba
except ImportError:
    numba = None


def _spherical_coordinates(
    points: np.ndarray, scan_position: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return range, elevation, and azimuth relative to the scanner."""
    delta = points - np.asarray(scan_position, dtype=np.float64)
    horizontal_range = np.hypot(delta[:, 0], delta[:, 1])
    ranges = np.hypot(horizontal_range, delta[:, 2])
    elevation = np.arctan2(delta[:, 2], horizontal_range)
    azimuth = np.arctan2(delta[:, 1], delta[:, 0])
    return ranges, elevation, azimuth


def _angular_bins(
    azimuth: np.ndarray, elevation: np.ndarray, scan_resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    """Round angles to integer scanner bins."""
    if scan_resolution <= 0:
        raise Py4DGeoError("scan_resolution must be > 0 degrees.")
    azimuth_bins = np.rint(np.rad2deg(azimuth) / scan_resolution).astype(np.int64)
    elevation_bins = np.rint(np.rad2deg(elevation) / scan_resolution).astype(
        np.int64
    )
    return azimuth_bins, elevation_bins


def _neighbor_offsets(increment: float) -> list[tuple[int, int]]:
    """Return the angular bin offsets."""
    increment = float(increment)
    if increment == 0.5:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if increment == 1.0:
        return [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    rounded = round(increment)
    if increment <= 0 or abs(increment - rounded) >= 1e-9:
        raise Py4DGeoError(
            "Increment value must be 0.5 or any integer value greater than 0."
        )

    radius = int(rounded)
    return [
        (azimuth, elevation)
        for azimuth in range(-radius, radius + 1)
        for elevation in range(-radius, radius + 1)
        if azimuth != 0 or elevation != 0
    ]


def _sorted_angular_bins(
    points: np.ndarray,
    phi_indices: np.ndarray,
    theta_indices: np.ndarray,
    phi_min: int,
    theta_min: int,
    theta_span: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group points by angular bin and return the sorted representation."""
    bin_ids = (phi_indices - phi_min) * theta_span + theta_indices - theta_min
    order = np.argsort(bin_ids)
    sorted_bin_ids = bin_ids[order]

    # The bin IDs are already sorted, so group boundaries can be found by
    # comparing adjacent IDs. This avoids sorting the same IDs again inside
    # np.unique.
    boundaries = np.empty(sorted_bin_ids.size, dtype=bool)
    if sorted_bin_ids.size:
        boundaries[0] = True
        boundaries[1:] = sorted_bin_ids[1:] != sorted_bin_ids[:-1]
    starts = np.flatnonzero(boundaries)
    unique_bins = sorted_bin_ids[starts]
    counts = np.diff(np.append(starts, sorted_bin_ids.size))

    return (
        order.astype(np.int64, copy=False),
        unique_bins.astype(np.int64, copy=False),
        starts.astype(np.int64, copy=False),
        counts.astype(np.int64, copy=False),
        np.ascontiguousarray(points[order], dtype=np.float64),
    )


def _neighbor_matrices(
    search_bin_ids: np.ndarray,
    candidate_bin_ids: np.ndarray,
    candidate_bin_starts: np.ndarray,
    candidate_bin_counts: np.ndarray,
    offsets: list[tuple[int, int]],
    theta_span: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the candidate bins at every requested angular offset."""
    offset_ids = np.array(
        [phi * theta_span + theta for phi, theta in offsets], dtype=np.int64
    )
    targets = search_bin_ids[:, None] + offset_ids
    positions = np.searchsorted(candidate_bin_ids, targets)

    # A flattened ID alone would allow vertical offsets to wrap from the top
    # of one phi column to the bottom of the next. Check the target theta
    # coordinate in the original 2D grid before accepting an ID match.
    theta_offsets = np.array(
        [theta for _, theta in offsets], dtype=np.int64
    )
    target_theta = search_bin_ids[:, None] % theta_span + theta_offsets
    valid = (
        (target_theta >= 0)
        & (target_theta < theta_span)
        & (positions < candidate_bin_ids.size)
    )
    matches = np.zeros_like(valid)
    matches[valid] = candidate_bin_ids[positions[valid]] == targets[valid]

    starts = np.full(targets.shape, -1, dtype=np.int64)
    counts = np.zeros(targets.shape, dtype=np.int64)
    starts[matches] = candidate_bin_starts[positions[matches]]
    counts[matches] = candidate_bin_counts[positions[matches]]
    return starts, counts


def _observed_distances_numpy(
    search_points: np.ndarray,
    search_order: np.ndarray,
    search_bin_starts: np.ndarray,
    search_bin_counts: np.ndarray,
    candidate_points: np.ndarray,
    neighbor_starts: np.ndarray,
    neighbor_counts: np.ndarray,
) -> np.ndarray:
    """Calculate each point's mean distance to its angular neighbors."""
    observed = np.full(len(search_points), 99999.0, dtype=np.float64)

    for bin_index, search_start in enumerate(search_bin_starts):
        search_count = search_bin_counts[bin_index]
        search_indices = search_order[search_start : search_start + search_count]
        points = search_points[search_start : search_start + search_count]
        distance_sum = np.zeros(search_count, dtype=np.float64)
        neighbor_total = 0

        for candidate_start, candidate_count in zip(
            neighbor_starts[bin_index], neighbor_counts[bin_index]
        ):
            if candidate_count == 0:
                continue
            candidates = candidate_points[
                candidate_start : candidate_start + candidate_count
            ]
            differences = candidates[:, None, :] - points[None, :, :]
            squared_distances = np.einsum(
                "mki,mki->mk", differences, differences, dtype=np.float64
            )
            distance_sum += np.sqrt(squared_distances, dtype=np.float64).sum(
                axis=0, dtype=np.float64
            )
            neighbor_total += candidate_count

        if neighbor_total:
            observed[search_indices] = distance_sum / neighbor_total

    return observed


if numba is not None:

    @numba.njit(parallel=True, cache=True)
    def _observed_distances_numba(
        search_points: np.ndarray,
        search_order: np.ndarray,
        search_bin_starts: np.ndarray,
        search_bin_counts: np.ndarray,
        candidate_points: np.ndarray,
        neighbor_starts: np.ndarray,
        neighbor_counts: np.ndarray,
    ) -> np.ndarray:
        """Compiled observed-distance calculation."""
        observed = np.full(search_points.shape[0], 99999.0, dtype=np.float64)

        # Angular bins are independent and can therefore be processed safely
        # in parallel. Each iteration writes to different search-point indices.
        for bin_index in numba.prange(search_bin_starts.shape[0]):
            search_start = search_bin_starts[bin_index]
            search_count = search_bin_counts[bin_index]

            for local_index in range(search_count):
                sorted_search_index = search_start + local_index
                original_search_index = search_order[sorted_search_index]
                search_x = search_points[sorted_search_index, 0]
                search_y = search_points[sorted_search_index, 1]
                search_z = search_points[sorted_search_index, 2]
                distance_sum = 0.0
                neighbor_total = 0

                for offset_index in range(neighbor_starts.shape[1]):
                    candidate_count = neighbor_counts[
                        bin_index, offset_index
                    ]
                    if candidate_count == 0:
                        continue

                    candidate_start = neighbor_starts[
                        bin_index, offset_index
                    ]
                    for candidate_index in range(
                        candidate_start, candidate_start + candidate_count
                    ):
                        dx = (
                            candidate_points[candidate_index, 0] - search_x
                        )
                        dy = (
                            candidate_points[candidate_index, 1] - search_y
                        )
                        dz = (
                            candidate_points[candidate_index, 2] - search_z
                        )
                        distance_sum += np.sqrt(
                            dx * dx + dy * dy + dz * dz
                        )
                    neighbor_total += candidate_count

                if neighbor_total > 0:
                    observed[original_search_index] = (
                        distance_sum / neighbor_total
                    )

        return observed

else:
    _observed_distances_numba = None


def _observed_distances(
    search_points: np.ndarray,
    search_order: np.ndarray,
    search_bin_starts: np.ndarray,
    search_bin_counts: np.ndarray,
    candidate_points: np.ndarray,
    neighbor_starts: np.ndarray,
    neighbor_counts: np.ndarray,
) -> np.ndarray:
    """Calculate observed distances with Numba when it is available."""
    if _observed_distances_numba is not None:
        return _observed_distances_numba(
            search_points,
            search_order,
            search_bin_starts,
            search_bin_counts,
            candidate_points,
            neighbor_starts,
            neighbor_counts,
        )

    return _observed_distances_numpy(
        search_points,
        search_order,
        search_bin_starts,
        search_bin_counts,
        candidate_points,
        neighbor_starts,
        neighbor_counts,
    )


def scan_outlier_ratio(
    search_point_epoch: Epoch,
    neighborhood_candidate_epochs: Epoch | Sequence[Epoch] | None = None,
    scan_position: Sequence[float] = (0.0, 0.0, 0.0),
    scan_resolution: float = 0.017,
    increment: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Scan Outlier Ratio (ScOR) for a point cloud.

    ScOR is a scanning- and survey-aware measure of local surface coherence.
    For every search point, the function constructs a neighborhood in the
    scanner's discrete angular grid. It compares the mean Euclidean distance
    to the points in these neighboring grid cells with the distance expected
    for a locally planar surface perpendicular to the laser beam.

    Parameters
    ----------
    search_point_epoch : py4dgeo.Epoch
        Epoch containing the points for which ScOR values are calculated.
    neighborhood_candidate_epochs : py4dgeo.Epoch or sequence of py4dgeo.Epoch, optional
        Epoch or epochs supplying the angular-neighborhood candidates. Multiple
        epochs can be used to construct a multi-temporal neighborhood, provided
        that they were acquired from the same scan position. If omitted,
        ``search_point_epoch`` is used for both search points and candidates.
    scan_position : sequence of float, default=(0.0, 0.0, 0.0)
        Cartesian scanner position ``(x, y, z)`` in the same coordinate system
        and unit as the point clouds.
    scan_resolution : float, default=0.017
        Angular scan resolution in degrees. The same resolution is assumed in
        the horizontal and vertical directions.
    increment : float, default=0.5
        Size of the angular neighborhood. ``0.5`` selects the four immediately
        adjacent grid cells. A positive integer selects all surrounding cells
        within that many angular bins.

    Returns
    -------
    scor_values : numpy.ndarray
        ScOR value for every search point in the interval ``[0, 1]``. Values
        near one indicate coherent surfaces; values near zero indicate
        detached or isolated points.
    expected_distances : numpy.ndarray
        Mean neighbor distances expected from the scan geometry.
    observed_distances : numpy.ndarray
        Mean Euclidean distances measured between each search point and its
        angular-neighborhood candidates.

    Raises
    ------
    py4dgeo.util.Py4DGeoError
        If ``scan_resolution`` is not positive or ``increment`` is neither
        ``0.5`` nor a positive integer.

    Notes
    -----
    Candidate epochs used for multi-temporal neighborhoods must share the scan
    position and scan pattern of the search epoch. ScOR does not replace
    radiometric filters and assumes a regular angular scan grid and a locally
    planar reference surface perpendicular to the incident laser beam.
    """
    # Use the search epoch itself for a single-epoch ScOR analysis. Supplying
    # other epochs instead creates the multi-temporal neighborhood described
    # in Section 3.3 of the paper.
    search_points = np.asarray(search_point_epoch.cloud, dtype=np.float64)
    if neighborhood_candidate_epochs is None:
        candidate_epochs = (search_point_epoch,)
    elif isinstance(neighborhood_candidate_epochs, Epoch):
        candidate_epochs = (neighborhood_candidate_epochs,)
    else:
        candidate_epochs = tuple(neighborhood_candidate_epochs)

    # In the standard single-epoch case, search points and neighborhood
    # candidates are identical. Scanner coordinates, angular bins, and 
    # grouped representation can be reused below.
    candidates_are_search_points = (
        len(candidate_epochs) == 1 and candidate_epochs[0] is search_point_epoch
    )
    if candidates_are_search_points:
        candidate_points = search_points
    else:
        candidate_points = np.concatenate(
            [epoch.cloud for epoch in candidate_epochs]
        )

    # Express every point in the scanner-centered spherical coordinate system.
    # The range is required for the expected object-space point spacing, while
    # azimuth (phi) and elevation (theta) locate a point in the scan pattern.
    search_ranges, search_theta, search_phi = _spherical_coordinates(
        search_points, scan_position
    )
    if candidates_are_search_points:
        candidate_theta = search_theta
        candidate_phi = search_phi
    else:
        _, candidate_theta, candidate_phi = _spherical_coordinates(
            candidate_points, scan_position
        )

    # Divide both scan angles by the angular resolution and round them to the
    # nearest integer, reproducing the discrete scan-grid indices from
    # Section 3.1 of the paper.
    search_phi, search_theta = _angular_bins(
        search_phi, search_theta, scan_resolution
    )
    if candidates_are_search_points:
        candidate_phi = search_phi
        candidate_theta = search_theta
    else:
        candidate_phi, candidate_theta = _angular_bins(
            candidate_phi, candidate_theta, scan_resolution
        )

    # Treat the angular bins as a 2D grid, with phi as the horizontal
    # axis and theta as the vertical axis. Define a shared grid origin
    # and height so each (phi, theta) cell can be converted to one integer ID.
    phi_min = int(min(search_phi.min(), candidate_phi.min()))
    theta_min = int(min(search_theta.min(), candidate_theta.min()))
    theta_max = int(max(search_theta.max(), candidate_theta.max()))
    theta_span = theta_max - theta_min + 1

    # Group points occupying the same angular cell. Sorting once allows all
    # candidate points in a neighboring cell to be retrieved as a contiguous
    # slice rather than by repeatedly scanning the complete point cloud.
    search_order, search_bins, search_starts, search_counts, sorted_search = (
        _sorted_angular_bins(
            search_points,
            search_phi,
            search_theta,
            phi_min,
            theta_min,
            theta_span,
        )
    )
    if candidates_are_search_points:
        candidate_bins = search_bins
        candidate_starts = search_starts
        candidate_counts = search_counts
        sorted_candidates = sorted_search
    else:
        (
            _,
            candidate_bins,
            candidate_starts,
            candidate_counts,
            sorted_candidates,
        ) = _sorted_angular_bins(
            candidate_points,
            candidate_phi,
            candidate_theta,
            phi_min,
            theta_min,
            theta_span,
        )

    # Construct each point's neighborhood in the angular scan domain. The
    # default consists of the immediate cells above, below, left, and right;
    # larger increments expand the neighborhood around the central cell.
    offsets = _neighbor_offsets(increment)
    neighbor_starts, neighbor_counts = _neighbor_matrices(
        search_bins,
        candidate_bins,
        candidate_starts,
        candidate_counts,
        offsets,
        theta_span,
    )

    # Calculate the mean spacing expected on an ideal, locally planar surface
    # perpendicular to the beam. This makes the metric inherently
    # range-aware: expected point spacing grows with scanner distance.
    offset_distances = np.sqrt(
        np.array([phi**2 + theta**2 for phi, theta in offsets], dtype=np.float64)
    )
    offset_angles = np.deg2rad(float(scan_resolution)) * offset_distances
    expected_distances = search_ranges * np.tan(offset_angles).mean()

    # Retrieve the 3D coordinates in the selected angular cells and average
    # their observed Euclidean distances to each search point.
    observed_distances = _observed_distances(
        sorted_search,
        search_order,
        search_starts,
        search_counts,
        sorted_candidates,
        neighbor_starts,
        neighbor_counts,
    )

    # Form the ratio of mean expected and observed spacings.
    # Values above one are capped because they do not represent
    # the detached points ScOR is designed to identify.
    denominator = np.where(
        observed_distances == 0.0, 1e-6, observed_distances
    )
    scor_values = np.clip(expected_distances / denominator, 0.0, 1.0)
    return scor_values, expected_distances, observed_distances


# Provide a short alternative with exactly the same behavior.
scor = scan_outlier_ratio
