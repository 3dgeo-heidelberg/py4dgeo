import numpy as np
import pytest

import py4dgeo

cKDTree = pytest.importorskip("scipy.spatial").cKDTree


def _scipy_sor_reference(cloud, k, std_dev_multiplier):
    n_neighbors = min(k, cloud.shape[0] - 1)

    if n_neighbors > 0:
        distances, _ = cKDTree(cloud).query(cloud, k=n_neighbors + 1)
        mean_distances = distances[:, 1:].mean(axis=1)
    else:
        mean_distances = np.zeros(cloud.shape[0], dtype=float)

    mean = mean_distances.mean()
    std = mean_distances.std()
    threshold = mean + std_dev_multiplier * std
    flags = (mean_distances > threshold).astype(int)

    return flags, mean_distances, threshold


def test_sor_matches_scipy_reference():
    # Compare SOR against the scipy implementation to validate neighbor distances and thresholding.
    cloud = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    epoch = py4dgeo.Epoch(cloud)

    result = py4dgeo.statistical_outlier_removal(epoch, k=2, std_dev_multiplier=1.0)
    expected_flags, _, _ = _scipy_sor_reference(cloud, k=2, std_dev_multiplier=1.0)
    filtered_epoch, flags = result

    assert len(result) == 2
    assert filtered_epoch is epoch
    assert np.array_equal(flags, expected_flags)


def test_sor_remove_points_preserves_aligned_point_data():
    # Ensure point removal keeps cloud, normals, extra dimensions, and returned arrays shape align with the reduced inlier cloud.
    cloud = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    normals = np.arange(12, dtype=float).reshape(4, 3)
    extra = np.zeros(4, dtype=[("point_id", "i4")])
    extra["point_id"] = np.arange(4)
    epoch = py4dgeo.Epoch(
        cloud,
        normals=normals,
        additional_dimensions=extra,
        timestamp="2020-01-01",
    )

    filtered_epoch, flags = py4dgeo.statistical_outlier_removal(
        epoch, k=1, std_dev_multiplier=1.0, remove_points=True
    )
    expected_flags, _, _ = _scipy_sor_reference(cloud, k=1, std_dev_multiplier=1.0)
    mask = expected_flags == 0

    assert np.allclose(filtered_epoch.cloud, cloud[mask])
    assert np.allclose(filtered_epoch.normals, normals[mask])
    assert np.array_equal(filtered_epoch.additional_dimensions, extra[mask])
    assert str(filtered_epoch.timestamp) == "2020-01-01 00:00:00"
    assert np.array_equal(flags, expected_flags[mask])


def test_sor_rejects_invalid_k():
    # k must be positive because SOR is defined by distances to neighbors.
    epoch = py4dgeo.Epoch(np.array([[0.0, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="k >= 1"):
        py4dgeo.statistical_outlier_removal(epoch, k=0)
