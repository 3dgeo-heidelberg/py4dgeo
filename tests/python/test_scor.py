import numpy as np

import py4dgeo


def _point_from_angles(distance, azimuth, elevation):
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    horizontal = distance * np.cos(elevation)
    return [
        horizontal * np.cos(azimuth),
        horizontal * np.sin(azimuth),
        distance * np.sin(elevation),
    ]


def test_scor_returns_expected_values():
    # Compare ScOR with distances that can be calculated directly.
    search_point = np.array([_point_from_angles(10.0, 0.0, 0.0)])
    neighbor_points = np.array(
        [
            _point_from_angles(12.0, -1.0, 0.0),
            _point_from_angles(12.0, 1.0, 0.0),
            _point_from_angles(12.0, 0.0, -1.0),
            _point_from_angles(12.0, 0.0, 1.0),
        ]
    )

    values, expected, observed = py4dgeo.scan_outlier_ratio(
        py4dgeo.Epoch(search_point),
        py4dgeo.Epoch(neighbor_points),
        scan_resolution=1.0,
    )

    expected_distance = 10.0 * np.tan(np.deg2rad(1.0))
    observed_distance = np.linalg.norm(
        neighbor_points - search_point[0], axis=1
    ).mean()

    assert np.allclose(expected, [expected_distance])
    assert np.allclose(observed, [observed_distance])
    assert np.allclose(values, [expected_distance / observed_distance])


def test_scor_accepts_multiple_candidate_epochs():
    # Candidate points from several epochs form one shared neighborhood.
    search_epoch = py4dgeo.Epoch(
        np.array([_point_from_angles(10.0, 0.0, 0.0)])
    )
    first_epoch = py4dgeo.Epoch(
        np.array(
            [
                _point_from_angles(12.0, -1.0, 0.0),
                _point_from_angles(12.0, 1.0, 0.0),
            ]
        )
    )
    second_epoch = py4dgeo.Epoch(
        np.array(
            [
                _point_from_angles(12.0, 0.0, -1.0),
                _point_from_angles(12.0, 0.0, 1.0),
            ]
        )
    )

    multiple = py4dgeo.scan_outlier_ratio(
        search_epoch,
        [first_epoch, second_epoch],
        scan_resolution=1.0,
    )
    combined = py4dgeo.scan_outlier_ratio(
        search_epoch,
        py4dgeo.Epoch(np.vstack((first_epoch.cloud, second_epoch.cloud))),
        scan_resolution=1.0,
    )

    for multiple_array, combined_array in zip(multiple, combined):
        assert np.allclose(multiple_array, combined_array)


def test_scor_expected_distance_for_larger_increment():
    # Every cell in a larger window contributes its own angular distance.
    search_epoch = py4dgeo.Epoch(
        np.array([_point_from_angles(10.0, 0.0, 0.0)])
    )
    candidate_epoch = py4dgeo.Epoch(
        np.array([_point_from_angles(10.0, 1.0, 0.0)])
    )

    _, expected, _ = py4dgeo.scan_outlier_ratio(
        search_epoch,
        candidate_epoch,
        scan_resolution=1.0,
        increment=2,
    )

    offsets = [
        (phi, theta)
        for phi in range(-2, 3)
        for theta in range(-2, 3)
        if phi != 0 or theta != 0
    ]
    offset_distances = np.array(
        [np.hypot(phi, theta) for phi, theta in offsets]
    )
    expected_distance = 10.0 * np.tan(
        np.deg2rad(offset_distances)
    ).mean()

    assert np.allclose(expected, [expected_distance])


def test_scor_does_not_wrap_angular_grid_boundaries():
    # These points occupy diagonally adjacent bins, not vertical neighbors.
    search_epoch = py4dgeo.Epoch(
        np.array([_point_from_angles(10.0, 0.0, 1.0)])
    )
    candidate_epoch = py4dgeo.Epoch(
        np.array([_point_from_angles(10.0, 1.0, 0.0)])
    )

    values, _, observed = py4dgeo.scan_outlier_ratio(
        search_epoch,
        candidate_epoch,
        scan_resolution=1.0,
    )

    assert observed[0] == 99999.0
    assert values[0] < 0.01
