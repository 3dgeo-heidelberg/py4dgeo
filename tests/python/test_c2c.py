import numpy as np
import pytest
import os
import tempfile

import py4dgeo

from py4dgeo.util import Py4DGeoError


def test_c2c_calculate_distances_on_arrays():
    source = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    target = np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)

    distances = py4dgeo.C2C(max_distance=10.0).calculate_distances(source, target)

    assert np.allclose(distances, np.array([1.0, 1.0]))


def test_c2c_default_filter_matches_none_filter():
    source = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    target = np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)

    d_default = py4dgeo.C2C(max_distance=10.0).calculate_distances(source, target)
    d_none = py4dgeo.C2C(
        max_distance=10.0, correspondence_filter="none"
    ).calculate_distances(source, target)

    assert np.allclose(d_default, d_none)


def test_c2c_calculate_distances_applies_max_distance():
    source = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    target = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    distances = py4dgeo.C2C(max_distance=1.5).calculate_distances(
        source, target
    )

    assert np.isclose(distances[0], 1.0)
    assert np.isnan(distances[1])


def test_c2c_mutual_nearest_neighbors_sets_non_mutual_to_nan():
    source = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)
    target = np.array([[0.05, 0.0, 0.0]], dtype=np.float64)

    distances = py4dgeo.C2C(
        max_distance=10.0, correspondence_filter="mutual_nearest_neighbors"
    ).calculate_distances(source, target)

    assert np.isclose(distances[0], 0.05)
    assert np.isnan(distances[1])


def test_c2c_mutual_nearest_neighbors_keeps_max_distance_behavior():
    source = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    target = np.array([[0.05, 0.0, 0.0]], dtype=np.float64)

    distances = py4dgeo.C2C(
        max_distance=0.2, correspondence_filter="mutual_nearest_neighbors"
    ).calculate_distances(source, target)

    assert np.isclose(distances[0], 0.05)
    assert np.isnan(distances[1])


def test_c2c_calculate_distances_on_epochs(epochs):
    epoch1, epoch2 = epochs
    distances = py4dgeo.C2C(max_distance=np.inf).calculate_distances(epoch1, epoch2)

    assert distances.shape[0] == epoch1.cloud.shape[0]
    assert np.isfinite(distances).all()


def test_c2c_class_run_with_corepoints(epochs):
    epoch1, epoch2 = epochs
    corepoints = epoch1.cloud[::50]

    c2c = py4dgeo.C2C(
        epochs=(epoch1, epoch2),
        corepoints=corepoints,
        max_distance=10.0,
    )

    distances = c2c.run()
    assert distances.shape[0] == corepoints.shape[0]
    assert np.isfinite(distances).all()


def test_c2c_mutual_nearest_neighbors_with_corepoints():
    epoch1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    epoch2 = np.array(
        [
            [0.05, 0.0, 0.0],
            [10.1, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    corepoints = epoch1[:2]

    c2c = py4dgeo.C2C(
        epochs=(epoch1, epoch2),
        corepoints=corepoints,
        max_distance=10.0,
        correspondence_filter="mutual_nearest_neighbors",
    )
    distances = c2c.run()

    assert distances.shape[0] == corepoints.shape[0]
    assert np.isclose(distances[0], 0.05)
    assert np.isnan(distances[1])


def test_c2c_invalid_correspondence_filter_raises():
    with pytest.raises(Py4DGeoError):
        py4dgeo.C2C(correspondence_filter="invalid")


def test_c2c_class_checks_epochs(epochs):
    epoch1, _ = epochs

    with pytest.raises(Py4DGeoError):
        py4dgeo.C2C(epochs=(epoch1,), max_distance=10.0)


def test_c2c_class_requires_epochs_for_run():
    with pytest.raises(Py4DGeoError):
        py4dgeo.C2C(max_distance=10.0).run()


def test_c2c_write_las_with_corepoints():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    target = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    corepoints = source[:2]

    c2c = py4dgeo.C2C(
        epochs=(source, target),
        corepoints=corepoints,
        max_distance=10.0,
    )
    distances = c2c.run()

    with tempfile.TemporaryDirectory() as directory:
        outfile = os.path.join(directory, "c2c_corepoints.las")
        py4dgeo.write_c2c_results_to_las(
            outfile,
            c2c,
            attribute_dict={"distances": distances},
        )

        import laspy

        in_file = laspy.read(outfile)
        coords = np.vstack((in_file.x, in_file.y, in_file.z)).transpose()
        stored_distances = getattr(in_file, "distances")

    assert np.allclose(coords, corepoints)
    assert np.allclose(stored_distances, distances, equal_nan=True)
