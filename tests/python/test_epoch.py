from py4dgeo.epoch import *
from py4dgeo.util import Py4DGeoError

import copy
import datetime
import gc
import numpy as np
import os
import pickle
import pytest
import tempfile


def test_epoch_pickle(epochs):
    epoch1, _ = epochs
    epoch1.build_kdtree()

    # Operate in a temporary directory
    with tempfile.TemporaryDirectory() as dir:
        # Pickle the given epoch
        filename = os.path.join(dir, "epoch.pickle")
        with open(filename, "wb") as f:
            pickle.dump(epoch1, f)

        # Unpickle it
        with open(filename, "rb") as f:
            loaded = pickle.load(f)

        # Assert that the two object behave the same
        assert loaded.cloud.shape[0] == epoch1.cloud.shape[0]
        assert np.allclose(
            loaded.kdtree.radius_search(np.array([0, 0, 0]), 10),
            epoch1.kdtree.radius_search(np.array([0, 0, 0]), 10),
        )


def test_epoch_saveload(epochs):
    epoch1, _ = epochs
    epoch1.build_kdtree()
    epoch1.build_octree()

    # Operate in a temporary directory
    with tempfile.TemporaryDirectory() as dir:
        # Save and load it
        filename = os.path.join(dir, "epoch")
        save_epoch(epoch1, filename)
        loaded = load_epoch(filename)

        # Assert that the two object behave the same
        assert loaded.cloud.shape[0] == epoch1.cloud.shape[0]
        assert np.allclose(
            loaded.kdtree.radius_search(np.array([0, 0, 0]), 10),
            epoch1.kdtree.radius_search(np.array([0, 0, 0]), 10),
        )
        assert np.allclose(
            loaded.octree.radius_search(np.array([0, 0, 0]), 10),
            epoch1.octree.radius_search(np.array([0, 0, 0]), 10),
        )


@pytest.mark.parametrize(
    "timestamp", [datetime.datetime.now(datetime.timezone.utc), "25. November 1986"]
)
def test_epoch_metadata_setters(epochs, timestamp):
    epoch, _ = epochs

    # Use all the property setters
    epoch.timestamp = timestamp

    # Test reconstruction of an Epoch from exported metadata
    epoch2 = Epoch(epoch.cloud, **epoch.metadata)
    assert epoch.timestamp - epoch2.timestamp == datetime.timedelta()


def test_as_epoch(epochs):
    epoch1, _ = epochs
    assert epoch1 is as_epoch(epoch1)
    assert np.allclose(epoch1.cloud, as_epoch(epoch1.cloud).cloud)


def test_read_from_xyz(epochs):
    epoch1, _ = epochs
    assert np.isclose(np.max(epoch1.cloud), 10)


def test_read_from_xyz_more_columns(tmp_path):
    filename = os.path.join(tmp_path, "more-than-three-columns.xyz")

    with open(filename, "w") as f:
        f.write("0 0 0 1 2 3")

    with pytest.raises(Py4DGeoError):
        epoch = read_from_xyz(filename)


def test_read_from_xyz_more_differing_colums(tmp_path):
    filename = os.path.join(tmp_path, "differing-columns.xyz")

    with open(filename, "w") as f:
        f.write("0 0 0\n")
        f.write("0 0")

    with pytest.raises(Py4DGeoError):
        epoch = read_from_xyz(filename)


def test_read_from_xyz_comma_delimited(tmp_path):
    filename = os.path.join(tmp_path, "comma-delimited.xyz")

    with open(filename, "w") as f:
        f.write("0,0,0\n")
        f.write("1,1,1")

    with pytest.raises(Py4DGeoError):
        epoch = read_from_xyz(filename)

    epoch = read_from_xyz(filename, delimiter=",")
    assert epoch.cloud.shape[0] == 2


def test_read_from_xyz_header(tmp_path):
    filename = os.path.join(tmp_path, "header.xyz")

    with open(filename, "w") as f:
        f.write("This is a header line\n")
        f.write("and another one\n")
        f.write("0 0 0\n")
        f.write("1 1 1\n")

    epoch = read_from_xyz(filename, skip_header=2)
    assert epoch.cloud.shape[0] == 2


def test_normalize_timestamp():
    assert normalize_timestamp(None) is None

    now = datetime.datetime.now(datetime.timezone.utc)
    assert normalize_timestamp(now) == now

    now = datetime.date.today()
    nnow = normalize_timestamp(now)
    assert nnow.year == now.year
    assert nnow.month == now.month
    assert nnow.day == now.day

    ts = normalize_timestamp((2010, 34))
    assert ts.year == 2010
    assert ts.month == 2
    assert ts.day == 3

    ts = normalize_timestamp("February 3rd, 2010, 9:27AM")
    assert ts.year == 2010
    assert ts.month == 2
    assert ts.day == 3
    assert ts.hour == 9
    assert ts.minute == 27

    with pytest.raises(Py4DGeoError):
        normalize_timestamp("foo")

    with pytest.raises(Py4DGeoError):
        normalize_timestamp(42)


def test_affine_trafo(epochs):
    epoch, _ = epochs
    copycloud = np.copy(epoch.cloud)

    # Apply a translation
    trafo = np.identity(4, dtype=np.float64)
    trafo[0, 3] = 1
    epoch.transform(affine_transformation=trafo)

    # Assert that the transformation was saved
    assert len(epoch.transformation) == 1
    assert np.allclose(epoch.transformation[0].affine_transformation, trafo)

    # Check the result
    assert np.allclose(epoch.cloud[:, 0] - 1, copycloud[:, 0])
    assert np.allclose(epoch.cloud[:, 1], copycloud[:, 1])
    assert np.allclose(epoch.cloud[:, 2], copycloud[:, 2])

    # Apply the inverse transformation and check for identity
    trafo[0, 3] = -1
    epoch.transform(affine_transformation=trafo)
    assert np.allclose(epoch.cloud, copycloud)
    assert len(epoch.transformation) == 2


def test_identity_3x4_trafo(epochs):
    epoch, _ = epochs
    copycloud = np.copy(epoch.cloud)

    # Define an identity 3x4 transformation
    trafo = np.zeros(shape=(3, 4))
    trafo[0, 0] = 1
    trafo[1, 1] = 1
    trafo[2, 2] = 1

    epoch.transform(affine_transformation=trafo)
    assert np.allclose(epoch.cloud, copycloud)
    assert len(epoch.transformation) == 1
    assert np.allclose(epoch.transformation[0].affine_transformation, np.identity(4))


def test_identity_rotate_translate(epochs):
    epoch, _ = epochs
    copycloud = np.copy(epoch.cloud)

    rotation = np.identity(3)
    translation = np.zeros(shape=(1, 3))

    epoch.transform(rotation=rotation, translation=translation)
    assert np.allclose(epoch.cloud, copycloud)
    assert len(epoch.transformation) == 1
    assert np.allclose(epoch.transformation[0].affine_transformation, np.identity(4))


def test_identity_reference_point_transformation(epochs):
    epoch, _ = epochs
    copycloud = np.copy(epoch.cloud)
    epoch.transform(
        affine_transformation=np.identity(4), reduction_point=np.array([127, 456, 578])
    )
    assert np.allclose(epoch.cloud, copycloud)
    assert len(epoch.transformation) == 1
    assert np.allclose(epoch.transformation[0].affine_transformation, np.identity(4))


def test_trafo_serialization(epochs):
    epoch, _ = epochs

    # Define and apply a transformation
    trafo = np.identity(4, dtype=np.float64)
    trafo[0, 3] = 1
    rp = np.array([1, 2, 3], dtype=np.float64)
    epoch.transform(affine_transformation=trafo, reduction_point=rp)

    # Operate in a temporary directory
    with tempfile.TemporaryDirectory() as dir:
        # Save and load it
        filename = os.path.join(dir, "epoch")
        save_epoch(epoch, filename)
        loaded = load_epoch(filename)

        # Assert that the two object behave the same
        assert len(loaded.transformation) == 1
        assert np.allclose(loaded.transformation[0].affine_transformation, trafo)
        assert np.allclose(loaded.transformation[0].reduction_point, rp)


def test_normal_computation(epochs):
    epoch, _ = epochs
    normals = epoch.calculate_normals(radius=1.5)
    assert normals.shape == epoch.cloud.shape


def test_epoch_saveload_w_normals(epochs_las_w_normals):
    epoch1_las, _ = epochs_las_w_normals

    # Operate in a temporary directory
    with tempfile.TemporaryDirectory() as dir:
        # Save and load it
        filename = os.path.join(dir, "epoch")
        save_epoch(epoch1_las, filename)
        loaded = load_epoch(filename)

        # Assert that the two object behave the same
        assert loaded.normals.shape[0] == epoch1_las.normals.shape[0]


def test_read_from_las_file_w_3_coords(epochs_las):
    epoch1, _ = epochs_las
    assert np.isclose(np.max(epoch1.cloud), 10)


def test_read_from_las_file_w_normals(epochs_las_w_normals):
    epoch1, _ = epochs_las_w_normals
    assert epoch1.cloud.shape[0] > 0
    assert epoch1.normals.shape[0] > 0
    assert epoch1.cloud.shape[1] == 3
    assert epoch1.normals.shape[1] == 3


def test_epoch_slicing(epochs_las_w_normals):
    epoch1, _ = epochs_las_w_normals

    # Take every second point
    epoch = epoch1[::2]

    assert epoch.cloud.shape[0] == (epoch1.cloud.shape[0] + 1) // 2
    assert epoch.normals.shape[0] == (epoch1.normals.shape[0] + 1) // 2
    assert (
        epoch.additional_dimensions.shape[0]
        == (epoch1.additional_dimensions.shape[0] + 1) // 2
    )
    assert len(epoch.metadata) == len(epoch1.metadata)
    for key in epoch.metadata:
        assert epoch.metadata[key] == epoch1.metadata[key]


def test_cpp_props_not_interchangeable(epochs):
    epoch, _ = epochs
    with pytest.raises(Py4DGeoError):
        epoch.cloud = 42
    with pytest.raises(Py4DGeoError):
        epoch.kdtree = None


def test_read_from_xyz_additional_dimensions(tmp_path):
    filename = os.path.join(tmp_path, "with_additional_dimensions.xyz")
    with open(filename, "w") as f:
        f.write("0 0 0 1 3\n")
        f.write("1 1 1 0 6\n")

    epoch = read_from_xyz(
        filename,
        additional_dimensions={3: "somebool", 4: "someint"},
        additional_dimensions_dtypes={"someint": np.int32},
    )

    assert epoch.additional_dimensions.shape == (2,)
    assert epoch.additional_dimensions["someint"].dtype == np.int32
    assert epoch.additional_dimensions["someint"][0] == 3
    assert epoch.additional_dimensions["someint"][1] == 6


def _deepcopy_epoch():
    """A small epoch with every optional per-point attribute set"""
    rng = np.random.default_rng(42)
    additional = np.empty((100, 1), dtype=[("intensity", "<f8")])
    additional["intensity"] = rng.random((100, 1))
    return Epoch(
        rng.random((100, 3)),
        normals=rng.random((100, 3)),
        additional_dimensions=additional,
        timestamp="2020-01-02 03:04:05",
    )


def test_deepcopy_cloud_is_independent():
    epoch = _deepcopy_epoch()
    copied = copy.deepcopy(epoch)
    expected = epoch.cloud.copy()

    assert np.array_equal(copied.cloud, expected)
    assert not np.shares_memory(copied.cloud, epoch.cloud)

    epoch.cloud[0, 0] += 1.0
    assert np.array_equal(copied.cloud, expected)


def test_deepcopy_preserves_timestamp():
    epoch = _deepcopy_epoch()
    assert copy.deepcopy(epoch).timestamp == epoch.timestamp


def test_deepcopy_kdtree_is_independent():
    epoch = _deepcopy_epoch()
    epoch.build_kdtree()
    query = np.array([0.5, 0.5, 0.5])
    expected = np.sort(epoch.kdtree.radius_search(query, 0.3))

    copied = copy.deepcopy(epoch)
    assert copied.kdtree is not epoch.kdtree
    assert copied.kdtree.leaf_parameter() == epoch.kdtree.leaf_parameter()
    assert np.array_equal(np.sort(copied.kdtree.radius_search(query, 0.3)), expected)

    # The copied tree must not depend on the original epoch's cloud or tree
    epoch.cloud[:] += 100.0
    epoch.build_kdtree(force_rebuild=True)
    assert np.array_equal(np.sort(copied.kdtree.radius_search(query, 0.3)), expected)

    del epoch
    gc.collect()
    assert np.array_equal(np.sort(copied.kdtree.radius_search(query, 0.3)), expected)


def test_deepcopy_octree_is_independent():
    epoch = _deepcopy_epoch()
    epoch.build_octree()
    copied = copy.deepcopy(epoch)

    assert copied.octree is not epoch.octree
    assert copied.octree.get_number_of_points() == epoch.octree.get_number_of_points()

    del epoch
    gc.collect()
    assert copied.octree.get_number_of_points() == 100


def test_deepcopy_unbuilt_kdtree_stays_separate():
    epoch = _deepcopy_epoch()
    copied = copy.deepcopy(epoch)

    assert copied.kdtree.leaf_parameter() == 0
    copied.build_kdtree()
    assert copied.kdtree.leaf_parameter() != 0
    assert epoch.kdtree.leaf_parameter() == 0


@pytest.mark.xfail(
    strict=True, raises=AttributeError, reason="__getstate__ does not store normals"
)
def test_deepcopy_normals_are_independent():
    epoch = _deepcopy_epoch()
    copied = copy.deepcopy(epoch)

    assert np.array_equal(copied.normals, epoch.normals)
    assert not np.shares_memory(copied.normals, epoch.normals)


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="__getstate__ does not store additional_dimensions",
)
def test_deepcopy_additional_dimensions_are_independent():
    epoch = _deepcopy_epoch()
    copied = copy.deepcopy(epoch)

    assert np.array_equal(copied.additional_dimensions, epoch.additional_dimensions)
    assert not np.shares_memory(
        copied.additional_dimensions, epoch.additional_dimensions
    )


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="__getstate__ does not store transformations",
)
def test_deepcopy_transformation_history_is_independent():
    epoch = _deepcopy_epoch()
    epoch.transform(translation=np.array([1.0, 0.0, 0.0]))
    copied = copy.deepcopy(epoch)

    assert len(copied.transformation) == 1
    copied.transform(translation=np.array([0.0, 1.0, 0.0]))
    assert len(epoch.transformation) == 1
