from py4dgeo.epoch import *
from py4dgeo.util import Py4DGeoError

import datetime
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


@pytest.mark.parametrize("timestamp", [datetime.datetime.utcnow(), "25. November 1986"])
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

    with pytest.raises(Py4DGeoError):
        epoch = read_from_xyz(filename)

    epoch = read_from_xyz(filename, skip_header=2)
    assert epoch.cloud.shape[0] == 2


def test_normalize_timestamp():
    assert normalize_timestamp(None) is None

    now = datetime.datetime.utcnow()
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
    epoch.transform(trafo)

    # Check the result
    assert np.allclose(epoch.cloud[:, 0] - 1, copycloud[:, 0])
    assert np.allclose(epoch.cloud[:, 1], copycloud[:, 1])
    assert np.allclose(epoch.cloud[:, 2], copycloud[:, 2])

    # Apply the inverse transformation and check for identity
    trafo[0, 3] = -1
    epoch.transform(trafo)
    assert np.allclose(epoch.cloud, copycloud)
