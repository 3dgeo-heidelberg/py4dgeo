from py4dgeo.epoch import *
from py4dgeo.util import Py4DGeoError

import numpy as np
import os
import pytest
import tempfile


def test_epoch_pickle(epochs):
    epoch1, _ = epochs
    epoch1.build_kdtree()

    # Operate in a temporary directory
    with tempfile.TemporaryDirectory() as dir:
        # Pickle the given KDTree
        fn = os.path.join(dir, "saved-epoch")
        save_epoch(epoch1, fn)

        # Unpickle it
        loaded = load_epoch(fn)

        # Assert that the two object behave the same
        assert loaded.cloud.shape[0] == epoch1.cloud.shape[0]
        assert np.allclose(loaded.geographic_offset, epoch1.geographic_offset)
        assert np.allclose(
            loaded.kdtree.radius_search(np.array([0, 0, 0]), 10),
            epoch1.kdtree.radius_search(np.array([0, 0, 0]), 10),
        )


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

    epoch = read_from_xyz(filename, header_lines=2)
    assert epoch.cloud.shape[0] == 2
