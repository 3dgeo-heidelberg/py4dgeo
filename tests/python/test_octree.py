from py4dgeo.util import get_memory_policy

import numpy as np
import os
import pickle
import pytest
import tempfile


def test_octree(epochs):
    epoch1, _ = epochs
    epoch1.build_octree()
    data = epoch1.cloud

    # Find all points in sufficiently large radius
    result = epoch1.octree.radius_search(np.array([0, 0, 0]), 100)
    assert result.shape[0] == data.shape[0]


def test_octree_pickle(epochs):
    epoch1, _ = epochs
    with pytest.raises(RuntimeError):
        with tempfile.TemporaryDirectory() as dir:
            fn = os.path.join(dir, "octree.pickle")
            with open(fn, "wb") as f:
                pickle.dump(epoch1.octree, f)


def test_rebuilding(epochs):
    epoch1, _ = epochs

    # Not build yet - number of points is 0
    assert epoch1.octree.get_number_of_points() == 0

    # Building with default - number of points is > 0
    epoch1.build_octree()
    assert epoch1.octree.get_number_of_points() > 0
