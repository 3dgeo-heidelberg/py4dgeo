from py4dgeo.util import get_memory_policy

import numpy as np
import os
import pickle
import pytest

from . import epochs

_test_files = [
    os.path.join(os.path.split(__file__)[0], "../data/plane_horizontal_t1.xyz")
]


def test_kdtree(epochs):
    epoch1, _ = epochs
    epoch1.build_kdtree()
    data = epoch1.cloud

    # Find all points in sufficiently large radius
    result = epoch1.kdtree.radius_search(np.array([0, 0, 0]), 100)
    assert result.shape[0] == data.shape[0]


def test_kdtree_pickle(epochs):
    epoch1, _ = epochs
    with pytest.raises(RuntimeError):
        fn = os.path.join("kdtree.pickle")
        with open(fn, "wb") as f:
            pickle.dump(epoch1.kdtree, f)


def test_rebuilding(epochs):
    epoch1, _ = epochs

    # Not build yet - leaf parameter is 0
    assert epoch1.kdtree.leaf_parameter() == 0

    # Building with default - leaf parameter is 10
    epoch1.build_kdtree()
    assert epoch1.kdtree.leaf_parameter() == 10

    # Non-forced rebuild is ignored - leaf parameter stays 10
    epoch1.build_kdtree(leaf_size=20)
    assert epoch1.kdtree.leaf_parameter() == 10

    # forced rebuild - leaf parameter is 20
    epoch1.build_kdtree(leaf_size=20, force_rebuild=20)
    assert epoch1.kdtree.leaf_parameter() == 20
