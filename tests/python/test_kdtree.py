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
