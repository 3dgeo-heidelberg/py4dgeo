from py4dgeo.util import get_memory_policy

import numpy as np
import os
import pickle
import pytest

from . import epoch1

_test_files = [
    os.path.join(os.path.split(__file__)[0], "../data/plane_horizontal_t1.xyz")
]


def test_kdtree(epoch1):
    data = epoch1.cloud

    # Find all points in sufficiently large radius
    result = epoch1.kdtree.radius_search(np.array([0, 0, 0]), 100)
    assert result.shape[0] == data.shape[0]

    # Trigger precomputations
    epoch1.kdtree.precompute(data[:20, :], 20, get_memory_policy())

    # Compare precomputed and real results
    for i in range(20):
        result1 = epoch1.kdtree.radius_search(data[i, :], 5)
        result2 = epoch1.kdtree.precomputed_radius_search(i, 5)
        assert result1.shape == result2.shape


def test_kdtree_pickle(epoch1):
    with pytest.raises(RuntimeError):
        fn = os.path.join("kdtree.pickle")
        with open(fn, "wb") as f:
            pickle.dump(epoch1.kdtree, f)
