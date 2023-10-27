from py4dgeo.util import get_memory_policy

import numpy as np
import os
import pickle
import pytest
import tempfile


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
        with tempfile.TemporaryDirectory() as dir:
            fn = os.path.join(dir, "kdtree.pickle")
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


def test_nearest_neighbors(epochs_las):
    epoch1, epoch2 = epochs_las
    epoch1.build_kdtree()

    checklist_pr = epoch1.kdtree.nearest_neighbors(epoch2.cloud)
    assert len(checklist_pr) > 0
    indices, distances = zip(*checklist_pr)

    for i in range(epoch1.cloud.shape[0]):
        assert i == indices[i]
        assert np.isclose(
            ((epoch1.cloud[i, :] - epoch2.cloud[i, :]) ** 2).sum(), distances[i]
        )
