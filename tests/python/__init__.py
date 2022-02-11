from py4dgeo.epoch import read_from_xyz

import numpy as np
import os
import pytest


# The path to our data directory
data_dir = os.path.join(os.path.split(__file__)[0], "..", "data")


def find_data_file(filename):
    return os.path.join(data_dir, filename)


def epoch_fixture(*filenames):
    """Wrap a given data file in an Epoch and make it a pytest fixture"""

    @pytest.fixture
    def _epoch_fixture():
        return read_from_xyz(*tuple(find_data_file(fn) for fn in filenames))

    return _epoch_fixture


# Instantiate one fixture per data dile
epochs = epoch_fixture("plane_horizontal_t1.xyz", "plane_horizontal_t2.xyz")


def compare_algorithms(alg1, alg2):
    """A helper to compare the output of two algorithms that should be equivalent"""
    # Run the two algorithms
    distances1, uncertainties1 = alg1.run()
    distances2, uncertainties2 = alg2.run()

    assert np.allclose(distances1, distances2)
    assert np.allclose(uncertainties1["lodetection"], uncertainties2["lodetection"])
    assert np.allclose(uncertainties1["stddev1"], uncertainties2["stddev1"])
    assert np.allclose(uncertainties1["stddev2"], uncertainties2["stddev2"])
    assert np.allclose(uncertainties1["num_samples1"], uncertainties2["num_samples1"])
    assert np.allclose(uncertainties1["num_samples2"], uncertainties2["num_samples2"])
