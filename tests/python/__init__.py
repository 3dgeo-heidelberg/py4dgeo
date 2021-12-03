from py4dgeo.epoch import read_from_xyz

import os
import pytest


# The path to our data directory
data_dir = os.path.join(os.path.split(__file__)[0], "..", "data")


def find_data_file(filename):
    return os.path.join(data_dir, filename)


def epoch_fixture(filename):
    """Wrap a given data file in an Epoch and make it a pytest fixture"""

    @pytest.fixture
    def _epoch_fixture():
        return read_from_xyz(find_data_file(filename))

    return _epoch_fixture


# Instantiate one fixture per data dile
epoch1 = epoch_fixture("plane_horizontal_t1.xyz")
epoch2 = epoch_fixture("plane_horizontal_t2.xyz")
