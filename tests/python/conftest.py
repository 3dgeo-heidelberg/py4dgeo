from py4dgeo.epoch import read_from_xyz
from py4dgeo.m3c2 import M3C2
from py4dgeo.segmentation import SpatiotemporalSegmentation
from py4dgeo.util import MemoryPolicy, set_memory_policy

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


@pytest.fixture
def segmentation(epochs):
    ref_epoch, epoch1 = epochs

    ref_epoch.timestamp = "March 9th 2022, 16:32"
    epoch1.timestamp = "March 9th 2022, 16:33"

    # TODO M3C2 should be refactored to not necessarily take epochs
    m3c2 = M3C2(
        epochs=(ref_epoch, epoch1),
        corepoints=ref_epoch.cloud,
        cyl_radii=[2.0],
        normal_radii=[2.0],
    )

    seg = SpatiotemporalSegmentation(m3c2=m3c2, reference_epoch=ref_epoch)

    seg.add_epoch(epoch1)

    return seg


@pytest.fixture(autouse=True)
def memory_policy_fixture():
    """This fixture ensures that all tests start with the default memory policy"""
    set_memory_policy(MemoryPolicy.COREPOINTS)
