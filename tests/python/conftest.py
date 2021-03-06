from py4dgeo.epoch import read_from_xyz
from py4dgeo.logger import set_py4dgeo_logfile
from py4dgeo.m3c2 import M3C2
from py4dgeo.segmentation import SpatiotemporalAnalysis
from py4dgeo.util import MemoryPolicy, set_memory_policy

import os
import pytest
import shutil
import subprocess
import tempfile


# The path to our data directory
data_dir = os.path.join(os.path.split(__file__)[0], "..", "data")
log_dir = tempfile.TemporaryDirectory()

# Ensure that the data directory has been downloaded
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    subprocess.call(["copy_py4dgeo_test_data", data_dir])


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
def analysis(tmp_path):
    shutil.copy(os.path.join(data_dir, "synthetic.zip"), tmp_path)
    return SpatiotemporalAnalysis(os.path.join(tmp_path, "synthetic.zip"))


@pytest.fixture(autouse=True)
def log_into_temporary_directory():
    set_py4dgeo_logfile(os.path.join((log_dir.name), "py4dgeo.log"))


@pytest.fixture(autouse=True)
def memory_policy_fixture():
    """This fixture ensures that all tests start with the default memory policy"""
    set_memory_policy(MemoryPolicy.COREPOINTS)
