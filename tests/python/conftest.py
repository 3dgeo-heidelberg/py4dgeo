from py4dgeo.epoch import read_from_xyz, read_from_las
from py4dgeo.logger import set_py4dgeo_logfile
from py4dgeo.m3c2 import M3C2
from py4dgeo.segmentation import SpatiotemporalAnalysis
from py4dgeo.util import MemoryPolicy, set_memory_policy

import numpy as np
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


def epoch_las_fixture(*filenames):
    """Wrap a given data file in an Epoch and make it a pytest fixture"""

    @pytest.fixture
    def _epoch_fixture():
        normal_columns = (
            ["NormalX", "NormalY", "NormalZ"] if ("normals" in filenames[0]) else None
        )
        return read_from_las(
            *tuple(find_data_file(fn) for fn in filenames),
            normal_columns=normal_columns,
        )

    return _epoch_fixture


def epoch_pbm3c2_fixture(*filenames, additional_dimensions):
    """Wrap a given data file in an Epoch and make it a pytest fixture"""

    @pytest.fixture
    def _epoch_pbm3c2_fixture():
        return read_from_xyz(
            *tuple(find_data_file(fn) for fn in filenames),
            additional_dimensions=additional_dimensions,
            delimiter=",",
        )

    return _epoch_pbm3c2_fixture


# Instantiate one fixture per data dile
epochs = epoch_fixture("plane_horizontal_t1.xyz", "plane_horizontal_t2.xyz")
epochs_las = epoch_las_fixture("plane_horizontal_t1.laz", "plane_horizontal_t2.laz")
epochs_las_w_normals = epoch_las_fixture(
    "plane_horizontal_t1_w_normals.laz", "plane_horizontal_t2_w_normals.laz"
)
epochs_segmented = epoch_pbm3c2_fixture(
    "plane_horizontal_t1_segmented.xyz",
    "plane_horizontal_t2_segmented.xyz",
    additional_dimensions={3: "segment_id"},
)


@pytest.fixture()
def pbm3c2_correspondences_file():
    return find_data_file("testdata-labelling2.csv")


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


@pytest.fixture()
def scanpos_info():
    filename = find_data_file("sps.json")
    with open(filename, "r") as load_f:
        try:
            json_str = load_f.read()
            json_dict = eval(json_str)
        except ValueError as err:
            return None
    return json_dict


def epoch_m3c2ep_fixture(*filenames, additional_dimensions):
    """Wrap a given data file in an Epoch and make it a pytest fixture"""

    @pytest.fixture
    def _epoch_m3c2ep_fixture():
        return read_from_las(
            *tuple(find_data_file(fn) for fn in filenames),
            additional_dimensions=additional_dimensions,
        )

    return _epoch_m3c2ep_fixture


epochs_m3c2ep = epoch_m3c2ep_fixture(
    "ahk_2017_652900_5189100_gnd_subarea.laz",
    "ahk_2018A_652900_5189100_gnd_subarea.laz",
    additional_dimensions={"point_source_id": "scanpos_id"},
)


@pytest.fixture()
def Cxx():
    covariance_matrix = np.loadtxt(
        find_data_file("Cxx.csv"), dtype=np.float64, delimiter=","
    )
    return covariance_matrix


@pytest.fixture()
def tfM():
    tf_matrix = np.loadtxt(find_data_file("tfM.csv"), dtype=np.float64, delimiter=",")
    return tf_matrix


@pytest.fixture()
def redPoint():
    reduction_point = np.loadtxt(
        find_data_file("redPoint.csv"), dtype=np.float64, delimiter=","
    )
    return reduction_point
