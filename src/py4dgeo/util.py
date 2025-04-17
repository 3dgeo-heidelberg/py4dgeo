import collections
import logging
import numpy as np
import os
import platform
import psutil
import pooch
import requests
import sys
import warnings
import xdg
import json
from datetime import datetime

from importlib import metadata

import _py4dgeo


# The current data archive URL
TEST_DATA_ARCHIVE = "https://github.com/3dgeo-heidelberg/py4dgeo-test-data/releases/download/2024-06-28/data.tar.gz"
TEST_DATA_CHECKSUM = "5ee51a43b008181b829113d8b967cdf519eae4ac37a3301f1eaf53d15d3016cc"

# Read the version from package metadata
__version__ = metadata.version(__package__)


class Py4DGeoError(Exception):
    def __init__(self, msg, loggername="py4dgeo"):
        # Initialize the base class
        super().__init__(msg)

        # Also write the message to the error stream
        logger = logging.getLogger(loggername)
        logger.error(self)


def download_test_data(path=pooch.os_cache("py4dgeo"), fatal=False):
    """Download the test data and copy it into the given path"""
    try:
        return pooch.retrieve(
            TEST_DATA_ARCHIVE,
            TEST_DATA_CHECKSUM,
            path=path,
            downloader=pooch.HTTPDownloader(timeout=(3, None)),
            processor=pooch.Untar(extract_dir="."),
        )
    except requests.RequestException as e:
        if fatal:
            raise e
        else:
            return []


def find_file(filename, fatal=True):
    """Find a file of given name on the file system.

    This function is intended to use in tests and demo applications
    to locate data files without resorting to absolute paths. You may
    use it for your code as well.

    It looks in the following locations:

    * If an absolute filename is given, it is used
    * Check whether the given relative path exists with respect to the current working directory
    * Check whether the given relative path exists with respect to the specified XDG data directory (e.g. through the environment variable :code:`XDG_DATA_DIRS`).
    * Check whether the given relative path exists in downloaded test data.

    :param filename:
        The (relative) filename to search for
    :type filename: str
    :param fatal:
        Whether not finding the file should be a fatal error
    :type fatal: bool
    :return: An absolute filename
    """

    # If the path is absolute, do not change it
    if os.path.isabs(filename):
        return filename

    # Gather a list of candidate paths for relative path
    candidates = []

    # Use the current working directory
    candidates.append(os.path.join(os.getcwd(), filename))

    # Use the XDG data directories
    if platform.system() in ["Linux", "Darwin"]:
        for xdg_dir in xdg.xdg_data_dirs():
            candidates.append(os.path.join(xdg_dir, filename))

    # Ensure that the test data is taken into account. This is properly
    # cached across sessions and uses a connection timeout.
    for datafile in download_test_data():
        if os.path.basename(datafile) == filename:
            candidates.append(datafile)

    # Iterate through the list to check for file existence
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    if fatal:
        raise FileNotFoundError(
            f"Cannot locate file {filename}. Tried the following locations: {', '.join(candidates)}"
        )

    return filename


class MemoryPolicy(_py4dgeo.MemoryPolicy):
    """A descriptor for py4dgeo's memory usage policy

    This can be used to describe the memory usage policy that py4dgeo
    should follow. The implementation of py4dgeo checks the currently
    set policy whenever it would make a memory allocation of the same order
    of magnitude as the input pointcloud or the set of corepoints.
    To globally set the policy, use :func:`~py4dgeo.set_memory_policy`.

    Currently the following policies are available:

    * :code:`STRICT`: py4dgeo is not allowed to do additional memory allocations.
      If such an allocation would be required, an error is thrown.
    * :code:`MINIMAL`: py4dgeo is allowed to do additional memory allocations if
      and only if they are necessary for a seemless operation of the library.
    * :code:`COREPOINTS`: py4dgeo is allowed to do additional memory allocations
      as part of performance trade-off considerations (e.g. precompute vs. recompute),
      but only if the allocation is on the order of the number of corepoints.
      This is the default behaviour of py4dgeo.
    * :code:`RELAXED`: py4dgeo is allowed to do additional memory allocations as
      part of performance trade-off considerations (e.g. precompute vs. recompute).
    """

    pass


# The global storage for the memory policy
_policy = MemoryPolicy.COREPOINTS


def set_memory_policy(policy: MemoryPolicy):
    """Globally set py4dgeo's memory policy

    For details about the memory policy, see :class:`~py4dgeo.MemoryPolicy`.
    Use this once before performing any operations. Changing the memory policy
    in the middle of the computation results in undefined behaviour.

    :param policy: The policy value to globally set
    :type policy: MemoryPolicy
    """
    global _policy
    _policy = policy


def get_memory_policy():
    """Access the globally set memory policy"""
    return _policy


def memory_policy_is_minimum(policy: MemoryPolicy):
    """Whether or not the globally set memory policy is at least the given one

    :param policy: The policy value to compare against
    :type policy: MemoryPolicy
    :returns: Whether the globally set policy is at least the given one
    :rtype: bool
    """
    return policy <= get_memory_policy()


def make_contiguous(arr: np.ndarray):
    """Make a numpy array contiguous

    This is a no-op if the array is already contiguous and makes
    a copy if it is not. It checks py4dgeo's memory policy before copying.

    :param arr: The numpy array
    :type arr: np.ndarray
    """

    if arr.flags["C_CONTIGUOUS"]:
        return arr

    if not memory_policy_is_minimum(MemoryPolicy.MINIMAL):
        raise Py4DGeoError(
            "Using non-contiguous memory layouts requires at least the MINIMAL memory policy"
        )

    return np.copy(arr, order="C")


def _as_dtype(arr, dtype, policy_check):
    if np.issubdtype(arr.dtype, dtype):
        return arr

    if policy_check and not memory_policy_is_minimum(MemoryPolicy.MINIMAL):
        raise Py4DGeoError(
            f"py4dgeo expected an input of type {np.dtype(dtype).name}, but got {np.dtype(arr.dtype).name}. Current memory policy forbids automatic cast."
        )

    return np.asarray(arr, dtype=dtype)


def as_double_precision(arr: np.ndarray, policy_check=True):
    """Ensure that a numpy array is double precision

    This is a no-op if the array is already double precision and makes a copy
    if it is not. It checks py4dgeo's memory policy before copying.

    :param arr: The numpy array
    :type arr: np.ndarray
    """
    return _as_dtype(arr, np.float64, policy_check)


def set_num_threads(num_threads: int):
    """Set the number of threads to use in py4dgeo

    :param num_threads: The number of threads to use
    "type num_threads: int
    """

    env_threads = os.environ.get("OMP_NUM_THREADS")
    if env_threads:
        try:
            env_threads_int = int(env_threads)
            if env_threads_int != num_threads:
                warnings.warn(
                    f"OMP_NUM_THREADS environment variable is set to {env_threads_int}, but set_num_threads({num_threads}) was called. The environment variable may override this setting."
                )
        except ValueError:
            raise Py4DGeoError(f"Invalid value for OMP_NUM_THREADS: '{env_threads}'")
    try:
        _py4dgeo.omp_set_num_threads(num_threads)
    except AttributeError:
        # The C++ library was built without OpenMP!
        if num_threads != 1:
            raise Py4DGeoError("py4dgeo was built without threading support!")


def get_num_threads():
    """Get the number of threads currently used by py4dgeo

    :return: The number of threads
    :rtype: int
    """

    try:
        return _py4dgeo.omp_get_max_threads()
    except AttributeError:
        # The C++ library was built without OpenMP!
        return 1


def append_file_extension(filename, extension):
    """Append a file extension if and only if the original filename has none"""

    _, ext = os.path.splitext(filename)
    if ext == "":
        return f"{filename}.{extension}"
    else:
        return filename


def is_iterable(obj):
    """Whether the object is an iterable (excluding a string)"""
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str)


def initialize_openmp_defaults():
    """Set OpenMP environment variables for optimal performance on Windows with llvm OpenMP"""

    # Only apply when using Windows
    if platform.system() != "Windows":
        return

    # Only set if the user has not already
    if "OMP_NUM_THREADS" not in os.environ:
        num_cores = psutil.cpu_count(logical=False)
        os.environ["OMP_NUM_THREADS"] = str(num_cores)

    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "threads")


def copy_test_data_entrypoint():
    # Define the target directory
    target = os.getcwd()
    if len(sys.argv) > 1:
        target = sys.argv[1]

    download_test_data(path=target, fatal=True)


def xyz_2_spherical(xyz):
    dxy = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
    r = np.sqrt(dxy**2 + xyz[:, 2]**2)          # radius r
    theta = np.arctan2(dxy, xyz[:, 2])          # theta θ   # for elevation angle defined from Z-axis down
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])      # phi ϕ
    return r, theta, phi                        # [m, rad, rad]


def get_delta_t(t1_file, 
    t2_file, 
    temporal_format='%y%m%d_%H%M%S'):
    """
    Calculate the time difference in seconds between two files based on their filenames.

    Args:
        t1_file (str): Path to the first file.
        t2_file (str): Path to the second file.
        temporal_format (str): The datetime format used in the filenames.

    Returns:
        float: Time difference in seconds.
    """
    # Extract the base filenames
    t1_filename = os.path.basename(t1_file)
    t2_filename = os.path.basename(t2_file)
    
    # Extract the timestamp part from the filenames
    try:
        t1_time_str = t1_filename.split(" ")[-1][:-4]  # Adjust slicing if needed
        t2_time_str = t2_filename.split(" ")[-1][:-4]
        
        # Parse the timestamps into datetime objects
        t1_time = datetime.strptime(t1_time_str, temporal_format)
        t2_time = datetime.strptime(t2_time_str, temporal_format)
    except (IndexError, ValueError) as e:
        print(f"Error parsing filenames: {e}")
        return None
    
    # Calculate the difference in seconds
    delta_seconds = (t2_time - t1_time).total_seconds()
    
    #print(f"Time difference between the two epochs: {delta_seconds} seconds")
    return delta_seconds


def create_project_structure(config) -> None:
    """
    Generate output folder structure if not existing.
    """
    sub_directories = [
        "01_Change_analysis_UHD_VAPC", 
        "02_Change_analysis_UHD_Change_Events",
        "03_Change_visualisation_UHD_Projected_Images",
        "04_Change_visualisation_UHD_Change_Events",
        "documentation"
    ]
    outdir = config["output_folder"]

    #Create Project Folder
    output_dir = os.path.join(outdir,
                              config["project_name"])
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    #Create Poject Subfolders
    out_folders = {}
    for sub_dir in sub_directories:
        sub_dir_path = os.path.join(output_dir,sub_dir)
        out_folders[sub_dir] = sub_dir_path
        if not os.path.isdir(sub_dir_path):
            os.mkdir(sub_dir_path)
    return config["project_name"], out_folders, config["temporal_format"]


def setup_configuration(config_file,t1_file,t2_file, timestamp):
    """
    Sets up the configuration for the change detection pipeline.

    Parameters:
    ----------
        config_file (str): Path to the JSON configuration file.
        t1_file (str): Path to the first timepoint file.
        t2_file (str): Path to the second timepoint file.
        timestamp (str): Timestamp to append to the project name if included.

    Returns:
    ----------
        tuple: A tuple containing:
            - configuration (dict): Loaded configuration settings.
            - t1_out_file (str): Path for the output t1 file.
            - t2_out_file (str): Path for the output t2 file.
            - m3c2_out_file (str): Path for the M3C2 output file.
            - m3c2_clustered (str): Path for the clustered M3C2 output file.
            - change_event_folder (str): Path to the change event folder.
            - change_event_file (str): Path to the change events JSON file.
            - delta_t (float): Time delta between t1 and t2.
            - project_name (str): Name of the project.
            - projected_image_folder (str): Path to the projected images folder.
            - projected_events_folder (str): Path to the projected events folder.
            
    Raises:
    ----------
        AssertionError: If any of the input files do not exist.
    """
    #Check if input is proper
    assert os.path.isfile(t1_file), "This file does not exist: %s"%t1_file
    assert os.path.isfile(t2_file), "This file does not exist: %s"%t2_file
    assert os.path.isfile(config_file), "Configuration file does not exist at: %s"%config_file

    with open(config_file, 'r') as file:
        configuration = json.load(file)
    if not os.path.isdir(configuration["project_setting"]["output_folder"]): os.mkdir(configuration["project_setting"]["output_folder"])
    if configuration["project_setting"]["include_timestamp"]:
        configuration["project_setting"]["project_name"] = configuration["project_setting"]["project_name"] + "_" + timestamp
    project_name, out_folders, temporal_format = create_project_structure(configuration["project_setting"])
    with open(os.path.join(out_folders["documentation"],configuration["project_setting"]["project_name"]+".json"), 'w') as f:
        json.dump(configuration, f,indent=4)

    delta_t = get_delta_t(t1_file,t2_file,temporal_format)

    combination_of_names = os.path.basename(t1_file)[:-4] + "_" + os.path.basename(t2_file)[:-4]
    t1_out_file = os.path.join(out_folders["01_Change_analysis_UHD_VAPC"],combination_of_names+"_t1.laz")
    t2_out_file = os.path.join(out_folders["01_Change_analysis_UHD_VAPC"],combination_of_names+"_t2.laz")
    m3c2_out_file = os.path.join(out_folders["02_Change_analysis_UHD_Change_Events"],combination_of_names+".laz")
    change_event_folder = os.path.join(out_folders["02_Change_analysis_UHD_Change_Events"])
    change_event_file = os.path.join(change_event_folder, "change_events.json")
    m3c2_clustered = os.path.join(change_event_folder,combination_of_names,"clustered.laz")

    projected_image_folder = out_folders["03_Change_visualisation_UHD_Projected_Images"]
    projected_events_folder =out_folders["04_Change_visualisation_UHD_Change_Events"]

    # m3c2_out_file = os.path.join(out_folders["02_Change_analysis_UHD_M3C2"],combination_of_names+".laz")
    # change_event_folder = os.path.join(out_folders["03_Change_analysis_UHD_Change_Events"])
    # m3c2_clustered = os.path.join(change_event_folder,combination_of_names,"clustered.laz")

    return configuration, t1_out_file,t2_out_file,m3c2_out_file,m3c2_clustered,change_event_folder, change_event_file, delta_t, project_name, projected_image_folder, projected_events_folder