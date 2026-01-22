import argparse
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

from importlib import metadata

import _py4dgeo


# The current data archive URL
TEST_DATA_ARCHIVE = "https://zenodo.org/records/16751963/files/"

# Read the version from package metadata
__version__ = metadata.version(__package__)


class Py4DGeoError(Exception):
    def __init__(self, msg, loggername="py4dgeo"):
        # Initialize the base class
        super().__init__(msg)

        # Also write the message to the error stream
        logger = logging.getLogger(loggername)
        logger.error(self)


def get_test_data_dir():
    return pooch.os_cache("py4dgeo")


def download_test_data(path=None):
    """Download the test data"""
    if path is None:
        path = get_test_data_dir()

    p = pooch.create(
        path=path,
        base_url=TEST_DATA_ARCHIVE,
        registry={
            "usage_data.zip": "md5:2340574f10d50a72f95766fcab5f8259",
            "synthetic.zip": "md5:534ffec84060f5462757b19d06b2b2e1",
            "pbm3c2.zip": "md5:dee1a9446e1cd1e30f59864c58fd793c",
        },
    )

    for archive in p.registry:
        p.fetch(
            archive,
            downloader=pooch.HTTPDownloader(timeout=(3, None)),
            processor=pooch.Unzip(extract_dir=os.path.join(path, "extracted")),
        )

    return path


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

    data_dir = get_test_data_dir()
    candidates.append(os.path.join(data_dir, filename))
    for root, _, files in os.walk(data_dir):
        if filename in files:
            candidates.append(os.path.join(root, filename))

    # Use the XDG data directories
    if platform.system() in ["Linux", "Darwin"]:
        for xdg_dir in xdg.xdg_data_dirs():
            candidates.append(os.path.join(xdg_dir, filename))

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


def copy_test_data_entrypoint():
    """A command line entry point to download test data"""

    parser = argparse.ArgumentParser(description="Download py4dgeo test data")

    parser.add_argument(
        "path",
        nargs="?",
        default=os.getcwd(),
        help="Target directory for test data (default: current working directory)",
    )

    args = parser.parse_args()

    download_test_data(path=args.path)
