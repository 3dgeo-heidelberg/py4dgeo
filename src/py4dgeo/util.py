import collections
import hashlib
import logging
import numpy as np
import os
import platform
import requests
import sys
import tarfile
import tempfile
import xdg

import py4dgeo._py4dgeo as _py4dgeo


# The current data archive URL
TEST_DATA_ARCHIVE = "https://github.com/3dgeo-heidelberg/py4dgeo-test-data/releases/download/2022-06-22/data.tar.gz"
TEST_DATA_CHECKSUM = "7b459de25c210da3c4815b19b09647dfc73cff0c692954cd7ec29b166ec2c42a"

# The directory where to place
_test_data_dir = tempfile.TemporaryDirectory()


class Py4DGeoError(Exception):
    def __init__(self, msg, loggername="py4dgeo"):
        # Initialize the base class
        super().__init__(msg)

        # Also write the message to the error stream
        logger = logging.getLogger(loggername)
        logger.error(self)


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

    # Use the temporary directory with test data (only contains something)
    # iff ensure_test_data_availability was called.
    candidates.append(os.path.join(_test_data_dir.name, filename))

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


def copy_test_data(target):
    """Download test data and copy it into the working directory"""

    # Create temporary directory for the download
    with tempfile.TemporaryDirectory() as tmp:
        archive = requests.get(TEST_DATA_ARCHIVE).content
        checksum = hashlib.sha256(archive).hexdigest()
        if checksum != TEST_DATA_CHECKSUM:
            raise ValueError("Checksum for test data archive failed.")

        archive_file = os.path.join(tmp, "data.tar.gz")
        with open(archive_file, "wb") as tar:
            tar.write(archive)

        with tarfile.open(archive_file, "r:gz") as tar:
            tar.extractall(path=target)


def copy_test_data_entrypoint():
    # Define the target directory
    target = os.getcwd()
    if len(sys.argv) > 1:
        target = sys.argv[1]

    copy_test_data(target)


def ensure_test_data_availability():
    """Ensure the availability of test data for notebooks etc."""
    copy_test_data(_test_data_dir.name)
