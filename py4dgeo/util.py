import numpy as np
import os
import xdg

import _py4dgeo


class Py4DGeoError(Exception):
    pass


def find_file(filename):
    """Find a file of given name on the file system.

    This function is intended to use in tests and demo applications
    to locate data files without resorting to absolute paths. You may
    use it for your code as well.

    It looks in the following locations:

    * If an absolute filename is given, it is used
    * Check whether the given relative path exists with respect to the current working directory
    * Check whether the given relative path exists with respect to the specified XDG data directory (e.g. through the environment variable XDG_DATA_DIRS).

    :param: filename
        The (relative) filename to search for
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
    for xdg_dir in xdg.xdg_data_dirs():
        candidates.append(os.path.join(xdg_dir, filename))

    # Iterate through the list to check for file existence
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Cannot locate file {filename}. Tried the following locations: {', '.join(candidates)}"
    )


class MemoryPolicy(_py4dgeo.MemoryPolicy):
    """A desciptor for py4dgeo's memory usage policy

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

    For details about the memory policy, see :ref:`~py4dgeo.MemoryPolicy`.
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
    a copy if it is not.

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
