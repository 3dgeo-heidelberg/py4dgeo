import os
import platform
import xdg


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
    * Check whether the given relative path exists with respect to the specified XDG data directory (e.g. through the environment variable XDG_DATA_DIR) - Linux/MacOS only.

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
    if platform.system() in ["Linux", "Darwin"]:
        for xdg_dir in xdg.xdg_data_dirs():
            candidates.append(os.path.join(xdg_dir, filename))

    # Iterate through the list to check for file existence
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Cannot locate file {filename}. Tried the following locations: {', '.join(candidates)}"
    )
