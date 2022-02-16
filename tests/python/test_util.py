from py4dgeo.util import *

import os
import platform
import pytest
import tempfile


def test_find_file(monkeypatch, tmp_path):
    # An absolute path is preserved
    abspath = os.path.abspath(__file__)
    assert abspath == find_file(abspath)

    # Check that a file in the current working directory is picked up correctly
    with tempfile.NamedTemporaryFile(dir=os.getcwd()) as tmp_file:
        assert os.path.join(os.getcwd(), tmp_file.name) == find_file(tmp_file.name)

    # Test with XDG data directory
    if platform.system() in ["Linux", "Darwin"]:
        monkeypatch.setenv("XDG_DATA_DIRS", str(tmp_path))
        abspath = os.path.join(tmp_path, "somefile.txt")
        open(abspath, "w").close()
        assert abspath == find_file("somefile.txt")

    with pytest.raises(FileNotFoundError):
        find_file("not.existent")


def test_memory_policy():
    set_memory_policy(MemoryPolicy.RELAXED)

    assert get_memory_policy() is MemoryPolicy.RELAXED
    assert memory_policy_is_minimum(MemoryPolicy.STRICT)


def test_make_contiguous():
    arr1 = np.full((42, 3), 1.0, order="C")
    arr1_c = make_contiguous(arr1)
    assert arr1 is arr1_c

    arr1_slice = arr1[::5]
    arr1_slice_c = make_contiguous(arr1_slice)
    assert arr1_slice.shape == arr1_slice_c.shape

    arr1_fort = np.full((42, 3), 1.0, order="F")
    arr1_fort_c = make_contiguous(arr1_fort)
    assert arr1_fort.shape == arr1_fort_c.shape


def test_make_contiguous_strict():
    set_memory_policy(MemoryPolicy.STRICT)

    arr1 = np.full((42, 3), 1.0, order="C")
    arr1_slice = arr1[::5]

    with pytest.raises(Py4DGeoError):
        make_contiguous(arr1_slice)


def test_as_double_precision():
    arr1 = np.full((42, 3), 1.0, dtype=np.float64)
    arr1_dp = as_double_precision(arr1)
    assert arr1 is arr1_dp

    arr1 = np.full((42, 3), 1.0, dtype=np.float32)
    arr1_dp = as_double_precision(arr1)
    assert np.allclose(arr1, arr1_dp)


def test_as_single_precision():
    arr1 = np.full((42, 3), 1.0, dtype=np.float32)
    arr1_sp = as_single_precision(arr1)
    assert arr1 is arr1_sp

    arr1 = np.full((42, 3), 1.0, dtype=np.float64)
    arr1_sp = as_single_precision(arr1)
    assert np.allclose(arr1, arr1_sp)


def test_as_double_precision_strict():
    set_memory_policy(MemoryPolicy.STRICT)

    arr = np.full((42, 3), 1.0, dtype=np.float32)

    with pytest.raises(Py4DGeoError):
        as_double_precision(arr)


def test_set_num_threads():
    try:
        set_num_threads(42)
        assert get_num_threads() == 42
    except Py4DGeoError as e:
        assert str(e) == "py4dgeo was built without threading support!"
        assert get_num_threads() == 1
