from py4dgeo.util import *

import os
import pytest
import tempfile


def test_find_file(monkeypatch, tmp_path):
    # An absolute path is preserved
    abspath = os.path.abspath(__file__)
    assert abspath == find_file(abspath)

    # Check that a file in the current working directory is picked up correctly
    with tempfile.NamedTemporaryFile(dir=os.getcwd()) as tmp_file:
        assert os.path.join(os.getcwd(), tmp_file.name) == find_file(tmp_file.name)

    if platform.system() in ["Linux", "Darwin"]:
        monkeypatch.setenv("XDG_DATA_DIRS", str(tmp_path))
        abspath = os.path.join(tmp_path, "somefile.txt")
        open(abspath, "w").close()
        assert abspath == find_file("somefile.txt")

    with pytest.raises(FileNotFoundError):
        find_file("not.existent")
