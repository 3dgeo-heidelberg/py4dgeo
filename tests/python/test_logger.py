from py4dgeo.logger import *
from py4dgeo.util import Py4DGeoError

import logging
import os
import pytest


def test_set_loggerfile(tmp_path):
    filename = os.path.join(tmp_path, "test.log")
    set_py4dgeo_logfile(filename)
    assert not os.path.exists(filename)
    logging.getLogger("py4dgeo").info("Some log message")
    assert os.stat(filename).st_size > 0


def test_log_exception(tmp_path):
    filename = os.path.join(tmp_path, "test.log")
    set_py4dgeo_logfile(filename)
    assert not os.path.exists(filename)

    with pytest.raises(Py4DGeoError):
        raise Py4DGeoError("This is some exception")

    assert os.stat(filename).st_size > 0
