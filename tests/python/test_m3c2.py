from py4dgeo.m3c2 import *
from py4dgeo.util import Py4DGeoError

from . import epoch1, epoch2

import pytest


def test_m3c2(epoch1, epoch2):
    # Try with wrong number of epochs
    with pytest.raises(Py4DGeoError):
        M3C2(epochs=(epoch1,), corepoints=epoch1.cloud, radii=(1.0,))

    # Instantiate an M3C2 instance
    m3c2 = M3C2(
        epochs=(epoch1, epoch2), corepoints=epoch1.cloud, radii=(1.0,), scales=(1.0,)
    )
