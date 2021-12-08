from py4dgeo.m3c2 import *
from py4dgeo.util import Py4DGeoError, set_memory_policy, MemoryPolicy

from . import epochs

import pytest


def test_m3c2(epochs):
    epoch1, epoch2 = epochs

    # Try with wrong number of epochs
    with pytest.raises(Py4DGeoError):
        M3C2(epochs=(epoch1,), corepoints=epoch1.cloud, radii=(1.0,))

    # Instantiate an M3C2 instance
    m3c2 = M3C2(
        epochs=(epoch1, epoch2), corepoints=epoch1.cloud, radii=(3.0,), scales=(2.0,)
    )

    # Run it
    distances, uncertainties = m3c2.run()

    # Running with the same epoch twice should yield all zeroes
    distances, uncertainties = M3C2(
        epochs=(epoch1, epoch1), corepoints=epoch1.cloud, radii=(3.0,), scales=(2.0,)
    ).run()
    assert (distances == 0).all()


def test_minimal_m3c2(epochs):
    epoch1, epoch2 = epochs
    set_memory_policy(MemoryPolicy.MINIMAL)

    # Instantiate an M3C2 instance
    m3c2 = M3C2(
        epochs=(epoch1, epoch2), corepoints=epoch1.cloud, radii=(3.0,), scales=(2.0,)
    )

    # Run it
    distances, uncertainties = m3c2.run()


def test_external_normals(epochs):
    epoch1, epoch2 = epochs
    # Instantiate an M3C2 instance
    d, u = M3C2(
        epochs=(epoch1, epoch2),
        corepoints=epoch1.cloud,
        radii=(3.0,),
        scales=(2.0,),
        corepoint_normals=np.array([[0, 0, 1]]),
    ).run()

    with pytest.raises(Py4DGeoError):
        d, u = M3C2(
            epochs=(epoch1, epoch2),
            corepoints=epoch1.cloud,
            radii=(3.0,),
            scales=(2.0,),
            corepoint_normals=np.array([[0, 0, 1], [0, 0, 1]]),
        ).run()
