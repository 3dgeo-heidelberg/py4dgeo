from py4dgeo.m3c2 import *
from py4dgeo.util import Py4DGeoError, set_memory_policy, MemoryPolicy

import pytest


@pytest.mark.parametrize("robust_aggr", (True, False))
def test_m3c2(epochs, robust_aggr):
    epoch1, epoch2 = epochs

    # Try with wrong number of epochs
    with pytest.raises(Py4DGeoError):
        M3C2(epochs=(epoch1,), corepoints=epoch1.cloud, cyl_radii=(1.0,))

    # Instantiate an M3C2 instance
    m3c2 = M3C2(
        epochs=(epoch1, epoch2),
        corepoints=epoch1.cloud,
        cyl_radii=(3.0,),
        normal_radii=(2.0,),
        robust_aggr=robust_aggr,
    )

    # Run it
    distances, uncertainties = m3c2.run()

    # Running with the same epoch twice should yield all zeroes
    distances, uncertainties = M3C2(
        epochs=(epoch1, epoch1),
        corepoints=epoch1.cloud,
        cyl_radii=(3.0,),
        normal_radii=(2.0,),
        robust_aggr=robust_aggr,
    ).run()
    assert np.allclose(distances, 0)


def test_minimal_m3c2(epochs):
    epoch1, epoch2 = epochs
    set_memory_policy(MemoryPolicy.MINIMAL)

    # Instantiate an M3C2 instance
    m3c2 = M3C2(
        epochs=(epoch1, epoch2),
        corepoints=epoch1.cloud,
        cyl_radii=(3.0,),
        normal_radii=(2.0,),
    )

    # Run it
    distances, uncertainties = m3c2.run()


def test_registration_error(epochs):
    epoch1, _ = epochs

    m3c2 = M3C2(
        epochs=(epoch1, epoch1),
        corepoints=epoch1.cloud,
        cyl_radii=(3.0,),
        normal_radii=(2.0,),
        registration_error=1.0,
    )

    # Run it and check that lodetection is at least 1.96
    _, uncertainties = m3c2.run()
    assert (uncertainties["lodetection"] > 1.96).all()


def test_external_normals(epochs):
    epoch1, epoch2 = epochs
    # Instantiate an M3C2 instance
    d, u = M3C2(
        epochs=(epoch1, epoch2),
        corepoints=epoch1.cloud,
        cyl_radii=(3.0,),
        normal_radii=(2.0,),
        corepoint_normals=np.array([[0, 0, 1]]),
    ).run()

    with pytest.raises(Py4DGeoError):
        d, u = M3C2(
            epochs=(epoch1, epoch2),
            corepoints=epoch1.cloud,
            cyl_radii=(3.0,),
            normal_radii=(2.0,),
            corepoint_normals=np.array([[0, 0, 1], [0, 0, 1]]),
        ).run()


def test_directions_radii(epochs):
    epoch1, epoch2 = epochs
    # Instantiate an M3C2 instance
    m3c2 = M3C2(
        epochs=(epoch1, epoch2),
        corepoints=epoch1.cloud,
        cyl_radii=(3.0,),
        normal_radii=(1.0, 2.0, 3.0),
    )

    # Run it
    m3c2.directions()

    assert m3c2._directions_radii is not None
    for i in range(m3c2.directions_radii().shape[0]):
        assert m3c2.directions_radii()[i] in (1.0, 2.0, 3.0)
