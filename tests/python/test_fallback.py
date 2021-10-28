from py4dgeo.fallback import *
from py4dgeo.m3c2 import M3C2

from . import epoch1, epoch2


def test_python_fallback_m3c2(epoch1, epoch2):
    # Instantiate a fallback M3C2 instance
    pym3c2 = PythonFallbackM3C2(
        epochs=(epoch1, epoch2), corepoints=epoch1.cloud, radii=(3.0,), scales=(2.0,)
    )

    # And a regular C++ based one
    m3c2 = M3C2(
        epochs=(epoch1, epoch2), corepoints=epoch1.cloud, radii=(3.0,), scales=(2.0,)
    )

    # The results should match
    distances, uncertainties = m3c2.run()
    fb_distances, fb_uncertainties = pym3c2.run()

    assert (distances == fb_distances).all()
    assert (uncertainties == fb_uncertainties).all()
