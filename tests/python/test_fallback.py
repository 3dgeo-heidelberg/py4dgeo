from py4dgeo.fallback import *
from py4dgeo.m3c2 import M3C2

from . import epoch1, epoch2

import pytest
import _py4dgeo


@pytest.mark.parametrize(
    "uncertainty_callback",
    [
        (_py4dgeo.standard_deviation_uncertainty, standard_deviation_uncertainty),
        (_py4dgeo.no_uncertainty, no_uncertainty),
    ],
)
@pytest.mark.parametrize(
    "workingset_callback",
    [
        (_py4dgeo.radius_workingset_finder, radius_workingset_finder),
        (_py4dgeo.cylinder_workingset_finder, cylinder_workingset_finder),
    ],
)
def test_fallback_implementations(
    epoch1, epoch2, uncertainty_callback, workingset_callback
):
    class CxxTestM3C2(M3C2):
        def callback_uncertainty_calculation(self):
            return uncertainty_callback[0]

        def callback_workingset_finder(self):
            return workingset_callback[0]

    class PythonTestM3C2(M3C2):
        def callback_uncertainty_calculation(self):
            return uncertainty_callback[1]

        def callback_workingset_finder(self):
            return workingset_callback[1]

    # Instantiate a fallback M3C2 instance
    pym3c2 = CxxTestM3C2(
        epochs=(epoch1, epoch2),
        corepoints=epoch1.cloud,
        radii=(3.0,),
        scales=(2.0,),
        max_cylinder_length=6.0,
    )

    # And a regular C++ based one
    m3c2 = PythonTestM3C2(
        epochs=(epoch1, epoch2),
        corepoints=epoch1.cloud,
        radii=(3.0,),
        scales=(2.0,),
        max_cylinder_length=6.0,
    )

    # The results should match
    distances, uncertainties = m3c2.run()
    fb_distances, fb_uncertainties = pym3c2.run()

    assert np.allclose(distances, fb_distances)
    assert np.allclose(uncertainties, fb_uncertainties)


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

    assert np.allclose(distances, fb_distances)
    assert np.allclose(uncertainties, fb_uncertainties)
