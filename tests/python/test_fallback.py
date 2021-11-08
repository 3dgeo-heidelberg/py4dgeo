from py4dgeo.fallback import *
from py4dgeo._py4dgeo import (
    cylinder_workingset_finder as cxx_cylinder_workingset_finder,
    no_uncertainty as cxx_no_uncertainty,
    radius_workingset_finder as cxx_radius_workingset_finder,
    standard_deviation_uncertainty as cxx_standard_deviation_uncertainty,
)
from py4dgeo.m3c2 import M3C2

from . import epoch1, epoch2

import pytest


@pytest.mark.parametrize(
    "uncertainty_callback",
    [
        (cxx_standard_deviation_uncertainty, standard_deviation_uncertainty),
        (cxx_no_uncertainty, no_uncertainty),
    ],
)
@pytest.mark.parametrize(
    "workingset_callback",
    [
        (cxx_radius_workingset_finder, radius_workingset_finder),
        (cxx_cylinder_workingset_finder, cylinder_workingset_finder),
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
    assert np.allclose(uncertainties["lodetection"], fb_uncertainties["lodetection"])
    assert np.allclose(uncertainties["stddev1"], fb_uncertainties["stddev1"])
    assert np.allclose(uncertainties["stddev2"], fb_uncertainties["stddev2"])
    assert np.allclose(uncertainties["num_samples1"], fb_uncertainties["num_samples1"])
    assert np.allclose(uncertainties["num_samples2"], fb_uncertainties["num_samples2"])


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
    assert np.allclose(uncertainties["lodetection"], fb_uncertainties["lodetection"])
    assert np.allclose(uncertainties["stddev1"], fb_uncertainties["stddev1"])
    assert np.allclose(uncertainties["stddev2"], fb_uncertainties["stddev2"])
    assert np.allclose(uncertainties["num_samples1"], fb_uncertainties["num_samples1"])
    assert np.allclose(uncertainties["num_samples2"], fb_uncertainties["num_samples2"])
