from py4dgeo.cloudcompare import CloudCompareM3C2
from py4dgeo.m3c2 import M3C2

from . import compare_algorithms, epochs


def test_cloudcompare_m3c2(epochs):
    epoch1, epoch2 = epochs

    # Instantiate an M3C2 instance
    m3c2 = M3C2(
        epochs=(epoch1, epoch2),
        corepoints=epoch1.cloud,
        cyl_radii=(1.5,),
        normal_radii=(1.0,),
    )

    # Instantiate Cloud compare variant
    cc_m3c2 = CloudCompareM3C2(
        epochs=(epoch1, epoch2),
        corepoints=epoch1.cloud,
        searchscale=(3.0,),
        normalscale=(2.0,),
    )

    compare_algorithms(m3c2, cc_m3c2)
