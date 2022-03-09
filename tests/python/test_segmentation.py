from py4dgeo.segmentation import *
from py4dgeo.m3c2 import M3C2


def test_segmentation(epochs):
    ref_epoch, epoch1 = epochs

    ref_epoch.timestamp = "March 9th 2022, 16:32"
    epoch1.timestamp = "March 9th 2022, 16:33"

    # TODO M3C2 should be refactored to not necessarily take epochs
    m3c2 = M3C2(
        epochs=(ref_epoch, epoch1),
        corepoints=ref_epoch.cloud,
        cyl_radii=[2.0],
        normal_radii=[2.0],
    )

    seg = SpatiotemporalSegmentation(m3c2=m3c2, reference_epoch=ref_epoch)
    seg.add_epoch(epoch1)
