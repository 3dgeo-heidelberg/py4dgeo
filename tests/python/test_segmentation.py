from py4dgeo.segmentation import *

import os
import tempfile

from .helpers import compare_segmentations


def test_segmentation(segmentation):
    # Basic assertions about the segmentation generated in fixture
    assert segmentation.distances.shape[0] == 1


def test_save_load_segmentation(segmentation):
    # Write and read the segmentation
    with tempfile.TemporaryDirectory() as dir:
        filename = os.path.join(dir, "bla")
        segmentation.save(os.path.join(dir, "bla"))
        loaded = SpatiotemporalSegmentation.load(filename, segmentation._m3c2)

    # Assert segmentations are the same
    compare_segmentations(segmentation, loaded)
