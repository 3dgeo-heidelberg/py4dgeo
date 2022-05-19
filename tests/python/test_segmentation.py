from py4dgeo.segmentation import *
from py4dgeo.util import Py4DGeoError

import pytest

from .helpers import compare_segmentations


def test_segmentation(spatiotemporal):
    # Basic assertions about the analysis generated in fixture
    assert spatiotemporal.distances.shape[1] == 1
    assert spatiotemporal.uncertainties.shape[1] == 1
    assert len(spatiotemporal.timedeltas) == 1

    with pytest.raises(Py4DGeoError):
        spatiotemporal.reference_epoch = spatiotemporal.reference_epoch


def test_construct_from_scratch(tmp_path):
    analysis = SpatiotemporalAnalysis(os.path.join(tmp_path, "scratch.zip"))


def test_modification_raises(spatiotemporal):
    # The property setters intended for construction from scratch cannot
    # be used on an existing analysis object
    with pytest.raises(Py4DGeoError):
        spatiotemporal.distances = spatiotemporal.distances

    with pytest.raises(Py4DGeoError):
        spatiotemporal.uncertainties = spatiotemporal.uncertainties

    with pytest.raises(Py4DGeoError):
        spatiotemporal.timedeltas = spatiotemporal.timedeltas
