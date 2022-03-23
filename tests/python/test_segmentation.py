from py4dgeo.segmentation import *
from py4dgeo.util import Py4DGeoError

import pytest

from .helpers import compare_segmentations


def test_segmentation(spatiotemporal):
    # Basic assertions about the analysis generated in fixture
    assert spatiotemporal.distances.shape[0] == 1
    assert spatiotemporal.uncertainties.shape[0] == 1
    assert len(spatiotemporal.timedeltas) == 1

    with pytest.raises(Py4DGeoError):
        spatiotemporal.reference_epoch = spatiotemporal.reference_epoch
