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


def test_filebacked_segmentation(spatiotemporal):
    filename = spatiotemporal.filename
    del spatiotemporal
    analysis = SpatiotemporalAnalysis(filename)
    assert analysis.distances.shape[1] == 1
    assert analysis.uncertainties.shape[1] == 1
    assert len(analysis.timedeltas) == 1

    with pytest.raises(Py4DGeoError):
        analysis.reference_epoch = analysis.reference_epoch


def test_construct_from_scratch(tmp_path, epochs):
    ref_epoch, epoch = epochs
    ref_epoch.timestamp = "March 9th 2022, 16:32"
    epoch.timestamp = "March 9th 2022, 16:33"
    analysis = SpatiotemporalAnalysis(os.path.join(tmp_path, "scratch.zip"))

    analysis.reference_epoch = ref_epoch
    analysis.corepoints = epoch.cloud
    analysis.distances = np.zeros(shape=(analysis.corepoints.cloud.shape[0], 1))
    analysis.timedeltas = [epoch.timestamp - ref_epoch.timestamp]


def test_modification_raises(spatiotemporal):
    # The property setters intended for construction from scratch cannot
    # be used on an existing analysis object
    with pytest.raises(Py4DGeoError):
        spatiotemporal.distances = spatiotemporal.distances

    with pytest.raises(Py4DGeoError):
        spatiotemporal.uncertainties = spatiotemporal.uncertainties

    with pytest.raises(Py4DGeoError):
        spatiotemporal.timedeltas = spatiotemporal.timedeltas


def test_region_growing_algorithm(spatiotemporal):
    algo = RegionGrowingAlgorithm()

    # We need better testing data for this
    # objects = algo.run(spatiotemporal)


def test_region_growing_seed():
    # Construct a seed
    seed = RegionGrowingSeed(0, 0, 1)

    # Assert the given properties
    assert seed.index == 0
    assert seed.start_epoch == 0
    assert seed.end_epoch == 1

    # Pickle and unpickle, then assert equality
    restored = pickle.loads(pickle.dumps(seed))

    assert seed.index == restored.index
    assert seed.start_epoch == restored.start_epoch
    assert seed.end_epoch == restored.end_epoch


def test_regular_corepoint_grid():
    grid = regular_corepoint_grid((0, 0), (1, 1), (4, 4))
    assert grid.shape == (16, 3)
