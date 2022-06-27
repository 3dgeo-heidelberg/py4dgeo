from py4dgeo.segmentation import *
from py4dgeo.m3c2 import M3C2
from py4dgeo.util import Py4DGeoError

import pytest

from .helpers import compare_segmentations


def test_segmentation(analysis):
    # Basic assertions about the analysis loaded in fixture
    assert len(analysis.distances.shape) == 2
    assert len(analysis.uncertainties.shape) == 2
    assert len(analysis.corepoints.cloud.shape) == 2
    assert len(analysis.timedeltas) > 0


def test_access_unassigned_properties(tmp_path, epochs):
    analysis = SpatiotemporalAnalysis(os.path.join(tmp_path, "unassigned.zip"))

    with pytest.raises(Py4DGeoError):
        analysis.corepoints

    with pytest.raises(Py4DGeoError):
        analysis.reference_epoch

    assert len(analysis.timedeltas) == 0

    # Set reference_epocha and corepoints to check unassigned distances
    # and uncertainties
    epoch, _ = epochs
    epoch.timestamp = "March 9th 2022, 16:33"
    analysis.reference_epoch = epochs[0]
    analysis.corepoints = epochs[0]

    assert analysis.distances.shape[1] == 0
    assert analysis.uncertainties.shape[1] == 0


def test_construct_from_epochs(epochs, tmp_path):
    ref_epoch, epoch1 = epochs

    ref_epoch.timestamp = "March 9th 2022, 16:32"
    epoch1.timestamp = "March 9th 2022, 16:33"

    m3c2 = M3C2(
        epochs=(ref_epoch, epoch1),
        corepoints=ref_epoch.cloud,
        cyl_radii=[2.0],
        normal_radii=[2.0],
    )

    analysis = SpatiotemporalAnalysis(os.path.join(tmp_path, "testanalysis.zip"))
    analysis.m3c2 = m3c2
    analysis.reference_epoch = ref_epoch
    analysis.corepoints = ref_epoch.cloud
    analysis.add_epochs(epoch1)

    assert analysis.distances.shape[1] == 1
    assert analysis.uncertainties.shape[1] == 1

    # Adding epoch again to trigger the code path overriding existing results
    analysis.add_epochs(epoch1)


def test_construct_from_scratch(tmp_path, epochs):
    ref_epoch, epoch = epochs
    ref_epoch.timestamp = "March 9th 2022, 16:32"
    epoch.timestamp = "March 9th 2022, 16:33"
    analysis = SpatiotemporalAnalysis(os.path.join(tmp_path, "scratch.zip"))

    analysis.reference_epoch = ref_epoch
    analysis.corepoints = epoch.cloud
    analysis.distances = np.zeros(shape=(analysis.corepoints.cloud.shape[0], 1))
    analysis.uncertainties = np.empty(
        (analysis.corepoints.cloud.shape[0], 0),
        dtype=np.dtype(
            [
                ("lodetection", "<f8"),
                ("spread1", "<f8"),
                ("num_samples1", "<i8"),
                ("spread2", "<f8"),
                ("num_samples2", "<i8"),
            ]
        ),
    )
    analysis.timedeltas = [epoch.timestamp - ref_epoch.timestamp]


def test_modification_raises(analysis):
    # The property setters intended for construction from scratch cannot
    # be used on an existing analysis object
    with pytest.raises(Py4DGeoError):
        analysis.distances = analysis.distances

    with pytest.raises(Py4DGeoError):
        analysis.timedeltas = analysis.timedeltas


def test_region_growing_algorithm(analysis, tmp_path):
    algo = RegionGrowingAlgorithm(neighborhood_radius=2.0, seed_subsampling=20)

    # We need better testing data for this
    objects = algo.run(analysis)

    # Check that seeds and objects were stored
    assert analysis.seeds
    assert len(objects) == len(analysis.objects)

    imgfile = os.path.join(tmp_path, "object.png")
    objects[0].plot(filename=imgfile)
    assert os.path.isfile(imgfile)


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
