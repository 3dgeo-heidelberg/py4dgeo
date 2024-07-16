from py4dgeo import LevelSetAlgorithm
import platform
import pytest
import numpy as np


def test_LevelSetAlgorithm(analysis):
    # Basic assertions about the analysis loaded in fixture
    analysis.invalidate_results(seeds=False, objects=True)

    if platform.system() == "Darwin":
        with pytest.raises(NotImplementedError):
            LevelSetAlgorithm()
        return 0

    algo = LevelSetAlgorithm(
        first_timestep=0,
        last_timestep=4,
        timestep_interval=10,
        alpha=0.1,
        iou_threshold=0.5,
        distance_threshold=1,
    )

    algo.run(analysis)

    objects = analysis.objects

    assert len(objects) == 1

    assert np.isclose(objects[0].polygons[0].area, 548.5, rtol=0.015)

    assert set(objects[0].indices.keys()) == set((0, 1, 2, 3))
    assert set(objects[0].distances.keys()) == set((0, 1, 2, 3))
    assert set(objects[0].coordinates.keys()) == set((0, 1, 2, 3))

    assert len(objects[0].polygons.values()) == 4

    assert objects[0].plot()
