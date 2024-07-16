from py4dgeo import LevelSetAlgorithm
import platform
import pytest


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
    )

    algo.run(analysis)

    objects = analysis.objects

    assert len(objects) == 1

    assert objects[0].polygons[0].area == 548.5

    polygon_coords = [
        (81.0, 63.0),
        (78.0, 62.0),
        (72.0, 61.0),
        (71.0, 61.0),
        (69.0, 61.0),
        (67.0, 61.0),
        (66.0, 61.0),
        (64.0, 63.0),
        (60.0, 65.0),
        (60.0, 66.0),
        (59.0, 73.0),
        (59.0, 74.0),
        (59.0, 75.0),
        (59.0, 77.0),
        (60.0, 78.0),
        (62.0, 79.0),
        (67.0, 84.0),
        (69.0, 86.0),
        (70.0, 86.0),
        (71.0, 86.0),
        (72.0, 86.0),
        (73.0, 86.0),
        (75.0, 86.0),
        (76.0, 86.0),
        (77.0, 86.0),
        (79.0, 86.0),
        (81.0, 85.0),
        (82.0, 84.0),
        (84.0, 81.0),
        (85.0, 79.0),
        (85.0, 78.0),
        (86.0, 74.0),
        (87.0, 70.0),
        (86.0, 69.0),
        (85.0, 68.0),
        (81.0, 63.0),
    ]

    assert sorted(list(objects[0].polygons[0].exterior.coords)) == sorted(
        polygon_coords
    )
    assert set(objects[0].indices.keys()) == set((0, 1, 2, 3))
    assert set(objects[0].distances.keys()) == set((0, 1, 2, 3))
    assert set(objects[0].coordinates.keys()) == set((0, 1, 2, 3))

    assert len(objects[0].indices[0]) == 533
    assert len(objects[0].distances[0]) == 533
    assert objects[0].coordinates[0].shape == (533, 3)
    assert len(objects[0].polygons.values()) == 4

    assert objects[0].plot()
