from py4dgeo import LevelSetAlgorithm


def test_LevelSetAlgorithm(analysis):
    # Basic assertions about the analysis loaded in fixture
    analysis.invalidate_results(seeds=False, objects=True)
    algo = py4dgeo.LevelSetAlgorithm(
        first_timestep=0,
        last_timestep=4,
        timestep_interval=10,
        alpha=0.1,
        iou_threshold=0.5,
    )

    algo.run(analysis)

    objects = analysis.objects

    assert len(objects) == 1

    assert set(objects[0].indices.keys()) == set(0, 1, 2, 3)
    assert set(objects[0].distances.keys()) == set(0, 1, 2, 3)
    assert set(objects[0].coordinates.keys()) == set(0, 1, 2, 3)

    assert len(objects[0].indices[0]) == 533
    assert len(objects[0].distances[0]) == 533
    assert objects[0].coordinates[0].shape == (533, 3)
    assert len(objects[0].polygons.values()) == 4

    assert objects[0].plot()
