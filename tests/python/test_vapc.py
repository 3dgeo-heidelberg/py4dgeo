import numpy as np
import pytest

from py4dgeo.epoch import Epoch
from py4dgeo.vapc import Vapc, enable_trace, enable_timeit

enable_trace(False)
enable_timeit(False)


def _simple_vapc():
    points = np.array(
        [
            [0.1, 0.1, 0.1],  # (0,0,0)
            [0.9, 0.2, 0.1],  # (0,0,0)
            [1.1, 0.1, 0.1],  # (1,0,0)
            [1.9, 0.1, 0.1],  # (1,0,0)
            [0.1, 1.1, 0.1],  # (0,1,0)
        ],
        dtype=float,
    )
    return Vapc(Epoch(points), voxel_size=1.0, origin=[0.0, 0.0, 0.0])


def _simple_vapc_with_extra():
    points = np.array(
        [
            [0.1, 0.1, 0.1],  # (0,0,0)
            [0.9, 0.2, 0.1],  # (0,0,0)
            [1.1, 0.1, 0.1],  # (1,0,0)
            [1.9, 0.1, 0.1],  # (1,0,0)
            [0.1, 1.1, 0.1],  # (0,1,0)
        ],
        dtype=float,
    )
    extra = np.zeros(points.shape[0], dtype=[("intensity", "f8")])
    extra["intensity"] = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return Vapc(
        Epoch(points, additional_dimensions=extra),
        voxel_size=1.0,
        origin=[0.0, 0.0, 0.0],
    )


def test_group_voxels_counts():
    v = _simple_vapc()
    v.group()

    counts = {tuple(vox): c for vox, c in zip(v.unique_voxels, v.counts)}
    assert counts == {(0, 0, 0): 2, (1, 0, 0): 2, (0, 1, 0): 1}


def test_centroids_and_centers():
    v = _simple_vapc()
    centroids = v.compute_centroids()
    centers = v.compute_voxel_centers()

    centroid_map = {tuple(vox): centroids[i] for i, vox in enumerate(v.unique_voxels)}
    center_map = {tuple(vox): centers[i] for i, vox in enumerate(v.unique_voxels)}

    assert np.allclose(centroid_map[(0, 0, 0)], [0.5, 0.15, 0.1])
    assert np.allclose(centroid_map[(1, 0, 0)], [1.5, 0.1, 0.1])
    assert np.allclose(centroid_map[(0, 1, 0)], [0.1, 1.1, 0.1])

    assert np.allclose(center_map[(0, 0, 0)], [0.5, 0.5, 0.5])
    assert np.allclose(center_map[(1, 0, 0)], [1.5, 0.5, 0.5])
    assert np.allclose(center_map[(0, 1, 0)], [0.5, 1.5, 0.5])


def test_reduce_to_centroid():
    v = _simple_vapc()
    reduced = v.reduce_to_feature("centroid")

    assert reduced.epoch.cloud.shape[0] == v.unique_voxels.shape[0]
    assert np.allclose(reduced.epoch.cloud, v.compute_centroids())


def test_reduce_to_voxel_center():
    v = _simple_vapc()
    reduced = v.reduce_to_feature("voxel_center")

    assert reduced.epoch.cloud.shape[0] == v.unique_voxels.shape[0]
    assert np.allclose(reduced.epoch.cloud, v.compute_voxel_centers())
    assert reduced.original_point_cloud_indices is None


@pytest.mark.parametrize(
    "feature_name,expected_fn",
    [
        ("closest_to_centroids", Vapc.compute_closest_to_centroids),
        ("closest_to_voxel_centers", Vapc.compute_closest_to_voxel_centers),
    ],
)
def test_reduce_to_representative_points(feature_name, expected_fn):
    v = _simple_vapc()
    expected = expected_fn(v)
    reduced = v.reduce_to_feature(feature_name)

    # Representative-point reductions should select real points.
    assert np.allclose(reduced.epoch.cloud, expected)
    assert reduced.original_point_cloud_indices is not None


def test_delta_vapc_labels():
    v1 = _simple_vapc()
    v2 = Vapc(
        Epoch(
            np.array(
                [
                    [1.1, 0.1, 0.1],  # (1,0,0)
                    [2.1, 0.1, 0.1],  # (2,0,0)
                ],
                dtype=float,
            )
        ),
        voxel_size=1.0,
        origin=[0.0, 0.0, 0.0],
    )

    delta = v1.delta_vapc(v2)
    label_map = {
        tuple(vox): lab
        for vox, lab in zip(delta.unique_voxels, delta.out["delta_vapc"])
    }
    assert label_map == {
        (0, 0, 0): 1,
        (0, 1, 0): 1,
        (1, 0, 0): 3,
        (2, 0, 0): 2,
    }


@pytest.mark.parametrize(
    "mode,expected_voxels",
    [
        ("in", {(0, 0, 0), (0, 1, 0)}),
        ("out", {(1, 0, 0)}),
    ],
)
def test_select_by_mask(mode, expected_voxels):
    v = _simple_vapc()
    mask_points = np.array(
        [
            [0.2, 0.2, 0.2],  # (0,0,0)
            [0.2, 1.2, 0.2],  # (0,1,0)
        ],
        dtype=float,
    )
    mask = Vapc(Epoch(mask_points), voxel_size=1.0, origin=[0.0, 0.0, 0.0])

    selected, sel = v.select_by_mask(mask, segment_in_or_out=mode)
    assert sel.shape[0] == v.epoch.cloud.shape[0]

    # Selection is voxel-based, so we check the resulting voxel set.
    selected.group()
    assert set(map(tuple, selected.unique_voxels)) == expected_voxels


def test_closest_to_voxel_centers_membership():
    v = _simple_vapc()
    closest = v.compute_closest_to_voxel_centers()
    coords = np.asarray(v.epoch.cloud)

    # Each representative must be a point from its voxel.
    for i in range(v.unique_voxels.shape[0]):
        pts = coords[v.inverse == i]
        assert any(np.allclose(closest[i], p) for p in pts)


def test_closest_to_centroids_min_distance():
    v = _simple_vapc()
    centroids = v.compute_centroids()
    closest = v.compute_closest_to_centroids()
    coords = np.asarray(v.epoch.cloud)

    # Validate that each representative minimizes distance to the centroid.
    for i in range(v.unique_voxels.shape[0]):
        pts = coords[v.inverse == i]
        d2 = np.sum((pts - centroids[i]) ** 2, axis=1)
        min_d2 = np.min(d2)
        chosen_d2 = np.sum((closest[i] - centroids[i]) ** 2)
        assert np.isclose(chosen_d2, min_d2)


def test_compute_features_keys_and_lengths():
    v = _simple_vapc()
    features = ["count", "centroid", "voxel_center", "eigenvalues", "covariance"]
    out = v.compute_features(features)

    expected_keys = {
        "count",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "voxel_center_x",
        "voxel_center_y",
        "voxel_center_z",
        "eigenvalue_1",
        "eigenvalue_2",
        "eigenvalue_3",
        "cov_xx",
        "cov_xy",
        "cov_xz",
        "cov_yx",
        "cov_yy",
        "cov_yz",
        "cov_zx",
        "cov_zy",
        "cov_zz",
    }

    assert expected_keys.issubset(out.keys())
    for key in expected_keys:
        assert out[key].shape[0] == v.unique_voxels.shape[0]


def test_map_features_to_points_count():
    v = _simple_vapc()
    v.compute_features(["count"])
    mapped = v.map_features_to_points()

    assert mapped["count"].shape[0] == v.epoch.cloud.shape[0]
    for i in range(v.unique_voxels.shape[0]):
        assert np.all(mapped["count"][v.inverse == i] == v.counts[i])


def test_reduce_to_centroid_aggregates_extra_dims():
    v = _simple_vapc_with_extra()
    reduced = v.reduce_to_feature("centroid")

    # Extra dimensions should be averaged per voxel in centroid mode.
    expected = np.zeros(v.unique_voxels.shape[0], dtype=float)
    for i in range(v.unique_voxels.shape[0]):
        expected[i] = v.extra_dims["intensity"][v.inverse == i].mean()
    assert np.allclose(reduced.out["intensity"], expected)


def test_map_features_to_points_includes_extra_dims():
    v = _simple_vapc_with_extra()
    v.compute_features(["count"])
    mapped = v.map_features_to_points()

    assert "intensity" in mapped
    assert np.allclose(mapped["intensity"], v.extra_dims["intensity"])


def test_bitemporal_mahalanobis_identical():
    v1 = _simple_vapc()
    v2 = _simple_vapc()

    # Identical clouds should yield only shared voxels and zero distances.
    result = v1.compute_bitemporal_mahalanobis(v2, alpha=0.999, min_points=1)

    assert np.array_equal(np.unique(result.out["change_type"]), np.array([3]))
    assert np.allclose(result.out["mahalanobis"], 0.0)
    assert np.all(result.out["significance"] == 0)
    assert np.all(result.out["changed"] == 0)


def test_reduce_to_representative_preserves_extra_dims():
    v = _simple_vapc_with_extra()
    reduced = v.reduce_to_feature("closest_to_centroids")

    idx = reduced.original_point_cloud_indices
    assert idx is not None
    assert np.allclose(reduced.out["intensity"], v.extra_dims["intensity"][idx])


def test_compute_features_count_and_density():
    v = _simple_vapc()
    out = v.compute_features(["count", "density"])

    assert np.array_equal(out["count"], v.counts)
    expected_density = v.counts / (v.voxel_size**3)
    assert np.allclose(out["density"], expected_density)


def test_bitemporal_mahalanobis_no_overlap():
    v1 = _simple_vapc()
    v2 = Vapc(
        Epoch(
            np.array(
                [
                    [2.1, 0.1, 0.1],  # (2,0,0)
                    [2.9, 0.2, 0.1],  # (2,0,0)
                ],
                dtype=float,
            )
        ),
        voxel_size=1.0,
        origin=[0.0, 0.0, 0.0],
    )

    result = v1.compute_bitemporal_mahalanobis(v2, alpha=0.999, min_points=1)

    assert set(np.unique(result.out["change_type"])) == {1, 2}
    assert np.all(np.isnan(result.out["mahalanobis"]))
    assert np.all(np.isnan(result.out["p_value"]))
    assert np.all(result.out["significance"] == 0)
    assert np.all(result.out["changed"] == 1)


def test_save_as_las_with_mapped_features(tmp_path):
    v = _simple_vapc_with_extra()
    v.compute_features(["density"])
    mapped = v.map_features_to_points()
    expected_density = mapped["density"].copy()

    outfile = tmp_path / "vapc.las"
    v.save_as_las(str(outfile))

    import laspy

    las = laspy.read(str(outfile))
    assert hasattr(las, "density")
    assert np.allclose(las["density"], expected_density)


def test_save_as_ply_with_features(tmp_path):
    plyfile = pytest.importorskip("plyfile")

    v = _simple_vapc()
    v.compute_features(["count"])

    outfile = tmp_path / "vapc.ply"
    v.save_as_ply(str(outfile), mode="voxel_center", features=["count"])

    ply = plyfile.PlyData.read(str(outfile))
    voxel_count = v.unique_voxels.shape[0]
    assert ply["vertex"].count == voxel_count * 8
    assert ply["face"].count == voxel_count * 12
    assert "count" in ply["vertex"].data.dtype.names
    assert np.allclose(ply["vertex"].data["count"].reshape(-1, 8)[:, 0], v.out["count"])
