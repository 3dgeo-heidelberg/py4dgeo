from py4dgeo.m3c2ep import *
from py4dgeo.util import Py4DGeoError
from py4dgeo import write_m3c2_results_to_las

import pytest
import tempfile
import os


def test_m3c2ep(epochs_m3c2ep, Cxx, tfM, redPoint, scanpos_info):
    epoch1, epoch2 = epochs_m3c2ep
    epoch1.scanpos_info = scanpos_info
    epoch2.scanpos_info = scanpos_info
    corepoints = epoch1.cloud[::8000]
    print("corepoints shape", corepoints.shape)
    # Instantiate an M3C2 instance
    m3c2ep = M3C2EP(
        epochs=(epoch1, epoch2),
        corepoints=corepoints,
        normal_radii=(0.5, 1.0, 2.0),
        cyl_radius=0.5,
        max_distance=3.0,
        Cxx=Cxx,
        tfM=tfM,
        refPointMov=redPoint,
    )

    # Run it and check results exists with correct shapes
    distances, uncertainties, covariance = m3c2ep.run()

    assert distances.shape[0] == corepoints.shape[0]
    assert uncertainties["num_samples1"].shape[0] == corepoints.shape[0]
    assert uncertainties["num_samples2"].shape[0] == corepoints.shape[0]
    assert uncertainties["spread1"].shape[0] == corepoints.shape[0]
    assert uncertainties["spread2"].shape[0] == corepoints.shape[0]
    assert uncertainties["lodetection"].shape[0] == corepoints.shape[0]
    cov1 = covariance["cov1"]
    cov2 = covariance["cov2"]
    assert (
        cov1.shape[0] == corepoints.shape[0]
        and cov1.shape[1] == 3
        and cov1.shape[2] == 3
    )
    assert (
        cov2.shape[0] == corepoints.shape[0]
        and cov2.shape[1] == 3
        and cov2.shape[2] == 3
    )


def test_m3c2ep_external_normals(epochs_m3c2ep, Cxx, tfM, redPoint, scanpos_info):
    epoch1, epoch2 = epochs_m3c2ep
    epoch1.scanpos_info = scanpos_info
    epoch2.scanpos_info = scanpos_info
    corepoints = epoch1.cloud[::8000]

    # Run and check normals should be one direction or one normal per point.
    with pytest.raises(Py4DGeoError):
        d, u, c = M3C2EP(
            epochs=(epoch1, epoch2),
            corepoints=corepoints,
            corepoint_normals=np.array([[0, 0, 1], [0, 0, 1]]),
            cyl_radius=0.5,
            max_distance=3.0,
            Cxx=Cxx,
            tfM=tfM,
            refPointMov=redPoint,
        ).run()

    # Instantiate an default M3C2 instance to get normals
    m3c2ep = M3C2EP(
        epochs=(epoch1, epoch2),
        corepoints=corepoints,
        normal_radii=(0.5, 1.0, 2.0),
        cyl_radius=0.5,
        max_distance=3.0,
        Cxx=Cxx,
        tfM=tfM,
        refPointMov=redPoint,
    )

    # Instantiate an M3C2 instance with specified corepoint normals
    corepoint_normals = m3c2ep.directions()
    m3c2ep_n = py4dgeo.M3C2EP(
        epochs=(epoch1, epoch2),
        corepoints=corepoints,
        corepoint_normals=corepoint_normals,
        cyl_radius=0.5,
        max_distance=3.0,
        Cxx=Cxx,
        tfM=tfM,
        refPointMov=redPoint,
    )
    # Check that corepoint normals same as algorithm directions
    assert np.allclose(m3c2ep_n.directions(), corepoint_normals)

    # Instantiate an M3C2 instance with one direction
    corepoint_normals = np.array([[0, 0, 1]])
    m = py4dgeo.M3C2EP(
        epochs=(epoch1, epoch2),
        corepoints=corepoints,
        corepoint_normals=corepoint_normals,
        cyl_radius=0.5,
        max_distance=3.0,
        Cxx=Cxx,
        tfM=tfM,
        refPointMov=redPoint,
    )
    # Check that corepoint normals same as algorithm directions
    assert np.allclose(m.directions(), corepoint_normals)


def test_m3c2ep_epoch_saveload(epochs_m3c2ep, scanpos_info):
    epoch1, epoch2 = epochs_m3c2ep
    epoch1._validate_search_tree()
    epoch2._validate_search_tree()
    epoch1.scanpos_info = scanpos_info
    epoch2.scanpos_info = scanpos_info
    with tempfile.TemporaryDirectory() as dir:
        # Save and load it
        filename1 = os.path.join(dir, "epoch1")
        filename2 = os.path.join(dir, "epoch2")
        epoch1.save(filename1)
        epoch2.save(filename2)
        load1 = py4dgeo.load_epoch(filename1)
        load2 = py4dgeo.load_epoch(filename2)
        load1._validate_search_tree()
        load2._validate_search_tree()
        # Assert that the two object behave the same
        assert load1.cloud.shape[0] == epoch1.cloud.shape[0]
        assert load2.cloud.shape[0] == epoch2.cloud.shape[0]

        assert np.allclose(load1.cloud - epoch1.cloud, 0)
        assert np.allclose(load2.cloud - epoch2.cloud, 0)

        bbox_extent_epoch1 = epoch1.cloud.max(axis=0) - epoch1.cloud.min(axis=0)
        radius1 = 0.25 * np.min(
            bbox_extent_epoch1
        )  # Quarter of the extent of the smallest dimension
        query_point_epoch1 = 0.5 * (epoch1.cloud.min(axis=0) + epoch1.cloud.max(axis=0))
        assert np.allclose(
            load1._radius_search(query_point_epoch1, radius1),
            epoch1._radius_search(query_point_epoch1, radius1),
        )

        bbox_extent_epoch2 = epoch2.cloud.max(axis=0) - epoch2.cloud.min(axis=0)
        radius2 = 0.25 * np.min(
            bbox_extent_epoch2
        )  # Quarter of the extent of the smallest dimension
        query_point_epoch2 = 0.5 * (epoch2.cloud.min(axis=0) + epoch2.cloud.max(axis=0))
        assert np.allclose(
            load2._radius_search(query_point_epoch2, radius2),
            epoch2._radius_search(query_point_epoch2, radius2),
        )


def test_m3c2ep_write_las(epochs_m3c2ep, Cxx, tfM, redPoint, scanpos_info):
    epoch1, epoch2 = epochs_m3c2ep
    epoch1.scanpos_info = scanpos_info
    epoch2.scanpos_info = scanpos_info
    corepoints = epoch1.cloud[::8000]

    # Instantiate an M3C2 instance
    m3c2ep = M3C2EP(
        epochs=(epoch1, epoch2),
        corepoints=corepoints,
        normal_radii=(0.5, 1.0, 2.0),
        cyl_radius=0.5,
        max_distance=3.0,
        Cxx=Cxx,
        tfM=tfM,
        refPointMov=redPoint,
    )

    # Run it
    distances, uncertainties, covariance = m3c2ep.run()

    def read_cp_from_las(path):
        import laspy

        inFile = laspy.read(path)
        coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
        try:
            distances = getattr(inFile, "distances", None)
        except:
            distances = None

        try:
            lod = getattr(inFile, "lod", None)
        except:
            lod = None
        return coords, distances, lod

    # save and load from las file and check results are same
    with tempfile.TemporaryDirectory() as dir:
        attr = {"distances": distances, "lod": uncertainties["lodetection"]}
        file = dir + "cp.las"
        write_m3c2_results_to_las(file, m3c2ep, attribute_dict=attr)
        c, d, l = read_cp_from_las(file)
        diff_c = corepoints - c
        diff_d = distances - d
        diff_d[np.isnan(diff_d)] = 0
        diff_l = uncertainties["lodetection"] - l
        diff_l[np.isnan(diff_l)] = 0
        assert np.allclose(diff_c, 0)
        assert np.allclose(diff_d, 0)
        assert np.allclose(diff_l, 0)
