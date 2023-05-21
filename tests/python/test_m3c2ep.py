from py4dgeo.m3c2ep import *
from py4dgeo.util import Py4DGeoError

import pytest
import tempfile
import os


def test_m3c2ep(epochs, Cxx, tfM, redPoint):
    epoch1, epoch2 = epochs
    corepoints = epoch1.cloud[::25]

    # Instantiate an M3C2 instance
    m3c2ep = M3C2EP(
        epochs=(epoch1, epoch2),
        corepoints=corepoints,
        normal_radii=(0.5, 1.0, 2.0),
        cyl_radii=(0.5,),
        max_distance=3.0,
        Cxx=Cxx,
        tfM=tfM,
        refPointMov=redPoint,
    )

    # Run it and check results shapes
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


def test_m3c2ep_external_normals(epochs, Cxx, tfM, redPoint):
    epoch1, epoch2 = epochs
    corepoints = epoch1.cloud[::25]

    with pytest.raises(Py4DGeoError):
        d, u, c = M3C2EP(
            epochs=(epoch1, epoch2),
            corepoints=corepoints,
            corepoint_normals=np.array([[0, 0, 1], [0, 0, 1]]),
            cyl_radii=(0.5,),
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
        cyl_radii=(0.5,),
        max_distance=3.0,
        Cxx=Cxx,
        tfM=tfM,
        refPointMov=redPoint,
    )
    distances, uncertainties, covariance = m3c2ep.run()

    # Instantiate an M3C2 instance with specified corepoint normals
    corepoint_normals = m3c2ep.directions()
    m3c2ep_n = py4dgeo.M3C2EP(
        epochs=(epoch1, epoch2),
        corepoints=corepoints,
        corepoint_normals=corepoint_normals,
        cyl_radii=(0.5,),
        max_distance=3.0,
        Cxx=Cxx,
        tfM=tfM,
        refPointMov=redPoint,
    )
    # Run it and check that distances is same when specify the same corepoint normals
    distances_n, uncertainties_n, covariance_n = m3c2ep_n.run()
    diff = distances_n - distances
    diff[np.isnan(diff)] = 0
    assert np.allclose(diff, 0)

    # Instantiate an M3C2 instance with one direction
    corepoint_normals = np.array([[0, 0, 1]])
    m3c2ep = py4dgeo.M3C2EP(
        epochs=(epoch1, epoch2),
        corepoints=corepoints,
        corepoint_normals=corepoint_normals,
        cyl_radii=(0.5,),
        max_distance=3.0,
        Cxx=Cxx,
        tfM=tfM,
        refPointMov=redPoint,
    )
    # Run it and check that corepoint normals same as directions
    distances, uncertainties, covariance = m3c2ep.run()
    assert np.allclose(m3c2ep.directions(), corepoint_normals)


def test_m3c2ep_epoch_saveload(epochs):
    epoch1, epoch2 = epochs
    epoch1.build_kdtree()
    epoch2.build_kdtree()

    with tempfile.TemporaryDirectory() as dir:
        # Save and load it
        filename1 = os.path.join(dir, "epoch1")
        filename2 = os.path.join(dir, "epoch2")
        epoch1.save(filename1)
        epoch2.save(filename2)
        load1 = py4dgeo.load_epoch(filename1)
        load2 = py4dgeo.load_epoch(filename2)

        # Assert that the two object behave the same
        assert load1.cloud.shape[0] == epoch1.cloud.shape[0]
        assert load2.cloud.shape[0] == epoch2.cloud.shape[0]

        assert np.allclose(
            load1.kdtree.radius_search(np.array([0, 0, 0]), 10),
            epoch1.kdtree.radius_search(np.array([0, 0, 0]), 10),
        )
        assert np.allclose(
            load2.kdtree.radius_search(np.array([0, 0, 0]), 10),
            epoch2.kdtree.radius_search(np.array([0, 0, 0]), 10),
        )


def test_m3c2ep_write_las(epochs, Cxx, tfM, redPoint):
    epoch1, epoch2 = epochs
    corepoints = epoch1.cloud[::25]

    # Instantiate an M3C2 instance
    m3c2ep = M3C2EP(
        epochs=(epoch1, epoch2),
        corepoints=corepoints,
        normal_radii=(0.5, 1.0, 2.0),
        cyl_radii=(0.5,),
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
        m3c2ep.write_las(file, attribute_dict=attr)
        c, d, l = read_cp_from_las(file)
        diff_c = corepoints - c
        diff_d = distances - d
        diff_d[np.isnan(diff_d)] = 0
        diff_l = uncertainties["lodetection"] - l
        diff_l[np.isnan(diff_l)] = 0
        assert np.allclose(diff_c, 0)
        assert np.allclose(diff_d, 0)
        assert np.allclose(diff_l, 0)
