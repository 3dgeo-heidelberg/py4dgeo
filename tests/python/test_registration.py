from py4dgeo.registration import *


def test_icp(epochs):
    epoch1, epoch2 = epochs
    epoch1.calculate_normals(radius=2.5)
    epoch2.calculate_normals(radius=2.5)

    trafo2 = point_to_plane_icp(epoch1, epoch2)

    assert trafo2.affine_transformation.shape == (4, 4)


def test_stable_area_icp(epochs):
    epoch1, epoch2 = epochs
    epoch1.calculate_normals(radius=10)
    epoch2.calculate_normals(radius=10)
    trafo2 = stable_area_icp(epoch1, epoch2, 10, 10, 0.2, 0.2, 0.5, 5, 1)

    assert trafo2.affine_transformation.shape == (4, 4)
