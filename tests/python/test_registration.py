from py4dgeo.registration import *


def test_icp(epochs):
    epoch1, epoch2 = epochs
    epoch1.calculate_normals(radius=2.5)
    epoch2.calculate_normals(radius=2.5)

    trafo2 = point_to_plane_icp(epoch1, epoch2)

    assert trafo2.affine_transformation.shape == (4, 4)
