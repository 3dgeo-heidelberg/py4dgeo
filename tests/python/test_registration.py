from py4dgeo.registration import *


def test_icp(epochs):
    epoch1, epoch2 = epochs
    epoch1.calculate_normals(radius=2.5)

    trafo1 = iterative_closest_point(epoch1, epoch2)
    trafo2 = point_to_plane_icp(epoch1, epoch2)
    trafo3 = point_to_plane_icp_LM(epoch1, epoch2)
    trafo4 = p_to_p_icp(epoch1, epoch2)

    assert trafo1.affine_transformation.shape == (4, 4)
    assert trafo2.affine_transformation.shape == (4, 4)
    assert trafo3.affine_transformation.shape == (4, 4)
    assert trafo4.affine_transformation.shape == (4, 4)
