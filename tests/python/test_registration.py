from py4dgeo.registration import *


def test_icp(epochs):
    epoch1, epoch2 = epochs

    trafo = iterative_closest_point(epoch1, epoch2)

    assert trafo.affine_transformation.shape == (4, 4)
