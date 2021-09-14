from py4dgeo.directions import *

from . import epoch1

import numpy as np


def test_multi_constant_direction():
    vec = np.array([[1, 0, 0]])
    dir = MultiConstantDirection(directions=vec)
    assert (dir.get() == vec).all()
    assert dir.num_dirs == 1


def test_constant_direction():
    vec = np.array([1, 0, 0])
    dir = ConstantDirection(direction=vec)
    assert (dir.get()[0, :] == vec).all()
    assert dir.num_dirs == 1


def test_corepoint_direction():
    vec = np.array([[[1, 0, 0]]])
    dir = CorePointDirection(directions=vec)
    assert (dir.get(core_idx=0) == vec[0, :, :]).all()
    assert dir.num_dirs == 1


def test_multiscale_direction(epoch1):
    dir = MultiScaleDirection(radii=[2.0, 10.0])
    dir.precompute(epoch=epoch1, corepoints=epoch1.cloud)
    assert dir._precomputation[0].shape == epoch1.cloud.shape
    for i in range(epoch1.cloud.shape[0]):
        dir.get(core_idx=i)
