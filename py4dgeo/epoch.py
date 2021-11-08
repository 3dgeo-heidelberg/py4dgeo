from py4dgeo.util import Py4DGeoError, as_double_precision, make_contiguous

import numpy as np

import _py4dgeo


class Epoch(_py4dgeo.Epoch):
    def __init__(self, cloud: np.ndarray):
        # Check the given array shapes
        if len(cloud.shape) != 2 or cloud.shape[1] != 3:
            raise Py4DGeoError("Clouds need to be an array of shape nx3")

        # Make sure that cloud is double precision and contiguous in memory
        cloud = as_double_precision(cloud)
        cloud = make_contiguous(cloud)

        # Call base class constructor
        super().__init__(cloud)

        # Build the KDTree index
        # TODO: When exactly should this be done and how do we want to
        #       the user interface to look like?
        self.kdtree.build_tree(10)
