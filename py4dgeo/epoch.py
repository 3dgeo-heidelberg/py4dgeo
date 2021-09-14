from py4dgeo.kdtree import KDTree
from py4dgeo.util import Py4DGeoError

import dataclasses
import numpy as np


@dataclasses.dataclass
class Epoch:
    cloud: np.ndarray = None
    kdtree: KDTree = None

    def __post_init__(self):
        # Check the given array shapes
        if len(self.cloud.shape) != 2 or self.cloud.shape[1] != 3:
            raise Py4DGeoError("Clouds need to be an array of shape nx3")

        # If the kdtree was not already built, build it now
        if self.kdtree is None:
            self.kdtree = KDTree(self.cloud)
            self.kdtree.build_tree(10)
