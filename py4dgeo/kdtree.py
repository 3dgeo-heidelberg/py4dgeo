import _py4dgeo
import numpy as np


class KDTree(_py4dgeo.KDTree):
    def __init__(self, points):
        # Instantiate the base class from the C++ bindings
        super(KDTree, self).__init__(points)

    def radius_search(self, query_point, radius):
        # This looks like an entire useless repetition, but it gives
        # me a stackframe that my statistic profiler sees and I can
        # present the docstring here quite nicely.
        return _py4dgeo.KDTree.radius_search(self, query_point, radius)

    def build_tree(self, leaf=10):
        return _py4dgeo.KDTree.build_tree(self, leaf)


class FixedQueryKDTree:
    def __init__(self, kdtree, query_points):
        self.kdtree = kdtree
        self.query_points = query_points

    def fixed_radius_search(self, query_idx, radius):
        return self.kdtree.radius_search(self, self.query_points[query_idx, :], radius)

    def build_tree(self, leaf=10):
        self.kdtree.build_tree(leaf)


class CachedKDTree:
    def __init__(self, kdtree, query_points, maxradius):
        self.kdtree = kdtree
        self.query_points = query_points
        self.maxradius = maxradius

    def fixed_radius_search(self, query_idx, radius):
        indices, distances = self.cache[query_idx]
        pos = np.searchsorted(distances, radius, side="right")
        return indices[:pos], distances[:pos]

    def build_tree(self, leaf=10):
        # Make sure that the original tree is built
        self.kdtree.build_tree(leaf)

        # Evaluate at each core point
        self.cache = []
        for i in range(self.query_points.shape[0]):
            self.cache.append(
                self.kdtree.radius_search(self.query_points[i, :], self.maxradius)
            )
