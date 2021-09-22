import _py4dgeo


class KDTree(_py4dgeo.KDTree):
    def __init__(self, points):
        # Instantiate the base class from the C++ bindings
        super(KDTree, self).__init__(points)

    def build_tree(self, leaf=10):
        return _py4dgeo.KDTree.build_tree(self, leaf)

    def precompute(self, querypoints, maxradius):
        return _py4dgeo.KDTree.precompute(self, querypoints, maxradius)

    def radius_search(self, query_point, radius):
        # This looks like an entire useless repetition, but it gives
        # me a stackframe that my statistic profiler sees and I can
        # present the docstring here quite nicely.
        return _py4dgeo.KDTree.radius_search(self, query_point, radius)

    def precomputed_radius_search(self, idx, radius):
        return _py4dgeo.KDTree.precomputed_radius_search(self, idx, radius)
