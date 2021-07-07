import _py4dgeo


class KDTree(_py4dgeo.NFPointCloud2):
    def __init__(self, points):
        # Instantiate the base class from the C++ bindings
        super(KDTree, self).__init__(points)
