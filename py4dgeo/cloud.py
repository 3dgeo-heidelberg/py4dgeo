import _py4dgeo


class PCLPointCloud(_py4dgeo.PCLPointCloud):
    def __init__(self, points):
        # Instantiate the base class from the C++ bindings
        super(PCLPointCloud, self).__init__(points)


class NFPointCloud2(_py4dgeo.NFPointCloud2):
    def __init__(self, points):
        # Instantiate the base class from the C++ bindings
        super(NFPointCloud2, self).__init__(points)
