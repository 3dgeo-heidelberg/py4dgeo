import _py4dgeo


class PCLPointCloud(_py4dgeo.PCLPointCloud):
    def __init__(self, points):
        # Instantiate the base class from the C++ bindings
        super(PCLPointCloud, self).__init__(points)
