#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py4dgeo/py4dgeo.hpp"
#include "py4dgeo/pybind11_helpers.hpp"

#include <tuple>

namespace py = pybind11;

namespace py4dgeo {

namespace impl {

KDTree
construct_kdtree(py::array_t<double> a)
{
  py::buffer_info info = a.request();

  // Make sure that the shape is correct
  if (info.shape[1] != 3)
    throw std::runtime_error("Shape of np.array expected to be (n, 3)");

  return KDTree(static_cast<double*>(info.ptr), info.shape[0]);
}

std::tuple<py::array_t<std::size_t>, py::array_t<double>>
radius_search(KDTree& self, py::array_t<double> point, double radius)
{
  // Extract the float* for the query point
  double* ptr = static_cast<double*>(point.request().ptr);

  // Allocate two vectors for the search algorithm
  std::vector<std::pair<std::size_t, double>> result;

  // Do the actual work
  auto ret = self.radius_search(ptr, radius, result);

  // TODO: Why on earth does this need to be std::vector<std::pair>???
  //       That absolutely ruins my ability to return the result as a
  //       a non-copy. For now, make copies instead.
  NumpyVector<std::size_t> indices;
  NumpyVector<double> distances;

  indices.as_std().resize(result.size());
  distances.as_std().resize(result.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    indices.as_std()[i] = result[i].first;
    distances.as_std()[i] = result[i].second;
  }

  return std::make_tuple(indices.as_numpy(), distances.as_numpy());
}

} // namespace impl

PYBIND11_MODULE(_py4dgeo, m)
{
  m.doc() = "Python Bindings for py4dgeo";

  py::class_<KDTree>(m, "NFPointCloud2", py::buffer_protocol())
    .def(py::init<>(&impl::construct_kdtree))
    .def("build_tree", &KDTree::build_tree, "Trigger building the search tree")
    .def("radius_search", &impl::radius_search, "Search point in given radius");
}

} // namespace py4dgeo
