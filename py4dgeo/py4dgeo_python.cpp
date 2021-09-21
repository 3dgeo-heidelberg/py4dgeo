#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py4dgeo/py4dgeo.hpp"
#include "py4dgeo/pybind11_numpy_interop.hpp"

#include <tuple>

namespace py = pybind11;

namespace py4dgeo {

namespace impl {

std::tuple<py::array_t<IndexType>, py::array_t<double>>
radius_search(KDTree& self, py::array_t<double> point, double radius)
{
  // Extract the float* for the query point
  double* ptr = static_cast<double*>(point.request().ptr);

  // Allocate two vectors for the search algorithm
  std::vector<std::pair<IndexType, double>> result;

  // Do the actual work
  auto ret = self.radius_search(ptr, radius, result);

  // TODO: Why on earth does this need to be std::vector<std::pair>???
  //       That absolutely ruins my ability to return the result as a
  //       a non-copy. For now, make copies instead.
  std::vector<IndexType> indices;
  std::vector<double> distances;

  indices.resize(result.size());
  distances.resize(result.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    indices[i] = result[i].first;
    distances[i] = result[i].second;
  }

  return std::make_tuple(py4dgeo::as_pyarray(std::move(indices)),
                         py4dgeo::as_pyarray(std::move(distances)));
}

} // namespace impl

PYBIND11_MODULE(_py4dgeo, m)
{
  m.doc() = "Python Bindings for py4dgeo";

  py::class_<KDTree>(m, "KDTree", py::buffer_protocol())
    .def(py::init<>(&KDTree::create))
    .def("build_tree", &KDTree::build_tree, "Trigger building the search tree")
    .def("radius_search", &impl::radius_search, "Search point in given radius");

  // m.def("compute_multiscale_directions",
  //       &compute_multiscale_directions,
  //       "Calculate multiscale directions for M3C2");
}

} // namespace py4dgeo
