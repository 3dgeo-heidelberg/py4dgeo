#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py4dgeo/compute.hpp"
#include "py4dgeo/py4dgeo.hpp"
#include "py4dgeo/pybind11_numpy_interop.hpp"

#include <sstream>
#include <string>
#include <tuple>

namespace py = pybind11;

namespace py4dgeo {

PYBIND11_MODULE(_py4dgeo, m)
{
  m.doc() = "Python Bindings for py4dgeo";

  // Expose the KDTree class
  py::class_<KDTree> kdtree(m, "KDTree", py::buffer_protocol());

  // Map __init__ to constructor
  kdtree.def(py::init<>(&KDTree::create));

  // Allow building the KDTree structure
  kdtree.def(
    "build_tree", &KDTree::build_tree, "Trigger building the search tree");

  // Expose the precomputation interface
  kdtree.def("precompute",
             &KDTree::precompute,
             "Precompute radius searches for a number of query points");

  // Add all the radius search methods
  kdtree.def(
    "radius_search",
    [](const KDTree& self, py::array_t<double> qp, double radius) {
      // Get a pointer for the query point
      double* ptr = static_cast<double*>(qp.request().ptr);

      KDTree::RadiusSearchResult result;
      self.radius_search(ptr, radius, result);

      return as_pyarray(std::move(result));
    },
    "Search point in given radius!");

  kdtree.def("precomputed_radius_search",
             [](const KDTree& self, IndexType idx, double radius) {
               KDTree::RadiusSearchResult result;
               self.precomputed_radius_search(idx, radius, result);
               return as_pyarray(std::move(result));
             });

  // Pickling support for the KDTree data structure
  kdtree.def(py::pickle(
    [](const KDTree& self) {
      std::stringstream buf;
      self.to_stream(buf);
      return py::bytes(buf.str());
    },
    [](const py::bytes& data) {
      std::stringstream buf(data.cast<std::string>());
      return KDTree::from_stream(buf);
    }));

  // Add compute interfaces
  m.def("compute_multiscale_directions",
        &compute_multiscale_directions,
        "Compute M3C2 multiscale directions");

  m.def("radius_workingset_finder", &radius_workingset_finder);
  m.def("cylidner_workingset_finder", &cylinder_workingset_finder);
  m.def("compute_distances", &compute_distances, "Compute M3C2 distances");
}

} // namespace py4dgeo
