#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py4dgeo/py4dgeo.hpp"
#include "py4dgeo/pybind11_numpy_interop.hpp"

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

  kdtree.def(py::pickle(
    [](const KDTree& self) {
      // Instantiate a stream object
      std::stringstream buf;

      // Write the cloud itself. This is very unfortunate as it is a redundant
      // copy of the point cloud, but deserializing a point cloud without it
      // is impossible.
      buf.write(reinterpret_cast<const char*>(&self._adaptor.n),
                sizeof(std::size_t));
      buf.write(reinterpret_cast<const char*>(self._adaptor.ptr),
                sizeof(double) * self._adaptor.n * 3);

      // Write the search index
      self._search->saveIndex(buf);

      // Make and return a bytes object
      return py::bytes(buf.str());
    },
    [](const py::bytes& data) {
      std::stringstream buf(data.cast<std::string>());

      std::size_t n;
      buf.read(reinterpret_cast<char*>(&n), sizeof(std::size_t));
      auto obj = new KDTree(n);
      buf.read(reinterpret_cast<char*>(obj->_adaptor.ptr),
               sizeof(double) * n * 3);

      obj->_search = std::make_shared<KDTree::KDTreeImpl>(
        3, obj->_adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
      obj->_search->loadIndex(buf);
      return obj;
    }));
}

} // namespace py4dgeo
