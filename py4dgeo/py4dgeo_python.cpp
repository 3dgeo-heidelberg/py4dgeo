#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py4dgeo/py4dgeo.hpp"
#include "py4dgeo/pybind11_numpy_interop.hpp"

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
  std::vector<std::size_t> indices;
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
    .def(py::init<>(&impl::construct_kdtree))
    .def("build_tree", &KDTree::build_tree, "Trigger building the search tree")
    .def("radius_search", &impl::radius_search, "Search point in given radius")
    .def(py::pickle(
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
