#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py4dgeo/py4dgeo.hpp"
#include "py4dgeo/pybind11_helpers.hpp"

#include <tuple>

namespace py = pybind11;

namespace py4dgeo {

namespace impl {

PCLPointCloud
constructor(py::array_t<float> a)
{
  py::buffer_info info = a.request();

  // Make sure that the shape is correct
  if (info.shape[1] != 3)
    throw std::runtime_error("Shape of np.array expected to be (n, 3)");

  return PCLPointCloud(static_cast<float*>(info.ptr), info.shape[0]);
}

std::tuple<py::array_t<int>, py::array_t<float>>
radius_search(PCLPointCloud& self, py::array_t<float> point, double radius)
{
  // Create a PointXYZ instance from numpy array of shape (3,)
  float* ptr = static_cast<float*>(point.request().ptr);
  pcl::PointXYZ pcl_point(ptr[0], ptr[1], ptr[2]);

  // Allocate two vectors for the search algorithm
  NumpyVector<int> indices;
  NumpyVector<float> distances;

  // Do the actual work
  self.radius_search(pcl_point, radius, indices.as_std(), distances.as_std());

  // Return a tuple of numpy arrays
  return std::make_tuple(indices.as_numpy(), distances.as_numpy());
}

py::array_t<float>
points(PCLPointCloud& self)
{
  // No-op py capsule to forbid Python any memory management
  // The ownership of the memory should always be with the C++ class
  py::capsule noop(&(self._cloud->points), [](void*) {});

  // This needs a custom definition of shape and strides to account
  // for the special SSE-friendly memory layout of pcl::PointXYZ.
  return py::array_t<float>(
    { py::ssize_t(self._cloud->points.size()), py::ssize_t(3) },
    { sizeof(float) * 4, sizeof(float) },
    reinterpret_cast<float*>(self._cloud->points.data()),
    noop);
}

NFPointCloud
nf_constructor(py::array_t<float> a)
{
  py::buffer_info info = a.request();

  // Make sure that the shape is correct
  if (info.shape[1] != 3)
    throw std::runtime_error("Shape of np.array expected to be (n, 3)");

  return NFPointCloud(static_cast<float*>(info.ptr), info.shape[0]);
}

std::size_t
nf_radius_search(NFPointCloud& self, py::array_t<float> point, double radius)
{
  // Extract the float* for the query point
  float* ptr = static_cast<float*>(point.request().ptr);

  // Allocate two vectors for the search algorithm
  NumpyVector<std::pair<NFPointCloud::KDTree::IndexType, float>> result;

  // Do the actual work
  auto ret = self.radius_search(ptr, radius, result.as_std());

  // for(auto [i, d] : result.as_std())
  //   std::cout << "bla Index " << i << " dist: " << std::sqrt(d) << std::endl;

  return ret;

  // TODO: Why on earth does this need to be std::vector<std::pair>???
}

NFPointCloud2
nf_constructor2(py::array_t<float> a)
{
  py::buffer_info info = a.request();

  // Make sure that the shape is correct
  if (info.shape[1] != 3)
    throw std::runtime_error("Shape of np.array expected to be (n, 3)");

  return NFPointCloud2(static_cast<float*>(info.ptr), info.shape[0]);
}

std::tuple<py::array_t<std::size_t>, py::array_t<float>>
nf_radius_search2(NFPointCloud2& self, py::array_t<float> point, double radius)
{
  // Extract the float* for the query point
  float* ptr = static_cast<float*>(point.request().ptr);

  // Allocate two vectors for the search algorithm
  std::vector<std::pair<std::size_t, float>> result;

  // Do the actual work
  auto ret = self.radius_search(ptr, radius, result);

  // TODO: Why on earth does this need to be std::vector<std::pair>???
  //       That absolutely ruins my ability to return the result as a
  //       a non-copy. For now, make copies instead.
  NumpyVector<std::size_t> indices;
  NumpyVector<float> distances;

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

  py::class_<PCLPointCloud>(m, "PCLPointCloud", py::buffer_protocol())
    .def(py::init<>(&impl::constructor))
    .def("build_tree",
         &PCLPointCloud::build_tree,
         "Trigger building the search tree")
    .def("radius_search", &impl::radius_search, "Search points in given radius")
    .def(
      "_points", &impl::points, "Direct access to the underlying point data");

  py::enum_<SearchStrategy>(m, "SearchStrategy")
    .value("kdtree", SearchStrategy::kdtree)
    .value("octree", SearchStrategy::octree)
    .value("bruteforce", SearchStrategy::bruteforce);

  py::class_<NFPointCloud>(m, "NFPointCloud", py::buffer_protocol())
    .def(py::init<>(&impl::nf_constructor))
    .def("build_tree",
         &NFPointCloud::build_tree,
         "Trigger building the search tree")
    .def(
      "radius_search", &impl::nf_radius_search, "Search point in given radius");

  py::class_<NFPointCloud2>(m, "NFPointCloud2", py::buffer_protocol())
    .def(py::init<>(&impl::nf_constructor2))
    .def("build_tree",
         &NFPointCloud2::build_tree,
         "Trigger building the search tree")
    .def("radius_search",
         &impl::nf_radius_search2,
         "Search point in given radius");
}

} // namespace py4dgeo
