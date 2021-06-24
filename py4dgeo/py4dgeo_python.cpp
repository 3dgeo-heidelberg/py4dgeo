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

} // namespace impl

PYBIND11_MODULE(_py4dgeo, m)
{
  m.doc() = "Python Bindings for py4dgeo";

  py::class_<PCLPointCloud>(m, "PCLPointCloud", py::buffer_protocol())
    .def(py::init<>(&impl::constructor))
    .def("build_tree",
         &PCLPointCloud::build_tree,
         "Trigger building the search tree")
    .def(
      "radius_search", &impl::radius_search, "Search points in given radius");

  py::enum_<SearchStrategy>(m, "SearchStrategy")
    .value("kdtree", SearchStrategy::kdtree)
    .value("octree", SearchStrategy::octree)
    .value("bruteforce", SearchStrategy::bruteforce);
}

} // namespace py4dgeo
