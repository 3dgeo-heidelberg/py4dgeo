#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "geolib4d/geolib4d.hpp"

namespace py = pybind11;

namespace geolib4d {

PYBIND11_MODULE(geolib4d, m) {
  m.doc() = "Python Bindings for GeoLib4d";

  py::class_<PointCloud>(m, "PointCloud")
      .def(py::init<>())
      .def("from_file", &PointCloud::from_file);
}

} // namespace geolib4d
