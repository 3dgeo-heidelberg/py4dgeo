#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "geolib4d/geolib4d.hpp"

namespace py = pybind11;

namespace geolib4d {

PYBIND11_MODULE(geolib4d, m)
{
  m.doc() = "Python Bindings for GeoLib4d";
}

} // namespace geolib4d
