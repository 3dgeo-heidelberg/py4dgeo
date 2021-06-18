#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py4dgeo/py4dgeo.hpp"

namespace py = pybind11;

namespace py4dgeo {

PYBIND11_MODULE(py4dgeo, m) { m.doc() = "Python Bindings for py4dgeo"; }

} // namespace py4dgeo
