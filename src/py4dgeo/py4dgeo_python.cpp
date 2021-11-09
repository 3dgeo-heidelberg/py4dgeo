#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py4dgeo/compute.hpp"
#include "py4dgeo/epoch.hpp"
#include "py4dgeo/kdtree.hpp"
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

  // The enum class for our memory policy
  py::enum_<MemoryPolicy>(m, "MemoryPolicy", py::arithmetic())
    .value("STRICT", MemoryPolicy::STRICT)
    .value("MINIMAL", MemoryPolicy::MINIMAL)
    .value("COREPOINTS", MemoryPolicy::COREPOINTS)
    .value("RELAXED", MemoryPolicy::RELAXED)
    .export_values();

  // Register a numpy structured type for uncertainty calculation. This allows
  // us to allocate memory in C++ and expose it as a structured numpy array in
  // Python. The given names will be usable in Python.
  PYBIND11_NUMPY_DTYPE(DistanceUncertainty,
                       lodetection,
                       stddev1,
                       num_samples1,
                       stddev2,
                       num_samples2);

  // Also expose the DistanceUncertainty data structure in Python, so that
  // Python fallbacks can use it directly to define their result.
  py::class_<DistanceUncertainty> unc(m, "DistanceUncertainty");
  unc.def(py::init<double, double, IndexType, double, IndexType>(),
          py::arg("lodetection") = 0.0,
          py::arg("stddev1") = 0.0,
          py::arg("num_samples1") = 0,
          py::arg("stddev2") = 0.0,
          py::arg("num_samples2") = 0);

  // The epoch class
  py::class_<Epoch> epoch(m, "Epoch");

  // Initializing with a numpy array prevents the numpy array from being
  // garbage collected as long as the Epoch object is alive
  epoch.def(py::init<EigenPointCloudRef>(), py::keep_alive<1, 2>());

  // We can directly access the point cloud and the kdtree
  epoch.def_readwrite("cloud", &Epoch::cloud);
  epoch.def_readwrite("kdtree", &Epoch::kdtree);

  // Pickling support for the Epoch class
  epoch.def(py::pickle(
    [](const Epoch& self) {
      // Serialize into in-memory stream
      std::stringstream buf;
      self.to_stream(buf);
      return py::bytes(buf.str());
    },
    [](const py::bytes& data) {
      std::stringstream buf(data.cast<std::string>());
      return Epoch::from_stream(buf);
    }));

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
      auto ptr = static_cast<const double*>(qp.request().ptr);

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
  kdtree.def("__getstate__", [](const KDTree&) {
    // If a user pickles KDTree itself, we end up redundantly storing
    // the point cloud itself, because the KDTree is only usable with the
    // cloud (scipy does exactly the same). We solve the problem by asking
    // users to pickle Epoch instead, which is the much cleaner solution.
    throw std::runtime_error{ "Please pickle Epoch instead of KDTree" };
  });

  // The main distance computation function that is the main entry point of M3C2
  m.def(
    "compute_distances",
    [](EigenPointCloudConstRef corepoints,
       double scale,
       const Epoch& epoch1,
       const Epoch& epoch2,
       EigenPointCloudConstRef directions,
       double max_cylinder_length,
       const WorkingSetFinderCallback& workingsetfinder,
       const UncertaintyMeasureCallback& uncertaintycalculator) {
      // Allocate memory for the return types
      DistanceVector distances;
      UncertaintyVector uncertainties;

      compute_distances(corepoints,
                        scale,
                        epoch1,
                        epoch2,
                        directions,
                        max_cylinder_length,
                        distances,
                        uncertainties,
                        workingsetfinder,
                        uncertaintycalculator);

      return std::make_tuple(as_pyarray(std::move(distances)),
                             as_pyarray(std::move(uncertainties)));
    },
    "Compute M3C2 distances");

  // Multiscale direction computation
  m.def("compute_multiscale_directions",
        &compute_multiscale_directions,
        "Compute M3C2 multiscale directions");

  // Callback implementations
  m.def("radius_workingset_finder", &radius_workingset_finder);
  m.def("cylinder_workingset_finder", &cylinder_workingset_finder);
  m.def("no_uncertainty", &no_uncertainty);
  m.def("standard_deviation_uncertainty", &standard_deviation_uncertainty);
}

} // namespace py4dgeo
