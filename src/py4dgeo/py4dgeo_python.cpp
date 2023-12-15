#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef PY4DGEO_WITH_OPENMP
#include <omp.h>
#endif

#include "py4dgeo/compute.hpp"
#include "py4dgeo/epoch.hpp"
#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/py4dgeo.hpp"
#include "py4dgeo/pybind11_numpy_interop.hpp"
#include "py4dgeo/registration.hpp"
#include "py4dgeo/segmentation.hpp"

#include <fstream>
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
                       spread1,
                       num_samples1,
                       spread2,
                       num_samples2);

  // Also expose the DistanceUncertainty data structure in Python, so that
  // Python fallbacks can use it directly to define their result.
  py::class_<DistanceUncertainty> unc(m, "DistanceUncertainty");
  unc.def(py::init<double, double, IndexType, double, IndexType>(),
          py::arg("lodetection") = 0.0,
          py::arg("spread1") = 0.0,
          py::arg("num_samples1") = 0,
          py::arg("spread2") = 0.0,
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

  // Allow updating KDTree from a given file
  kdtree.def("load_index", [](KDTree& self, std::string filename) {
    std::ifstream stream(filename, std::ios::binary | std::ios::in);
    self.loadIndex(stream);
  });

  // Allow dumping KDTree to a file
  kdtree.def("save_index", [](const KDTree& self, std::string filename) {
    std::ofstream stream(filename, std::ios::binary | std::ios::out);
    self.saveIndex(stream);
  });

  // Allow building the KDTree structure
  kdtree.def(
    "build_tree", &KDTree::build_tree, "Trigger building the search tree");

  // Allow invalidating the KDTree structure
  kdtree.def("invalidate", &KDTree::invalidate, "Invalidate the search tree");

  // Give access to the leaf parameter that the tree has been built with
  kdtree.def("leaf_parameter",
             &KDTree::get_leaf_parameter,
             "Retrieve the leaf parameter that the tree has been built with.");

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

  kdtree.def(
    "nearest_neighbors",
    [](const KDTree& self, EigenPointCloudConstRef cloud) {
      KDTree::NearestNeighborsDistanceResult result;
      self.nearest_neighbors_with_distances(cloud, result);

      return std::make_tuple(as_pyarray(std::move(result.first)),
                             as_pyarray(std::move(result.second)));
    },
    "Find nearest neighbors for all points in a cloud!");

  // Pickling support for the KDTree data structure
  kdtree.def("__getstate__", [](const KDTree&) {
    // If a user pickles KDTree itself, we end up redundantly storing
    // the point cloud itself, because the KDTree is only usable with the
    // cloud (scipy does exactly the same). We solve the problem by asking
    // users to pickle Epoch instead, which is the much cleaner solution.
    throw std::runtime_error{
      "Please pickle Epoch instead of KDTree. Otherwise unpickled KDTree does "
      "not know the point cloud."
    };
  });

  // The main distance computation function that is the main entry point of M3C2
  m.def(
    "compute_distances",
    [](EigenPointCloudConstRef corepoints,
       double scale,
       const Epoch& epoch1,
       const Epoch& epoch2,
       EigenNormalSetConstRef directions,
       double max_distance,
       double registration_error,
       const WorkingSetFinderCallback& workingsetfinder,
       const DistanceUncertaintyCalculationCallback& distancecalculator) {
      // Allocate memory for the return types
      DistanceVector distances;
      UncertaintyVector uncertainties;

      {
        // compute_distances may spawn multiple threads that may call Python
        // functions (which requires them to acquire the GIL), so we need to
        // first release the GIL on the main thread before calling
        // compute_distances
        py::gil_scoped_release release_gil;
        compute_distances(corepoints,
                          scale,
                          epoch1,
                          epoch2,
                          directions,
                          max_distance,
                          registration_error,
                          distances,
                          uncertainties,
                          workingsetfinder,
                          distancecalculator);
      }

      return std::make_tuple(as_pyarray(std::move(distances)),
                             as_pyarray(std::move(uncertainties)));
    },
    "The main M3C2 distance calculation algorithm");

  // Multiscale direction computation
  m.def("compute_multiscale_directions",
        &compute_multiscale_directions,
        "Compute M3C2 multiscale directions");

  // Callback parameter structs
  py::class_<WorkingSetFinderParameters> ws_params(
    m, "WorkingSetFinderParameters");
  ws_params.def_property_readonly(
    "epoch", [](const WorkingSetFinderParameters& self) { return self.epoch; });
  ws_params.def_property_readonly(
    "radius",
    [](const WorkingSetFinderParameters& self) { return self.radius; });
  ws_params.def_property_readonly(
    "corepoint",
    [](const WorkingSetFinderParameters& self) { return self.corepoint; });
  ws_params.def_property_readonly(
    "cylinder_axis",
    [](const WorkingSetFinderParameters& self) { return self.cylinder_axis; });
  ws_params.def_property_readonly(
    "max_distance",
    [](const WorkingSetFinderParameters& self) { return self.max_distance; });

  py::class_<DistanceUncertaintyCalculationParameters> d_params(
    m, "DistanceUncertaintyCalculationParameters");
  d_params.def_property_readonly(
    "workingset1", [](const DistanceUncertaintyCalculationParameters& self) {
      return self.workingset1;
    });
  d_params.def_property_readonly(
    "workingset2", [](const DistanceUncertaintyCalculationParameters& self) {
      return self.workingset2;
    });
  d_params.def_property_readonly(
    "corepoint", [](const DistanceUncertaintyCalculationParameters& self) {
      return self.corepoint;
    });
  d_params.def_property_readonly(
    "normal", [](const DistanceUncertaintyCalculationParameters& self) {
      return self.normal;
    });
  d_params.def_property_readonly(
    "registration_error",
    [](const DistanceUncertaintyCalculationParameters& self) {
      return self.registration_error;
    });

  // The ObjectByChange class is used as the return type for spatiotemporal
  // segmentations
  py::class_<ObjectByChange> obc(m, "ObjectByChange");
  obc.def_property_readonly(
    "indices_distances",
    [](const ObjectByChange& self) { return self.indices_distances; });
  obc.def_property_readonly(
    "start_epoch", [](const ObjectByChange& self) { return self.start_epoch; });
  obc.def_property_readonly(
    "end_epoch", [](const ObjectByChange& self) { return self.end_epoch; });
  obc.def_property_readonly(
    "threshold", [](const ObjectByChange& self) { return self.threshold; });
  obc.def(py::pickle(
    [](const ObjectByChange& self) {
      // Serialize into in-memory stream
      std::stringstream buf;

      // Write indices
      std::size_t size = self.indices_distances.size();
      buf.write(reinterpret_cast<const char*>(&size), sizeof(std::size_t));
      for (auto p : self.indices_distances)
        buf.write(reinterpret_cast<const char*>(&p),
                  sizeof(std::pair<IndexType, double>));

      // Write other data
      buf.write(reinterpret_cast<const char*>(&self.start_epoch),
                sizeof(IndexType));
      buf.write(reinterpret_cast<const char*>(&self.end_epoch),
                sizeof(IndexType));
      buf.write(reinterpret_cast<const char*>(&self.threshold), sizeof(double));
      return py::bytes(buf.str());
    },
    [](const py::bytes& data) {
      std::stringstream buf(data.cast<std::string>());
      ObjectByChange obj;

      std::size_t size;
      buf.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));
      std::pair<IndexType, double> buffer;
      for (std::size_t i = 0; i < size; ++i) {
        buf.read(reinterpret_cast<char*>(&buffer),
                 sizeof(std::pair<IndexType, double>));
        obj.indices_distances.insert(buffer);
      }
      buf.read(reinterpret_cast<char*>(&obj.start_epoch), sizeof(IndexType));
      buf.read(reinterpret_cast<char*>(&obj.end_epoch), sizeof(IndexType));
      buf.read(reinterpret_cast<char*>(&obj.threshold), sizeof(double));
      return obj;
    }));

  py::class_<RegionGrowingSeed> rgs(m, "RegionGrowingSeed");
  rgs.def(py::init<IndexType, IndexType, IndexType>(),
          py::arg("index"),
          py::arg("start_epoch"),
          py::arg("end_epoch"));
  rgs.def_property_readonly(
    "index", [](const RegionGrowingSeed& self) { return self.index; });
  rgs.def_property_readonly("start_epoch", [](const RegionGrowingSeed& self) {
    return self.start_epoch;
  });
  rgs.def_property_readonly(
    "end_epoch", [](const RegionGrowingSeed& self) { return self.end_epoch; });
  rgs.def(py::pickle(
    [](const RegionGrowingSeed& self) {
      // Serialize into in-memory stream
      std::stringstream buf;
      buf.write(reinterpret_cast<const char*>(&self.index), sizeof(IndexType));
      buf.write(reinterpret_cast<const char*>(&self.start_epoch),
                sizeof(IndexType));
      buf.write(reinterpret_cast<const char*>(&self.end_epoch),
                sizeof(IndexType));
      return py::bytes(buf.str());
    },
    [](const py::bytes& data) {
      std::stringstream buf(data.cast<std::string>());
      IndexType index, start_epoch, end_epoch;
      buf.read(reinterpret_cast<char*>(&index), sizeof(IndexType));
      buf.read(reinterpret_cast<char*>(&start_epoch), sizeof(IndexType));
      buf.read(reinterpret_cast<char*>(&end_epoch), sizeof(IndexType));
      return RegionGrowingSeed{ index, start_epoch, end_epoch };
    }));

  py::class_<RegionGrowingAlgorithmData> rgwd(m, "RegionGrowingAlgorithmData");
  rgwd.def(py::init<EigenSpatiotemporalArrayConstRef,
                    const Epoch&,
                    double,
                    RegionGrowingSeed,
                    std::vector<double>,
                    std::size_t,
                    std::size_t>(),
           py::arg("data"),
           py::arg("epoch"),
           py::arg("radius"),
           py::arg("seed"),
           py::arg("thresholds"),
           py::arg("min_segments"),
           py::arg("max_segments"));

  py::class_<TimeseriesDistanceFunctionData> tdfd(
    m, "TimeseriesDistanceFunctionData");
  tdfd.def(py::init<EigenTimeSeriesConstRef, EigenTimeSeriesConstRef>(),
           py::arg("ts1"),
           py::arg("ts2"));
  tdfd.def_property_readonly(
    "ts1", [](const TimeseriesDistanceFunctionData& self) { return self.ts1; });
  tdfd.def_property_readonly(
    "ts2", [](const TimeseriesDistanceFunctionData& self) { return self.ts2; });
  tdfd.def_property_readonly(
    "norm1",
    [](const TimeseriesDistanceFunctionData& self) { return self.norm1; });
  tdfd.def_property_readonly(
    "norm2",
    [](const TimeseriesDistanceFunctionData& self) { return self.norm2; });

  py::class_<ChangePointDetectionData> cpdd(m, "ChangePointDetectionData");
  cpdd.def(
    py::
      init<EigenTimeSeriesConstRef, IndexType, IndexType, IndexType, double>(),
    py::arg("ts"),
    py::arg("window_size"),
    py::arg("min_size"),
    py::arg("jump"),
    py::arg("penalty"));

  m.def("transform_pointcloud_inplace",
        [](EigenPointCloudRef cloud,
           const py::array_t<double>& t,
           EigenPointCloudConstRef rp) {
          Transformation trafo;

          auto r = t.unchecked<2>();
          for (IndexType i = 0; i < 4; ++i)
            for (IndexType j = 0; j < 4; ++j)
              trafo(i, j) = r(i, j);

          transform_pointcloud_inplace(cloud, trafo, rp);
        });

  // The main algorithms for the spatiotemporal segmentations
  m.def("region_growing",
        [](const RegionGrowingAlgorithmData& data,
           const TimeseriesDistanceFunction& distance_function) {
          // The region_growing function may call Python callback functions
          py::gil_scoped_release release_gil;
          return region_growing(data, distance_function);
        });
  m.def("change_point_detection", &change_point_detection);

  // Callback implementations
  m.def("radius_workingset_finder", &radius_workingset_finder);
  m.def("cylinder_workingset_finder", &cylinder_workingset_finder);
  m.def("mean_stddev_distance", &mean_stddev_distance);
  m.def("median_iqr_distance", &median_iqr_distance);
  m.def("dtw_distance", &dtw_distance);
  m.def("normalized_dtw_distance", &normalized_dtw_distance);

  // Expose OpenMP threading control
#ifdef PY4DGEO_WITH_OPENMP
  m.def("omp_set_num_threads", &omp_set_num_threads);
  m.def("omp_get_max_threads", &omp_get_max_threads);
#endif
}

} // namespace py4dgeo
