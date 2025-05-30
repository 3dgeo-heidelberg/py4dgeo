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
#include "py4dgeo/octree.hpp"
#include "py4dgeo/py4dgeo.hpp"
#include "py4dgeo/pybind11_numpy_interop.hpp"
#include "py4dgeo/registration.hpp"
#include "py4dgeo/searchtree.hpp"
#include "py4dgeo/segmentation.hpp"

#include <algorithm>
#include <fstream>
#include <optional>
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

  // The enum class for the type of search tree
  py::enum_<SearchTree>(m, "SearchTree")
    .value("KDTreeSearch", SearchTree::KDTree)
    .value("OctreeSearch", SearchTree::Octree)
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

  // We can directly access the point cloud, the kdtree and the octree
  epoch.def_readwrite("_cloud", &Epoch::cloud);
  epoch.def_readwrite("_kdtree", &Epoch::kdtree);
  epoch.def_readwrite("_octree", &Epoch::octree);

  epoch.def(
    "_radius_search",
    [](Epoch& self, py::array_t<double> qp, double radius) {
      // Ensure appropriate search tree has been built
      if (Epoch::get_default_radius_search_tree() == SearchTree::KDTree) {
        if (self.kdtree.get_leaf_parameter() == 0) {
          self.kdtree.build_tree(10);
        }
      } else {
        if (self.octree.get_number_of_points() == 0) {
          self.octree.build_tree();
        }
      }

      // Get a pointer for the query point
      auto ptr = static_cast<const double*>(qp.request().ptr);

      // Now perform the radius search
      RadiusSearchResult result;
      auto radius_search_func = get_radius_search_function(self, radius);
      Eigen::Vector3d query_point(ptr[0], ptr[1], ptr[2]);
      radius_search_func(query_point, result);

      return as_pyarray(std::move(result));
    },
    py::arg("query_point"),
    py::arg("radius"),
    "Perform a radius search");

  // Set and get default search trees
  epoch.def_static(
    "set_default_radius_search_tree",
    [](const std::string& tree_name_input) {
      std::string tree_name = tree_name_input;
      std::transform(
        tree_name.begin(), tree_name.end(), tree_name.begin(), ::tolower);

      if (tree_name == "kdtree") {
        Epoch::set_default_radius_search_tree(SearchTree::KDTree);
      } else if (tree_name == "octree") {
        Epoch::set_default_radius_search_tree(SearchTree::Octree);
      } else {
        throw std::invalid_argument("Unknown search tree type: " +
                                    tree_name_input);
      }
    },
    py::arg("tree_name"),
    "Set the default search tree for radius searches (\"kdtree\" or "
    "\"octree\")");

  epoch.def_static(
    "set_default_nearest_neighbor_tree",
    [](const std::string& tree_name_input) {
      std::string tree_name = tree_name_input;
      std::transform(
        tree_name.begin(), tree_name.end(), tree_name.begin(), ::tolower);

      if (tree_name == "kdtree") {
        Epoch::set_default_nearest_neighbor_tree(SearchTree::KDTree);
      } else if (tree_name == "octree") {
        Epoch::set_default_nearest_neighbor_tree(SearchTree::Octree);
      } else {
        throw std::invalid_argument("Unknown search tree type: " +
                                    tree_name_input);
      }
    },
    py::arg("tree_name"),
    "Set the default search tree for nearest neighbor searches (\"kdtree\" or "
    "\"octree\")");

  epoch.def_static("get_default_radius_search_tree",
                   &Epoch::get_default_radius_search_tree);

  epoch.def_static("get_default_nearest_neighbor_tree",
                   &Epoch::get_default_nearest_neighbor_tree);

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
    "build_tree", &KDTree::build_tree, "Trigger building the search k-d tree");

  // Allow invalidating the KDTree structure
  kdtree.def(
    "invalidate", &KDTree::invalidate, "Invalidate the search k-d tree");

  // Give access to the leaf parameter that the k-d tree has been built with
  kdtree.def(
    "leaf_parameter",
    &KDTree::get_leaf_parameter,
    "Retrieve the leaf parameter that the k-d tree has been built with.");

  // Add all the radius search methods
  kdtree.def(
    "radius_search",
    [](const KDTree& self, py::array_t<double> qp, double radius) {
      // Get a pointer for the query point
      auto ptr = static_cast<const double*>(qp.request().ptr);

      RadiusSearchResult result;
      self.radius_search(ptr, radius, result);

      return as_pyarray(std::move(result));
    },
    "Search point in given radius!");

  kdtree.def(
    "nearest_neighbors",
    [](const KDTree& self, EigenPointCloudConstRef cloud, int k) {
      NearestNeighborsDistanceResult result;
      self.nearest_neighbors_with_distances(cloud, result, k);

      py::array_t<long int> indices_array(result.size());
      py::array_t<double> distances_array(result.size());

      auto indices_array_ptr = indices_array.mutable_data();
      auto distances_array_ptr = distances_array.mutable_data();

      for (size_t i = 0; i < result.size(); ++i) {
        *indices_array_ptr++ = result[i].first[result[i].first.size() - 1];
        *distances_array_ptr++ = result[i].second[result[i].second.size() - 1];
      }

      return std::make_pair(indices_array, distances_array);
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

  // Expose the Octree class
  py::class_<Octree> octree(m, "Octree", py::buffer_protocol());

  // Map __init__ to constructor
  octree.def(py::init<>(&Octree::create));

  // Allow updating Octree from a given file
  octree.def("load_index", [](Octree& self, std::string filename) {
    std::ifstream stream(filename, std::ios::binary | std::ios::in);
    self.loadIndex(stream);
  });

  // Allow dumping Octree to a file
  octree.def("save_index", [](const Octree& self, std::string filename) {
    std::ofstream stream(filename, std::ios::binary | std::ios::out);
    self.saveIndex(stream);
  });

  // Allow building the Octree structure
  octree.def(
    "build_tree", &Octree::build_tree, "Trigger building the search octree");

  // Allow invalidating the Octree structure
  octree.def("invalidate", &Octree::invalidate, "Invalidate the search octree");

  // Allow extraction of number of points
  octree.def("get_number_of_points",
             &Octree::get_number_of_points,
             "Return the number of points in the associated cloud");

  // Allow extraction of bounding box size
  octree.def("get_box_size",
             &Octree::get_box_size,
             "Return the side length of the bounding box");

  // Allow extraction of min point
  octree.def("get_min_point",
             &Octree::get_min_point,
             "Return the minimum point of the bounding box");

  // Allow extraction of max point
  octree.def(
    "get_max_point", &Octree::get_max_point, "Return 8-bit dilated integer");

  // Allow extraction of cell sizes
  octree.def("get_cell_size",
             &Octree::get_cell_size,
             "Return the size of cells at a level of depth");

  // Allow extraction of number of occupied cells per level
  octree.def("get_occupied_cells_per_level",
             &Octree::get_occupied_cells_per_level,
             "Return the number of occupied cells per level of depth");

  // Allow extraction of maximum amount of points
  octree.def("get_max_cell_population_per_level",
             &Octree::get_max_cell_population_per_level,
             "Return the maximum number of points per cell per level of depth");

  // Allow extraction of average amount of points
  octree.def("get_average_cell_population_per_level",
             &Octree::get_average_cell_population_per_level,
             "Return the average number of points per cell per level of depth");

  // Allow extraction of std of amount of points
  octree.def("get_std_cell_population_per_level",
             &Octree::get_std_cell_population_per_level,
             "Return the standard deviation of number of points per cell per "
             "level of depth");

  // Allow extraction of spatial keys
  octree.def("get_spatial_keys",
             &Octree::get_spatial_keys,
             "Return the computed spatial keys");

  // Allow extraction of point indices
  octree.def("get_point_indices",
             &Octree::get_point_indices,
             "Return the sorted point indices");

  // Allow cell index computation
  octree.def(
    "get_cell_index_start",
    &Octree::get_cell_index_start,
    "Return first the index of a cell in the sorted array of point indices "
    "and point spatial keys");

  // Allow cell index computation
  octree.def(
    "get_cell_index_end",
    &Octree::get_cell_index_end,
    "Return the last index of a cell in the sorted array of point indices "
    "and point spatial keys");

  // Allow extraction from points in cell
  octree.def(
    "get_points_indices_from_cells",
    [](const Octree& self,
       const Octree::KeyContainer& keys,
       unsigned int level) {
      RadiusSearchResult result;
      size_t num_points =
        self.get_points_indices_from_cells(keys, level, result);

      // Create NumPy arrays for indices and keys
      py::array_t<Octree::SpatialKey> indices_array(num_points);

      auto indices_ptr = indices_array.mutable_data();

      // Fill the arrays
      for (size_t i = 0; i < num_points; ++i) {
        indices_ptr[i] = result[i];
      }

      return indices_array;
    },
    "Retrieve point indices and spatial keys for a given cell",
    py::arg("spatial_key"),
    py::arg("level"));

  // Allow extraction from points in cell
  octree.def(
    "get_cells_intersected_by_sphere",
    [](const Octree& self,
       const Eigen::Vector3d& query_point,
       double radius,
       unsigned int level) {
      Octree::KeyContainer cells_inside;
      Octree::KeyContainer cells_intersecting;
      self.get_cells_intersected_by_sphere(
        query_point, radius, level, cells_inside, cells_intersecting);

      return py::make_tuple(as_pyarray(std::move(cells_inside)),
                            as_pyarray(std::move(cells_intersecting)));
    },
    "Retrieve the spatial keys of cells intersected by a sphere.",
    py::arg("query_point"),
    py::arg("radius"),
    py::arg("level"));

  // Allow computation of level of depth at which a radius search will be most
  // efficient
  octree.def("find_appropriate_level_for_radius_search",
             &Octree::find_appropriate_level_for_radius_search,
             "Return the level of depth at which a radius search will be most "
             "efficient");

  // Allow radius search with optional depth level specification
  octree.def(
    "radius_search",
    [](const Octree& self,
       Eigen::Ref<const Eigen::Vector3d> query_point,
       double radius,
       std::optional<unsigned int> level) {
      unsigned int lvl =
        level.value_or(self.find_appropriate_level_for_radius_search(radius));

      RadiusSearchResult result;
      self.radius_search(query_point, radius, lvl, result);

      return as_pyarray(std::move(result));
    },
    "Search point in given radius!",
    py::arg("query_point"),
    py::arg("radius"),
    py::arg("level") = std::nullopt);

  // Pickling support for the Octree data structure
  octree.def("__getstate__", [](const Octree&) {
    // If a user pickles Octree itself, we end up redundantly storing
    // the point cloud itself, because the Octree is only usable with the
    // cloud (scipy does exactly the same). We solve the problem by asking
    // users to pickle Epoch instead, which is the much cleaner solution.
    throw std::runtime_error{
      "Please pickle Epoch instead of Octree. Otherwise unpickled Octree does "
      "not know the point cloud."
    };
  });

  // Segment point cloud into a supervoxels
  m.def("segment_pc_in_supervoxels",
        [](Epoch& epoch,
           const KDTree& kdtree,
           EigenNormalSetConstRef normals,
           double resolution,
           int k,
           int minSVPvalue) {
          std::vector<Supervoxel> supervoxels =
            segment_pc(epoch, kdtree, normals, resolution, k, minSVPvalue);

          py::list np_arrays_cloud;
          py::list np_arrays_centroid;
          py::list np_arrays_boundary_points;
          py::list np_arrays_normals;

          for (const auto& sv : supervoxels) {
            // Convert Eigen::MatrixXd to a NumPy array
            auto np_array_cloud = py::array_t<double>(
              sv.cloud.rows() * sv.cloud.cols(), sv.cloud.data());
            auto np_array_normals = py::array_t<double>(
              sv.normals.rows() * sv.normals.cols(), sv.normals.data());
            auto np_array_centroid =
              py::array_t<double>(sv.centroid.size(), sv.centroid.data());
            auto np_array_boundary_points = py::array_t<double>(
              sv.boundary_points.rows() * sv.boundary_points.cols(),
              sv.boundary_points.data());

            // Reshape the arrays to their original shape
            np_array_cloud.resize({ sv.cloud.rows(), sv.cloud.cols() });
            np_array_normals.resize({ sv.normals.rows(), sv.normals.cols() });
            np_array_centroid.resize({ sv.centroid.size() });
            np_array_boundary_points.resize(
              { sv.boundary_points.rows(), sv.boundary_points.cols() });

            np_arrays_cloud.append(np_array_cloud);
            np_arrays_normals.append(np_array_normals);
            np_arrays_centroid.append(np_array_centroid);
            np_arrays_boundary_points.append(np_array_boundary_points);
          }

          return std::make_tuple(np_arrays_cloud,
                                 np_arrays_normals,
                                 np_arrays_centroid,
                                 np_arrays_boundary_points);
        });

  // Perform a transformation of a point cloud using Gauss-Newton method
  m.def("fit_transform_GN",
        [](EigenPointCloudConstRef cloud1,
           EigenPointCloudConstRef cloud2,
           EigenNormalSetConstRef normals) {
          Eigen::Matrix4d transformation =
            fit_transform_GN(cloud1, cloud2, normals);
          return transformation;
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
  m.def(
    "compute_multiscale_directions",
    [](const Epoch& epoch,
       EigenPointCloudConstRef corepoints,
       const std::vector<double>& normal_radii,
       EigenNormalSetConstRef orientation) {
      EigenNormalSet result(corepoints.rows(), 3);
      std::vector<double> used_radii;

      compute_multiscale_directions(
        epoch, corepoints, normal_radii, orientation, result, used_radii);

      return std::make_tuple(std::move(result),
                             as_pyarray(std::move(used_radii)));
    },
    py::arg("epoch"),
    py::arg("corepoints"),
    py::arg("normal_radii"),
    py::arg("orientation"),
    "Compute M3C2 multiscale directions");

  // Corresponence distances computation
  m.def("compute_correspondence_distances",
        &compute_correspondence_distances,
        "Compute correspondence distances");

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
           EigenPointCloudConstRef rp,
           EigenNormalSetRef normals) {
          Transformation trafo;

          auto r = t.unchecked<2>();
          for (IndexType i = 0; i < 4; ++i)
            for (IndexType j = 0; j < 4; ++j)
              trafo(i, j) = r(i, j);

          transform_pointcloud_inplace(cloud, trafo, rp, normals);
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
